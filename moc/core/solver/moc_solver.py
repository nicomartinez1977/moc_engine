# moc/core/solver/moc_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from moc.core.models.network import Network
from moc.core.models.node import Node
from moc.core.models.pipe import Pipe
from moc.core.build.discretize import DiscretizationResult
from moc.core.build.wave_speed import WaveSpeedResult
from moc.core.build.timestep import TimeStepResult


# ============================================================
# Config / Results
# ============================================================

@dataclass(frozen=True)
class MocRunConfig:
    q0_m3s: float
    t_end_s: float

    # Cavitación (umbral en carga H)
    hvap_m: float = -9.5
    enable_cavitation_clamp: bool = True

    # Fricción (fallback si pipe.friction es None)
    default_f_darcy: float = 0.02

    # Condición inicial
    # - "flat": H plano (o lineal si hay 2 estanques)
    # - "steady_friction": H consistente con pérdidas por fricción para Q0
    init_mode: str = "steady_friction"

    # Estabilización suave del término de fricción en el pie:
    # fric = R*q|q| / (1 + alpha*R*|q|)
    friction_alpha: float = 0.5

    # Debug / guard rails
    fail_on_nan: bool = True
    # si |H| o |Q| se van a valores absurdos, cortamos antes de que explote
    max_abs_H: float = 1e6
    max_abs_Q: float = 1e3

    # Si quieres “pinchar” en un step específico para inspección
    debug_raise_at_it: Optional[int] = None


@dataclass(frozen=True)
class MocResults:
    times: np.ndarray             # [nt]
    H: np.ndarray                 # [nt, npts]
    Q: np.ndarray                 # [nt, npts]
    cavitation: np.ndarray        # [nt, npts] bool
    Hmax: np.ndarray              # [npts]
    Hmin: np.ndarray              # [npts]
    meta: Dict[str, object]


# ============================================================
# Helpers
# ============================================================

def _build_chain_order(network: Network) -> Tuple[str, List[Pipe], str]:
    """
    Build a simple linear chain from an upstream node to downstream node.

    v0.1 expects a single chain:
      - each internal node has exactly 1 outgoing pipe (in direction)
      - no cycles
    """
    pipes_by_from: Dict[str, List[Pipe]] = {}
    for p in network.pipes.values():
        pipes_by_from.setdefault(p.node_from, []).append(p)

    candidates: List[str] = []
    for n in network.nodes.values():
        if n.bc_type in ("reservoir", "pump") and n.bc_value is not None and n.uid in pipes_by_from:
            candidates.append(n.uid)

    if not candidates:
        raise ValueError("Cannot build chain: no upstream reservoir/pump with outgoing pipe found.")

    candidates.sort(key=lambda nid: len(pipes_by_from.get(nid, [])))
    start = candidates[0]

    chain: List[Pipe] = []
    visited = set()
    cur = start

    while True:
        visited.add(cur)
        outs = pipes_by_from.get(cur, [])
        if len(outs) == 0:
            end = cur
            break
        if len(outs) > 1:
            raise ValueError(f"Network is not a simple chain: node {cur} has {len(outs)} outgoing pipes.")
        p = outs[0]
        chain.append(p)
        cur = p.node_to
        if cur in visited:
            raise ValueError("Cycle detected. v0.1 solver expects a simple chain (no cycles).")

    if len(chain) == 0:
        raise ValueError("Chain has zero pipes.")

    return start, chain, end


def _event_valve_factor(network: Network, t: float, *, target_uid: Optional[str] = None) -> Optional[float]:
    """
    Returns multiplicative factor phi(t) for downstream flow:
      Q_end = Q0 * phi(t)

    Supported types: valve_close, pump_trip
    Linear ramp 1 -> 0 between [t_start, t_end].
    """
    for e in network.events:
        if e.event_type not in ("valve_close", "pump_trip"):
            continue
        if target_uid is not None and getattr(e, "target_uid", None) != target_uid:
            continue

        if t <= e.t_start:
            return 1.0
        if t >= e.t_end:
            return 0.0
        if e.t_end == e.t_start:
            return 0.0
        return 1.0 - (t - e.t_start) / (e.t_end - e.t_start)

    return None


def _fric_term_vec(R: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    qabs = np.abs(q)
    denom = 1.0 + alpha * R * qabs
    return (R * q * qabs) / denom


def _fric_term(R: float, q: float, alpha: float) -> float:
    qabs = abs(q)
    denom = 1.0 + alpha * R * qabs
    return (R * q * qabs) / denom


# ============================================================
# Main solver (CFL general + interpolación)
# ============================================================

def run_moc_v01(
    network: Network,
    disc: DiscretizationResult,
    ws: WaveSpeedResult,
    ts: TimeStepResult,
    cfg: MocRunConfig,
) -> MocResults:
    """
    MOC v0.1 (single chain) for CFL variable (λ <= 1) using linear interpolation at the
    foot of characteristics.

    For interior node i:
      λL = a_{i-1} dt / dx_{i-1}
      λR = a_{i}   dt / dx_{i}
      left foot is between (i-1, i):  U* = (1-λL)U_i + λL U_{i-1}
      right foot is between (i, i+1): U* = (1-λR)U_i + λR U_{i+1}

    Characteristics (with friction evaluated at the foot, i.e. explicit-in-foot):
      C+ = H* + B_L Q* - fric(R_L, Q*)
      C- = H* - B_R Q* - fric(R_R, Q*)

    Node solve (linear):
      H = 0.5 (C+ + C-)
      Q = (C+ - C-) / (B_L + B_R)

    Boundaries:
      - Upstream fixed H, solve Q from C-
      - Downstream fixed H (if reservoir) else fixed Q (event)
    """
    g = float(network.gravity)
    alpha = float(cfg.friction_alpha)

    up_uid, chain, dn_uid = _build_chain_order(network)
    up_node: Node = network.nodes[up_uid]
    dn_node: Node = network.nodes[dn_uid]

    if up_node.bc_type not in ("reservoir", "pump") or up_node.bc_value is None:
        raise ValueError("Upstream node must be reservoir/pump with bc_value (head in m).")
    H_up = float(up_node.bc_value)

    H_dn: Optional[float] = None
    if dn_node.bc_type == "reservoir" and dn_node.bc_value is not None:
        H_dn = float(dn_node.bc_value)

    dt = float(ts.dt)
    nt = int(np.ceil(cfg.t_end_s / dt)) + 1
    times = np.arange(nt) * dt

    # ------------------------------------------------------------
    # Build global 1D grid by concatenating pipe meshes (no overlap)
    # ------------------------------------------------------------
    mesh_list: List[Tuple[Pipe, object]] = []
    for p in chain:
        m = disc.meshes_by_pipe[p.uid]
        mesh_list.append((p, m))

    n_cells_total = int(sum(m.n_cells for _, m in mesh_list))
    npts = n_cells_total + 1

    # Segment arrays length = n_cells_total
    a_seg = np.zeros(n_cells_total, dtype=float)
    A_seg = np.zeros(n_cells_total, dtype=float)
    D_seg = np.zeros(n_cells_total, dtype=float)
    f_seg = np.zeros(n_cells_total, dtype=float)
    dx_seg = np.zeros(n_cells_total, dtype=float)

    seg0 = 0
    for p, m in mesh_list:
        a = float(ws.wave_speed_by_pipe[p.uid])
        A = float(p.area)
        D = float(p.diameter)
        f = float(p.friction) if p.friction is not None else float(cfg.default_f_darcy)
        dx = float(m.dx)

        n = int(m.n_cells)
        a_seg[seg0:seg0 + n] = a
        A_seg[seg0:seg0 + n] = A
        D_seg[seg0:seg0 + n] = D
        f_seg[seg0:seg0 + n] = f
        dx_seg[seg0:seg0 + n] = dx
        seg0 += n

    B_seg = a_seg / (g * A_seg)  # [s/m^2]
    R_seg = (f_seg * dx_seg) / (2.0 * g * D_seg * (A_seg ** 2))

    # CFL / lambda check
    lam_seg = (a_seg * dt) / dx_seg
    lam_max = float(np.max(lam_seg))
    lam_min = float(np.min(lam_seg))
    if lam_max > 1.0 + 1e-9:
        bad = np.where(lam_seg > 1.0 + 1e-9)[0][:10]
        raise ValueError(
            f"CFL>1 detectado (max λ={lam_max:.6g}). "
            f"Esto requiere dt más chico o dx mayor. Ej índices seg: {bad.tolist()}"
        )

    # ------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------
    Q0 = float(cfg.q0_m3s)
    Q = np.full(npts, Q0, dtype=float)

    init_mode = (cfg.init_mode or "steady_friction").strip().lower()
    if init_mode == "flat":
        if H_dn is not None:
            H = np.linspace(H_up, H_dn, npts, dtype=float) # lineal entre estanques con paso uniforme npts
        else:
            H = np.full(npts, H_up, dtype=float) # plano todo igual a H_up

    elif init_mode == "steady_friction":
        # Integrate losses with Q0
        H = np.zeros(npts, dtype=float) # reserva memoria para el arreglo de H con puros 0, tamaño npts que es n_cells_total + 1
        H[0] = H_up
        for i in range(npts - 1):
            H[i + 1] = H[i] - _fric_term(R_seg[i], Q0, alpha)

        if H_dn is not None:
            H_end_pred = float(H[-1])
            delta = H_end_pred - float(H_dn)
            if abs(delta) > 0.0:
                cumL = np.zeros(npts, dtype=float)
                cumL[1:] = np.cumsum(dx_seg)
                Ltot = float(cumL[-1]) if float(cumL[-1]) > 0 else float(npts - 1)
                H = H - delta * (cumL / Ltot)
                H[0] = H_up
                H[-1] = float(H_dn)
    else:
        raise ValueError(f"init_mode inválido: {cfg.init_mode!r}. Usa 'flat' o 'steady_friction'.")

    # ------------------------------------------------------------
    # Allocate histories
    # ------------------------------------------------------------
    H_hist = np.zeros((nt, npts), dtype=float)
    Q_hist = np.zeros((nt, npts), dtype=float)
    cav = np.zeros((nt, npts), dtype=bool)

    # Precompute index arrays for vectorized interior update
    i = np.arange(1, npts - 1)          # node indices
    sL = i - 1                          # left segment index
    sR = i                              # right segment index

    lamL = lam_seg[sL]                  # λ for left segment
    lamR = lam_seg[sR]                  # λ for right segment

    BL = B_seg[sL]
    BR = B_seg[sR]
    RL = R_seg[sL]
    RR = R_seg[sR]
    denom = (BL + BR)

    # ------------------------------------------------------------
    # Time marching
    # ------------------------------------------------------------
    for it, t in enumerate(times):
        H_hist[it, :] = H
        Q_hist[it, :] = Q
        cav[it, :] = H < cfg.hvap_m

        if cfg.debug_raise_at_it is not None and it == cfg.debug_raise_at_it:
            raise RuntimeError(f"Debug stop at it={it}, t={t:.6f}s")

        Hn = H.copy()
        Qn = Q.copy()

        # -----------------------------
        # Interior nodes (vectorized)
        # -----------------------------
        # Left foot (between i-1 and i)
        HL = (1.0 - lamL) * H[i] + lamL * H[i - 1]
        QL = (1.0 - lamL) * Q[i] + lamL * Q[i - 1]

        # Right foot (between i and i+1)
        HR = (1.0 - lamR) * H[i] + lamR * H[i + 1]
        QR = (1.0 - lamR) * Q[i] + lamR * Q[i + 1]

        Cp = HL + BL * QL - _fric_term_vec(RL, QL, alpha)
        Cm = HR - BR * QR - _fric_term_vec(RR, QR, alpha)

        Hn[i] = 0.5 * (Cp + Cm)
        Qn[i] = (Cp - Cm) / denom

        # Cavitation clamp (vectorized)
        if cfg.enable_cavitation_clamp:
            mask_cav = Hn < cfg.hvap_m
            if np.any(mask_cav):
                Hn[mask_cav] = cfg.hvap_m

        # -----------------------------
        # Upstream boundary: fixed H
        # Use C- from right foot between node 0 and 1
        # -----------------------------
        Hn[0] = H_up
        lam0 = lam_seg[0]
        B0 = B_seg[0]
        R0 = R_seg[0]

        # right foot from x0 + a dt (between 0 and 1)
        H0r = (1.0 - lam0) * H[0] + lam0 * H[1]
        Q0r = (1.0 - lam0) * Q[0] + lam0 * Q[1]

        Cm0 = H0r - B0 * Q0r - _fric_term(R0, Q0r, alpha)
        Qn[0] = (Hn[0] - Cm0) / B0

        # -----------------------------
        # Downstream boundary
        # Use C+ from left foot between node N-2 and N-1
        # -----------------------------
        lamN = lam_seg[-1]
        BN = B_seg[-1]
        RN = R_seg[-1]

        # left foot from xN - a dt (between N and N-1)
        HNl = (1.0 - lamN) * H[-1] + lamN * H[-2]
        QNl = (1.0 - lamN) * Q[-1] + lamN * Q[-2]

        CpN = HNl + BN * QNl - _fric_term(RN, QNl, alpha)

        if H_dn is not None:
            Hn[-1] = H_dn
            Qn[-1] = (CpN - Hn[-1]) / BN
        else:
            phi = _event_valve_factor(network, t, target_uid=dn_uid)
            if phi is None:
                phi = _event_valve_factor(network, t, target_uid=None)

            if phi is not None:
                Qn[-1] = Q0 * float(phi)
            else:
                Qn[-1] = Q0

            Hn[-1] = CpN - BN * Qn[-1]
            if cfg.enable_cavitation_clamp and Hn[-1] < cfg.hvap_m:
                Hn[-1] = cfg.hvap_m

        # -----------------------------
        # Guard rails (raise temprano con info útil)
        # -----------------------------
        if np.max(np.abs(Hn)) > cfg.max_abs_H or np.max(np.abs(Qn)) > cfg.max_abs_Q:
            imaxH = int(np.argmax(np.abs(Hn)))
            imaxQ = int(np.argmax(np.abs(Qn)))
            raise RuntimeError(
                f"Valores absurdos en it={it}, t={t:.6f}s | "
                f"max|H|={float(np.max(np.abs(Hn))):.6g} @i={imaxH}, "
                f"max|Q|={float(np.max(np.abs(Qn))):.6g} @i={imaxQ}"
            )

        if cfg.fail_on_nan:
            if not np.all(np.isfinite(Hn)) or not np.all(np.isfinite(Qn)):
                badH = np.where(~np.isfinite(Hn))[0][:20]
                badQ = np.where(~np.isfinite(Qn))[0][:20]
                raise RuntimeError(f"NaN/Inf at it={it}, t={t:.6f}s, badH={badH}, badQ={badQ}")

        H, Q = Hn, Qn

    Hmax = H_hist.max(axis=0)
    Hmin = H_hist.min(axis=0)

    meta = {
        "dt": dt,
        "nt": nt,
        "npts": npts,
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "up_node_uid": up_uid,
        "dn_node_uid": dn_uid,
        "has_downstream_reservoir": H_dn is not None,
        "assumption": "v0.1 single-chain MOC (CFL variable) with linear interpolation at characteristic feet",
        "init_mode": init_mode,
        "friction_alpha": alpha,
    }

    return MocResults(
        times=times,
        H=H_hist,
        Q=Q_hist,
        cavitation=cav,
        Hmax=Hmax,
        Hmin=Hmin,
        meta=meta,
    )
