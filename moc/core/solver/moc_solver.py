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


@dataclass(frozen=True)
class MocRunConfig:
    q0_m3s: float
    t_end_s: float
    hvap_m: float = -9.5          # head threshold for cavitation detection (relative)
    enable_cavitation_clamp: bool = True
    default_f_darcy: float = 0.02


@dataclass(frozen=True)
class MocResults:
    times: np.ndarray             # [nt]
    H: np.ndarray                 # [nt, npts]
    Q: np.ndarray                 # [nt, npts]
    cavitation: np.ndarray        # [nt, npts] boolean
    Hmax: np.ndarray              # [npts]
    Hmin: np.ndarray              # [npts]
    meta: Dict[str, object]


# -----------------------------
# Helpers
# -----------------------------
def _find_reservoir_nodes(network: Network) -> List[Node]:
    return [n for n in network.nodes.values() if n.bc_type == "reservoir" and n.bc_value is not None]


def _build_chain_order(network: Network) -> Tuple[str, List[Pipe], str]:
    """
    Build a linear chain order (single path) from an upstream node to downstream node.

    Strategy:
    - pick upstream as a reservoir node with at least one outgoing pipe (node_from)
    - follow unique outgoing connections until no more pipes
    - requires exactly one outgoing pipe per internal node in the chain
    """
    pipes_by_from: Dict[str, List[Pipe]] = {}
    for p in network.pipes.values():
        pipes_by_from.setdefault(p.node_from, []).append(p)

    # Candidate upstream nodes: reservoirs that have outgoing pipes
    candidates = []
    for n in network.nodes.values():
        if n.bc_type == "reservoir" and n.bc_value is not None and n.uid in pipes_by_from:
            candidates.append(n.uid)

    if not candidates:
        raise ValueError("Cannot build chain: no reservoir node with outgoing pipe found (need upstream reservoir).")

    # Prefer reservoir with exactly one outgoing pipe
    candidates.sort(key=lambda nid: len(pipes_by_from.get(nid, [])))
    start = candidates[0]

    chain: List[Pipe] = []
    visited_nodes = set()
    cur = start

    while True:
        visited_nodes.add(cur)
        outs = pipes_by_from.get(cur, [])
        if len(outs) == 0:
            # end reached
            end = cur
            break
        if len(outs) > 1:
            raise ValueError(
                f"Network is not a simple chain: node(uid={cur}) has {len(outs)} outgoing pipes."
            )
        p = outs[0]
        chain.append(p)
        cur = p.node_to
        if cur in visited_nodes:
            raise ValueError("Cycle detected while building chain. This v0.1 solver expects a simple chain.")

    if len(chain) == 0:
        raise ValueError("Chain has zero pipes.")

    return start, chain, end


def _event_valve_factor(network: Network, t: float) -> Optional[float]:
    """
    Returns a multiplicative factor phi(t) for downstream flow if an applicable event exists.
    Supported event types (v0.1):
      - valve_close
      - pump_trip
    Interpretation: Q_end = Q0 * phi(t), where phi transitions 1 -> 0 over [t_start, t_end]
    """
    # In v0.1 we apply only one active event if exists (first match).
    for e in network.events:
        if e.event_type not in ("valve_close", "pump_trip"):
            continue
        if t <= e.t_start:
            return 1.0
        if t >= e.t_end:
            return 0.0
        # linear closure
        return 1.0 - (t - e.t_start) / (e.t_end - e.t_start)
    return None


# -----------------------------
# Main solver
# -----------------------------
def run_moc_v01(
    network: Network,
    disc: DiscretizationResult,
    ws: WaveSpeedResult,
    ts: TimeStepResult,
    cfg: MocRunConfig,
) -> MocResults:
    """
    Minimal MOC solver for a single chain of pipes.

    - Upstream boundary: reservoir head fixed (H = H_res)
    - Downstream boundary:
        * if downstream node is reservoir: head fixed
        * else if events include valve_close or pump_trip: enforce Q_end = Q0*phi(t)
        * else: keep Q_end = Q0 (not realistic, but keeps model defined)

    Cavitation:
    - detect H < Hvap
    - optionally clamp H to Hvap (no cavity volume model yet)
    """
    g = network.gravity

    up_uid, chain, dn_uid = _build_chain_order(network)
    up_node = network.nodes[up_uid]
    dn_node = network.nodes[dn_uid]

    if up_node.bc_type != "reservoir" or up_node.bc_value is None:
        raise ValueError("Upstream node must be a reservoir with bc_value (head).")

    H_up = float(up_node.bc_value)

    H_dn = None
    if dn_node.bc_type == "reservoir" and dn_node.bc_value is not None:
        H_dn = float(dn_node.bc_value)

    dt = float(ts.dt)
    nt = int(np.ceil(cfg.t_end_s / dt)) + 1
    times = np.arange(nt) * dt

    # Build global grid by concatenating pipe meshes (avoid double-count at junctions)
    # We'll assign segment properties per segment index.
    # Global points: npts = sum(n_cells) + 1
    mesh_list = []
    for p in chain:
        m = disc.meshes_by_pipe[p.uid]
        mesh_list.append((p, m))

    n_cells_total = sum(m.n_cells for _, m in mesh_list)
    npts = n_cells_total + 1

    # Segment arrays length = n_cells_total
    a_seg = np.zeros(n_cells_total)
    A_seg = np.zeros(n_cells_total)
    D_seg = np.zeros(n_cells_total)
    f_seg = np.zeros(n_cells_total)
    dx_seg = np.zeros(n_cells_total)

    # Fill per pipe segments
    seg0 = 0
    for p, m in mesh_list:
        a = float(ws.wave_speed_by_pipe[p.uid])
        A = float(p.area)
        D = float(p.diameter)
        f = float(p.friction) if p.friction is not None else float(cfg.default_f_darcy)
        dx = float(m.dx)

        n = m.n_cells
        a_seg[seg0:seg0 + n] = a
        A_seg[seg0:seg0 + n] = A
        D_seg[seg0:seg0 + n] = D
        f_seg[seg0:seg0 + n] = f
        dx_seg[seg0:seg0 + n] = dx
        seg0 += n

    # MOC coefficients per segment
    B_seg = a_seg / (g * A_seg)  # [s/m^2]
    R_seg = (f_seg * dx_seg) / (2.0 * g * D_seg * (A_seg ** 2))  # multiplies Q|Q|

    # Initial conditions: Q uniform, H linear between reservoirs if both exist, else flat at H_up
    Q0 = float(cfg.q0_m3s)
    Q = np.full(npts, Q0, dtype=float)

    if H_dn is not None:
        H = np.linspace(H_up, H_dn, npts, dtype=float)
    else:
        H = np.full(npts, H_up, dtype=float)

    H_hist = np.zeros((nt, npts), dtype=float)
    Q_hist = np.zeros((nt, npts), dtype=float)
    cav = np.zeros((nt, npts), dtype=bool)

    for it, t in enumerate(times):
        H_hist[it, :] = H
        Q_hist[it, :] = Q

        cav[it, :] = H < cfg.hvap_m

        # next step
        Hn = H.copy()
        Qn = Q.copy()

        # interior nodes (1..npts-2)
        for i in range(1, npts - 1):
            # left segment is i-1, right segment is i
            B_L = B_seg[i - 1]
            R_L = R_seg[i - 1]
            B_R = B_seg[i] if i < len(B_seg) else B_seg[-1]
            R_R = R_seg[i] if i < len(R_seg) else R_seg[-1]

            Cplus = H[i - 1] + B_L * Q[i - 1] - R_L * Q[i - 1] * abs(Q[i - 1])
            Cminus = H[i + 1] - B_R * Q[i + 1] - R_R * Q[i + 1] * abs(Q[i + 1])

            # average B for update (starter approach)
            B_avg = 0.5 * (B_L + B_R)

            Hn[i] = 0.5 * (Cplus + Cminus)
            Qn[i] = (Cplus - Cminus) / (2.0 * B_avg)

            if cfg.enable_cavitation_clamp and Hn[i] < cfg.hvap_m:
                Hn[i] = cfg.hvap_m

        # upstream boundary: fixed head
        Hn[0] = H_up
        # get Q from C- using segment 0
        B0 = B_seg[0]
        R0 = R_seg[0]
        Cminus0 = H[1] - B0 * Q[1] - R0 * Q[1] * abs(Q[1])
        Qn[0] = (Hn[0] - Cminus0) / B0

        # downstream boundary
        if H_dn is not None:
            Hn[-1] = H_dn
            Bn = B_seg[-1]
            Rn = R_seg[-1]
            CplusN = H[-2] + Bn * Q[-2] - Rn * Q[-2] * abs(Q[-2])
            Qn[-1] = (CplusN - Hn[-1]) / Bn
        else:
            # Apply event-based closure if exists
            phi = _event_valve_factor(network, t)
            if phi is not None:
                Qn[-1] = Q0 * phi
                # compute H using C+ (from last interior point)
                Bn = B_seg[-1]
                Rn = R_seg[-1]
                CplusN = H[-2] + Bn * Q[-2] - Rn * Q[-2] * abs(Q[-2])
                Hn[-1] = CplusN - Bn * Qn[-1]
            else:
                # fallback: keep Q constant
                Qn[-1] = Q0
                Bn = B_seg[-1]
                Rn = R_seg[-1]
                CplusN = H[-2] + Bn * Q[-2] - Rn * Q[-2] * abs(Q[-2])
                Hn[-1] = CplusN - Bn * Qn[-1]

            if cfg.enable_cavitation_clamp and Hn[-1] < cfg.hvap_m:
                Hn[-1] = cfg.hvap_m

        H, Q = Hn, Qn

    Hmax = H_hist.max(axis=0)
    Hmin = H_hist.min(axis=0)

    meta = {
        "dt": dt,
        "npts": npts,
        "up_node_uid": up_uid,
        "dn_node_uid": dn_uid,
        "has_downstream_reservoir": H_dn is not None,
        "assumption": "v0.1 single-chain MOC solver",
    }

    return MocResults(times=times, H=H_hist, Q=Q_hist, cavitation=cav, Hmax=Hmax, Hmin=Hmin, meta=meta)
