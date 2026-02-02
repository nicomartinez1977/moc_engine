from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from moc.core.models.network import Network
from moc.core.build.discretize import DiscretizationResult, PipeMesh
from moc.core.solver.moc_solver import MocResults


@dataclass(frozen=True)
class ProfileSummary:
    s_m: np.ndarray          # distance accumulated [m], size npts
    hmin_m: np.ndarray       # size npts
    hmax_m: np.ndarray       # size npts
    cav_any: np.ndarray      # bool size npts (any time cavitated at point)
    meta: Dict[str, object]


@dataclass(frozen=True)
class PipeRange:
    pipe_uid: str
    s_start_m: float
    s_end_m: float
    i_start: int             # global point index start
    i_end: int               # global point index end (inclusive)


def _build_chain_order(network: Network) -> Tuple[str, List[str], str]:
    """
    Returns (up_node_uid, [pipe_uid ordered], dn_node_uid) for a simple chain.
    Must match solver's chain assumption.
    """
    pipes_by_from: Dict[str, List[str]] = {}
    for puid, p in network.pipes.items():
        pipes_by_from.setdefault(p.node_from, []).append(puid)

    # upstream candidates: reservoir with outgoing pipes
    candidates = []
    for nuid, n in network.nodes.items():
        if n.bc_type in ("reservoir", "pump") and n.bc_value is not None and nuid in pipes_by_from:
            candidates.append(nuid)

    if not candidates:
        raise ValueError("El nodo inicial debe ser 'estanque' o 'bomba' con bc_value (carga en m).")

    candidates.sort(key=lambda nid: len(pipes_by_from.get(nid, [])))
    start = candidates[0]

    pipe_order: List[str] = []
    visited = set()
    cur = start

    while True:
        visited.add(cur)
        outs = pipes_by_from.get(cur, [])
        if len(outs) == 0:
            end = cur
            break
        if len(outs) > 1:
            raise ValueError("Network is not a simple chain (node has multiple outgoing pipes).")
        puid = outs[0]
        pipe_order.append(puid)
        cur = network.pipes[puid].node_to
        if cur in visited:
            raise ValueError("Cycle detected while building chain.")

    if not pipe_order:
        raise ValueError("Chain has zero pipes.")
    return start, pipe_order, end


def build_pipe_ranges_accumulated(network: Network, disc: DiscretizationResult) -> List[PipeRange]:
    """
    Builds global point indexing and accumulated distance ranges for the solver-style concatenated grid.

    Convention:
    - Global points are concatenated across ordered pipes, avoiding double-count at junctions.
    - If a pipe has n_cells, it contributes n_cells segments and n_cells points after the first shared junction;
      total global points = sum(n_cells) + 1.
    """
    _, pipe_uids, _ = _build_chain_order(network)

    ranges: List[PipeRange] = []
    s_acc = 0.0
    i0 = 0  # global point index start for first pipe

    for puid in pipe_uids:
        mesh = disc.meshes_by_pipe[puid]
        L = float(network.pipes[puid].length)

        # This pipe contributes n_cells segments and n_cells points beyond its start point
        i_start = i0
        i_end = i0 + mesh.n_cells  # inclusive index of last point of this pipe
        s_start = s_acc
        s_end = s_acc + L

        ranges.append(PipeRange(
            pipe_uid=puid,
            s_start_m=s_start,
            s_end_m=s_end,
            i_start=i_start,
            i_end=i_end,
        ))

        # next pipe starts at this pipe's last point
        i0 = i_end
        s_acc = s_end

    return ranges


def summarize_profile_accumulated(
    network: Network,
    disc: DiscretizationResult,
    results: MocResults,
) -> ProfileSummary:
    """
    Produces accumulated-distance profile arrays aligned with MocResults global points.
    """
    ranges = build_pipe_ranges_accumulated(network, disc)

    npts = results.H.shape[1]
    s = np.zeros(npts, dtype=float)

    # Fill accumulated distance using each pipe mesh local x
    for r in ranges:
        mesh: PipeMesh = disc.meshes_by_pipe[r.pipe_uid]
        # local x is 0..L with length n_cells+1
        x_local = np.array(mesh.x, dtype=float)
        # global indices slice
        i_start, i_end = r.i_start, r.i_end
        # length of slice
        expected_len = i_end - i_start + 1
        if expected_len != len(x_local):
            raise ValueError(
                f"Mesh length mismatch for pipe {r.pipe_uid}: "
                f"expected {expected_len}, got {len(x_local)}"
            )
        s[i_start:i_end + 1] = r.s_start_m + x_local

    cav_any = results.cavitation.any(axis=0)

    meta = dict(results.meta)
    meta.update({
        "profile_type": "accumulated_distance",
        "s_total_m": float(s[-1]) if len(s) else 0.0,
        "npts": int(npts),
        "n_pipes": len(ranges),
    })

    return ProfileSummary(
        s_m=s,
        hmin_m=results.Hmin.copy(),
        hmax_m=results.Hmax.copy(),
        cav_any=cav_any,
        meta=meta,
    )
