from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from moc.core.models.network import Network
from moc.core.models.pipe import Pipe


@dataclass(frozen=True)
class PipeMesh:
    pipe_uid: str
    n_cells: int
    dx: float
    # coordinates along the pipe axis (0..L), size = n_cells+1
    x: Tuple[float, ...]


@dataclass(frozen=True)
class DiscretizationResult:
    meshes_by_pipe: Dict[str, PipeMesh]


def _choose_dx(pipe: Pipe, *, default_dx_m: float) -> float:
    """
    Picks dx for a pipe:
    - use pipe.dx_target if provided
    - else use default_dx_m
    """
    if pipe.dx_target is not None:
        dx = float(pipe.dx_target)
    else:
        dx = float(default_dx_m)

    if dx <= 0:
        raise ValueError(f"Invalid dx_target/default_dx_m: {dx} for pipe '{pipe.name}'")
    return dx


def discretize_network(
    network: Network,
    *,
    default_dx_m: float = 10.0,
    min_cells_per_pipe: int = 5,
    max_cells_per_pipe: int = 5000,
) -> DiscretizationResult:
    """
    Discretize each pipe into cells with constant dx (per pipe).
    - dx is chosen from pipe.dx_target or default_dx_m
    - n_cells is rounded to an integer and dx is adjusted to fit exactly length

    Returns meshes for each pipe.
    """
    meshes: Dict[str, PipeMesh] = {}

    for puid, p in network.pipes.items():
        L = float(p.length)
        if L <= 0:
            raise ValueError(f"Pipe(uid={puid}, name={p.name}) has invalid length {L}")

        dx_raw = _choose_dx(p, default_dx_m=default_dx_m)

        # choose n_cells
        n_cells = int(round(L / dx_raw))
        if n_cells < min_cells_per_pipe:
            n_cells = min_cells_per_pipe

        if n_cells > max_cells_per_pipe:
            raise ValueError(
                f"Pipe(uid={puid}, name={p.name}) would create too many cells: {n_cells}. "
                f"Increase dx_target or default_dx_m."
            )

        dx = L / n_cells  # adjust dx to exactly match L

        # coordinates
        x = tuple(i * dx for i in range(n_cells + 1))

        meshes[puid] = PipeMesh(
            pipe_uid=puid,
            n_cells=n_cells,
            dx=dx,
            x=x,
        )

    return DiscretizationResult(meshes_by_pipe=meshes)
