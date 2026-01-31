from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from moc.core.build.discretize import DiscretizationResult
from moc.core.build.wave_speed import WaveSpeedResult
from moc.core.models.network import Network


@dataclass(frozen=True)
class TimeStepResult:
    dt: float                      # [s] global timestep
    cfl_by_pipe: Dict[str, float]  # pipe_uid -> CFL (= a*dt/dx)
    limiting_pipe_uid: str         # which pipe controls dt
    dt_reason: str                 # small explanation


def compute_global_dt(
    network: Network,
    disc: DiscretizationResult,
    ws: WaveSpeedResult,
    *,
    cfl_target: float = 1.0,
    dt_max: Optional[float] = None,
    dt_min: float = 1e-6
) -> TimeStepResult:
    """
    Compute global dt using CFL target (default 1.0):
        dt_i = cfl_target * dx_i / a_i
        dt = min(dt_i)

    Optionally clamp by dt_max and dt_min.
    """
    if cfl_target <= 0:
        raise ValueError(f"cfl_target must be > 0, got {cfl_target}")
    if dt_min <= 0:
        raise ValueError(f"dt_min must be > 0, got {dt_min}")

    best_dt = None
    limiting_pipe_uid = None

    # Compute candidate dt for each pipe
    for pipe_uid, mesh in disc.meshes_by_pipe.items():
        if pipe_uid not in ws.wave_speed_by_pipe:
            raise ValueError(f"Missing wave_speed for pipe_uid={pipe_uid}. Run compute_wave_speed first.")
        a = float(ws.wave_speed_by_pipe[pipe_uid])
        dx = float(mesh.dx)
        if a <= 0:
            raise ValueError(f"Invalid wave_speed a={a} for pipe_uid={pipe_uid}")
        if dx <= 0:
            raise ValueError(f"Invalid dx={dx} for pipe_uid={pipe_uid}")

        dt_i = cfl_target * dx / a
        if best_dt is None or dt_i < best_dt:
            best_dt = dt_i
            limiting_pipe_uid = pipe_uid

    if best_dt is None or limiting_pipe_uid is None:
        raise ValueError("Cannot compute dt: no pipes in discretization result.")

    # Clamp
    dt = best_dt
    reason = f"dt = min(cfl_target*dx/a) across pipes (cfl_target={cfl_target})"

    if dt_max is not None:
        dt_max = float(dt_max)
        if dt_max <= 0:
            raise ValueError(f"dt_max must be > 0, got {dt_max}")
        if dt > dt_max:
            dt = dt_max
            reason = f"dt clamped to dt_max={dt_max} (was {best_dt})"

    if dt < dt_min:
        raise ValueError(f"Computed dt={dt} is below dt_min={dt_min}. Increase dx or verify wave speeds.")

    # CFL per pipe
    cfl_by_pipe: Dict[str, float] = {}
    for pipe_uid, mesh in disc.meshes_by_pipe.items():
        a = float(ws.wave_speed_by_pipe[pipe_uid])
        dx = float(mesh.dx)
        cfl_by_pipe[pipe_uid] = a * dt / dx

    return TimeStepResult(
        dt=dt,
        cfl_by_pipe=cfl_by_pipe,
        limiting_pipe_uid=limiting_pipe_uid,
        dt_reason=reason
    )
