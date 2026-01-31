from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from moc.core.models.network import Network
from moc.core.models.pipe import Pipe


@dataclass(frozen=True)
class WaveSpeedResult:
    """Computed (or adopted) wave speed per pipe uid."""
    wave_speed_by_pipe: Dict[str, float]   # pipe_uid -> a [m/s]


def compute_wave_speed(
    network: Network,
    *,
    K_pa: float = 2.2e9,        # bulk modulus of water [Pa] ~2.0–2.2 GPa
    rho: float = 1000.0,        # density [kg/m3]
    default_E_pa: float = 2.0e11,  # steel as conservative default if unknown [Pa]
    default_nu: float = 0.30,      # Poisson ratio typical for metals
    default_thickness_m: Optional[float] = None,  # if None, require thickness when computing
) -> WaveSpeedResult:
    """
    Computes wave speed a for each pipe.
    If pipe.wave_speed is already provided, it is used as-is.

    Formula (classic elastic pipe, simplified):
      a = 1 / sqrt( rho * ( 1/K + (D/(E*e)) * (1 - nu^2) ) )

    Notes:
    - If thickness is missing and default_thickness_m is None, raises ValueError.
    - If elasticity is missing, uses default_E_pa.
    - If poisson is missing, uses default_nu.
    """

    out: Dict[str, float] = {}

    for uid, p in network.pipes.items():
        if p.wave_speed is not None:
            a = float(p.wave_speed)
            if a <= 0:
                raise ValueError(f"Pipe(uid={uid}, name={p.name}) has non-positive wave_speed={a}")
            out[uid] = a
            continue

        D = float(p.diameter)
        if D <= 0:
            raise ValueError(f"Pipe(uid={uid}, name={p.name}) has invalid diameter={D}")

        e = p.thickness if p.thickness is not None else default_thickness_m
        if e is None:
            raise ValueError(
                f"Pipe(uid={uid}, name={p.name}) wave_speed missing and thickness is missing. "
                f"Provide 'espesor_m' in Excel or set default_thickness_m."
            )
        e = float(e)
        if e <= 0:
            raise ValueError(f"Pipe(uid={uid}, name={p.name}) has invalid thickness={e}")

        E = float(p.elasticity) if p.elasticity is not None else float(default_E_pa)
        if E <= 0:
            raise ValueError(f"Pipe(uid={uid}, name={p.name}) has invalid elasticity E={E}")

        nu = float(p.poisson) if p.poisson is not None else float(default_nu)
        if not (-0.5 < nu < 0.5):
            raise ValueError(f"Pipe(uid={uid}, name={p.name}) has invalid poisson nu={nu}")

        # wave speed
        term = (1.0 / K_pa) + (D / (E * e)) * (1.0 - nu**2)
        a = 1.0 / (rho * term) ** 0.5

        if a <= 0 or a != a:
            raise ValueError(f"Computed invalid wave speed for Pipe(uid={uid}, name={p.name}): a={a}")

        out[uid] = a

    return WaveSpeedResult(wave_speed_by_pipe=out)
