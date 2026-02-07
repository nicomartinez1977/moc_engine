# moc/core/hidraulics/friction.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math


@dataclass(frozen=True)
class FrictionResult:
    f_by_id: Dict[str, float]
    Re_by_id: Dict[str, float]
    V_by_id: Dict[str, float]
    rr_by_id: Dict[str, float]  # eps/D
    meta: Dict[str, object]


def swamee_jain_f(*, eps_over_D: float, Re: float) -> float:
    """
    Swamee–Jain (turbulento), válida ~Re>5000 (pero buena aproximación general).
    f = 0.25 / [log10( eps/(3.7D) + 5.74/Re^0.9 )]^2
    """
    if Re <= 0:
        return float("nan")
    term = (eps_over_D / 3.7) + (5.74 / (Re ** 0.9))
    return 0.25 / (math.log10(term) ** 2)


def darcy_f(*, Re: float, eps_over_D: float) -> float:
    """Factor de fricción Darcy: laminar (64/Re) + transición simple a Swamee–Jain."""
    if Re <= 0:
        return float("nan")
    if Re < 2000.0:
        return 64.0 / Re
    return swamee_jain_f(eps_over_D=eps_over_D, Re=Re)


def velocity_from_q(*, q_m3s: float, area_m2: float) -> float:
    return (q_m3s / area_m2) if area_m2 > 0 else float("nan")


def reynolds(*, V_m_s: float, D_m: float, nu_m2s: float) -> float:
    return (V_m_s * D_m / nu_m2s) if (D_m > 0 and nu_m2s > 0) else float("nan")


def rr_eps_over_D(*, eps_m: float, D_m: float) -> float:
    return (eps_m / D_m) if D_m > 0 else float("nan")
