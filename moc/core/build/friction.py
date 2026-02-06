from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import math

from moc.core.models.network import Network
from moc.core.models.pipe import Pipe


@dataclass(frozen=True)
class FrictionResult:
    f_by_pipe: Dict[str, float]
    Re_by_pipe: Dict[str, float]
    V_by_pipe: Dict[str, float]
    rr_by_pipe: Dict[str, float]  # eps/D
    meta: Dict[str, object]


def _swamee_jain_f(eps_over_D: float, Re: float) -> float:
    # Swamee–Jain (turbulento), válida para Re>~5000 (pero funciona bien como aproximación general)
    # f = 0.25 / [log10( eps/(3.7D) + 5.74/Re^0.9 )]^2
    term = (eps_over_D / 3.7) + (5.74 / (Re ** 0.9))
    return 0.25 / (math.log10(term) ** 2)


def _darcy_f(Re: float, eps_over_D: float) -> float:
    if Re <= 0:
        return float("nan")
    if Re < 2000.0:
        return 64.0 / Re
    # transición simple (sin complicarnos con Churchill por ahora)
    return _swamee_jain_f(eps_over_D, Re)


def compute_friction_from_q0(
    network: Network,
    *,
    q0_m3s: float,
    nu_m2s: float,
    default_eps_m: float = 1e-5,
    override_existing: bool = False,
) -> Tuple[Network, FrictionResult]:
    """
    Calcula f (Darcy) por tubería asumiendo un caudal operativo único Q0 en toda la cadena.

    Inputs:
      - q0_m3s: caudal operativo [m3/s]
      - nu_m2s: viscosidad cinemática del fluido [m2/s]
      - default_eps_m: rugosidad por defecto si Pipe.roughness es None [m]
      - override_existing: si True, recalcula aunque Pipe.friction ya exista

    Devuelve:
      - new Network (no muta original) con Pipe.friction completada
      - FrictionResult con Re, V, rr y f por pipe
    """
    if nu_m2s <= 0:
        raise ValueError(f"nu_m2s debe ser > 0. Se recibió: {nu_m2s!r}")

    f_by_pipe: Dict[str, float] = {}
    Re_by_pipe: Dict[str, float] = {}
    V_by_pipe: Dict[str, float] = {}
    rr_by_pipe: Dict[str, float] = {}

    new_pipes: Dict[str, Pipe] = {}

    for puid, p in network.pipes.items():
        if (p.friction is not None) and (not override_existing):
            # respetar fricción existente
            new_pipes[puid] = p
            # igual reportamos
            A = float(p.area)
            D = float(p.diameter)
            V = (q0_m3s / A) if A > 0 else float("nan")
            Re = (V * D / nu_m2s) if (D > 0 and nu_m2s > 0) else float("nan")
            eps = float(p.roughness) if getattr(p, "roughness", None) is not None else float(default_eps_m)
            rr = (eps / D) if D > 0 else float("nan")
            V_by_pipe[puid] = V
            Re_by_pipe[puid] = Re
            rr_by_pipe[puid] = rr
            f_by_pipe[puid] = float(p.friction)
            continue

        A = float(p.area)
        D = float(p.diameter)

        if A <= 0 or D <= 0:
            raise ValueError(f"Pipe(uid={puid}) tiene área/diámetro inválido: A={A}, D={D}")

        V = q0_m3s / A
        Re = V * D / nu_m2s

        eps = float(p.roughness) if getattr(p, "roughness", None) is not None else float(default_eps_m)
        rr = eps / D

        f = _darcy_f(Re, rr)

        V_by_pipe[puid] = V
        Re_by_pipe[puid] = Re
        rr_by_pipe[puid] = rr
        f_by_pipe[puid] = f

        new_pipes[puid] = replace(p, friction=f)

    new_network = Network(
        nodes=network.nodes,
        pipes=new_pipes,
        events=network.events,
        gravity=network.gravity,
        density=network.density,
    )

    result = FrictionResult(
        f_by_pipe=f_by_pipe,
        Re_by_pipe=Re_by_pipe,
        V_by_pipe=V_by_pipe,
        rr_by_pipe=rr_by_pipe,
        meta={
            "method": "Darcy f via laminar (64/Re) + Swamee-Jain (turbulent)",
            "q0_m3s": q0_m3s,
            "nu_m2s": nu_m2s,
            "default_eps_m": default_eps_m,
            "override_existing": override_existing,
        },
    )


    return new_network, result
