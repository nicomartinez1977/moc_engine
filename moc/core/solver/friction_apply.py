# moc/core/solver/friction_apply.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

from moc.core.models.network import Network
from moc.core.models.pipe import Pipe

from moc.core.hydraulics.friction import (
    FrictionResult,
    darcy_f,
    velocity_from_q,
    reynolds,
    rr_eps_over_D,
)


def compute_friction_from_q0(
    network: Network,
    *,
    q0_m3s: float,
    nu_m2s: float,
    default_eps_m: float = 1e-5,
    override_existing: bool = False,
) -> Tuple[Network, FrictionResult]:
    """
    Calcula f (Darcy) por tubería asumiendo un caudal operativo único Q0.
    Devuelve un Network nuevo (no muta el original) + reporte.
    """
    if nu_m2s <= 0:
        raise ValueError(f"nu_m2s debe ser > 0. Se recibió: {nu_m2s!r}")

    f_by_id: Dict[str, float] = {}
    Re_by_id: Dict[str, float] = {}
    V_by_id: Dict[str, float] = {}
    rr_by_id: Dict[str, float] = {}

    new_pipes: Dict[str, Pipe] = {}

    for puid, p in network.pipes.items():
        A = float(p.area)
        D = float(p.diameter)
        if A <= 0 or D <= 0:
            raise ValueError(f"Pipe(uid={puid}) tiene área/diámetro inválido: A={A}, D={D}")

        V = velocity_from_q(q_m3s=q0_m3s, area_m2=A)
        Re = reynolds(V_m_s=V, D_m=D, nu_m2s=nu_m2s)

        eps = float(p.roughness) if getattr(p, "roughness", None) is not None else float(default_eps_m)
        rr = rr_eps_over_D(eps_m=eps, D_m=D)

        if (p.friction is not None) and (not override_existing):
            f = float(p.friction)
            new_pipes[puid] = p
        else:
            f = darcy_f(Re=Re, eps_over_D=rr)
            new_pipes[puid] = replace(p, friction=f)

        V_by_id[puid] = V
        Re_by_id[puid] = Re
        rr_by_id[puid] = rr
        f_by_id[puid] = f

    new_network = Network(
        nodes=network.nodes,
        pipes=new_pipes,
        events=network.events,
        gravity=network.gravity,
        density=network.density,
    )

    result = FrictionResult(
        f_by_id=f_by_id,
        Re_by_id=Re_by_id,
        V_by_id=V_by_id,
        rr_by_id=rr_by_id,
        meta={
            "method": "Darcy f via laminar (64/Re) + Swamee-Jain (turbulent)",
            "q0_m3s": q0_m3s,
            "nu_m2s": nu_m2s,
            "default_eps_m": default_eps_m,
            "override_existing": override_existing,
        },
    )

    return new_network, result
