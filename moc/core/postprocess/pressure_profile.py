from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from moc.core.postprocess.summary import ProfileSummary
from moc.core.postprocess.geometry import GeometryProfile


@dataclass(frozen=True)
class PressureProfileSummary:
    s_m: np.ndarray
    z_m: np.ndarray

    hmin_m: np.ndarray
    hmax_m: np.ndarray

    pmin_mca: np.ndarray      # (Hmin - z)
    pmax_mca: np.ndarray      # (Hmax - z)
    pmin_bar: np.ndarray
    pmax_bar: np.ndarray

    cav_any: np.ndarray
    meta: Dict[str, object]


def build_pressure_profile(
    prof: ProfileSummary,
    geom: GeometryProfile,
    *,
    rho: float = 1000.0,
    g: float = 9.81
) -> PressureProfileSummary:
    if len(prof.s_m) != len(geom.s_m):
        raise ValueError("El perfil hidráulico y geométrico no tienen el mismo tamaño.")
    if np.max(np.abs(prof.s_m - geom.s_m)) > 1e-6:
        raise ValueError("El eje s_m no coincide entre perfil hidráulico y geométrico.")

    pmin_mca = prof.hmin_m - geom.z_m
    pmax_mca = prof.hmax_m - geom.z_m
    pmin_bar = (rho * g * pmin_mca) / 1e5
    pmax_bar = (rho * g * pmax_mca) / 1e5

    meta = dict(prof.meta)
    meta.update({"pressure_basis": "gauge_head = H - z"})

    return PressureProfileSummary(
        s_m=prof.s_m,
        z_m=geom.z_m,
        hmin_m=prof.hmin_m,
        hmax_m=prof.hmax_m,
        pmin_mca=pmin_mca,
        pmax_mca=pmax_mca,
        pmin_bar=pmin_bar,
        pmax_bar=pmax_bar,
        cav_any=prof.cav_any,
        meta=meta
    )
