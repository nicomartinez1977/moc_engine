# moc/core/hydraulics/headloss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


HeadlossModelName = Literal["darcy_weisbach", "chezy"]


@dataclass(frozen=True)
class HeadlossModel:
    model: HeadlossModelName = "darcy_weisbach"
    include_minor: bool = False
    g_m_s2: float = 9.80665

    def segment_R(
        self,
        *,
        D_m: float,
        L_m: float,
        A_m2: float,
        f: Optional[float],
        K_minor: float = 0.0,
    ) -> float:
        """Escalar: R para dH = R * Q * |Q|."""
        if self.model != "darcy_weisbach":
            raise NotImplementedError(self.model)
        if f is None:
            raise ValueError("Darcy-Weisbach requiere f.")
        if D_m <= 0 or A_m2 <= 0 or L_m < 0:
            raise ValueError(f"Geometría inválida: D={D_m}, L={L_m}, A={A_m2}")

        R = (float(f) * (L_m / D_m)) / (2.0 * self.g_m_s2 * (A_m2**2))
        if self.include_minor and K_minor:
            R += float(K_minor) / (2.0 * self.g_m_s2 * (A_m2**2))
        return float(R)

    def segment_R_vec(
        self,
        *,
        D_m: np.ndarray,
        L_m: np.ndarray,
        A_m2: np.ndarray,
        f: np.ndarray,
        K_minor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorizado: arrays (n_seg,) -> R_seg (n_seg,)."""
        if self.model != "darcy_weisbach":
            raise NotImplementedError(self.model)

        D_m = np.asarray(D_m, dtype=float)
        L_m = np.asarray(L_m, dtype=float)
        A_m2 = np.asarray(A_m2, dtype=float)
        f = np.asarray(f, dtype=float)

        if K_minor is None:
            K_minor = 0.0
        K_minor = np.asarray(K_minor, dtype=float)

        if np.any(D_m <= 0) or np.any(A_m2 <= 0) or np.any(L_m < 0):
            raise ValueError("segment_R_vec: geometría inválida (D<=0, A<=0 o L<0).")

        R = (f * (L_m / D_m)) / (2.0 * self.g_m_s2 * (A_m2**2))
        if self.include_minor:
            R = R + (K_minor / (2.0 * self.g_m_s2 * (A_m2**2)))
        return R


def fric_term_vec(R: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    """Tu estabilización: R*q|q|/(1+alpha*R*|q|)"""
    qabs = np.abs(q)
    denom = 1.0 + alpha * R * qabs
    return (R * q * qabs) / denom


def fric_term(R: float, q: float, alpha: float) -> float:
    qabs = abs(q)
    denom = 1.0 + alpha * R * qabs
    return float((R * q * qabs) / denom)
