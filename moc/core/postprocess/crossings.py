from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

from moc.core.postprocess.summary import ProfileSummary


@dataclass(frozen=True)
class CrossingResult:
    crossing_id: str
    name: str
    s_m: float
    hmin_m: float
    hmax_m: float
    cav_any: bool


def _interp_linear(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """
    Linear interpolation y(xq) given monotonic x.
    Raises if xq is outside range.
    """
    if xq < float(x[0]) or xq > float(x[-1]):
        raise ValueError(
            f"El cruce en s_m={xq} está fuera del rango del perfil "
            f"[{float(x[0])}, {float(x[-1])}] m."
        )
    # index of right neighbor
    j = int(np.searchsorted(x, xq, side="right"))
    if j == 0:
        return float(y[0])
    if j >= len(x):
        return float(y[-1])

    x0, x1 = float(x[j - 1]), float(x[j])
    y0, y1 = float(y[j - 1]), float(y[j])

    if x1 == x0:
        return float(y0)

    w = (xq - x0) / (x1 - x0)
    return y0 * (1.0 - w) + y1 * w


def _interp_bool_any_neighborhood(x: np.ndarray, b: np.ndarray, xq: float) -> bool:
    """
    For cavitation flags, linear interpolation doesn't make physical sense.
    Instead, take the OR of the two bracketing points (conservative).
    """
    if xq < float(x[0]) or xq > float(x[-1]):
        raise ValueError(
            f"El cruce en s_m={xq} está fuera del rango del perfil "
            f"[{float(x[0])}, {float(x[-1])}] m."
        )

    j = int(np.searchsorted(x, xq, side="right"))
    if j == 0:
        return bool(b[0])
    if j >= len(x):
        return bool(b[-1])

    return bool(b[j - 1] or b[j])


def evaluate_crossings_linear(profile: ProfileSummary, crossings_csv: str) -> List[CrossingResult]:
    """
    Evaluates crossings by linear interpolation on hmin/hmax.
    For cavitation, uses conservative OR of neighbor points.
    """
    df = pd.read_csv(crossings_csv)
    required = {"cruce_id", "nombre", "s_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El archivo de cruces no contiene las columnas requeridas: {sorted(missing)}")

    # Ensure profile.s_m is increasing
    if not np.all(np.diff(profile.s_m) >= 0):
        raise ValueError("PEl perfil s_m no es monótono creciente; no se puede interpolar.")

    out: List[CrossingResult] = []
    for _, r in df.iterrows():
        cid = str(r["cruce_id"]).strip()
        name = str(r["nombre"]).strip()
        s = float(r["s_m"])

        hmin = _interp_linear(profile.s_m, profile.hmin_m, s)
        hmax = _interp_linear(profile.s_m, profile.hmax_m, s)
        cav = _interp_bool_any_neighborhood(profile.s_m, profile.cav_any, s)

        out.append(CrossingResult(
            crossing_id=cid,
            name=name,
            s_m=s,
            hmin_m=float(hmin),
            hmax_m=float(hmax),
            cav_any=bool(cav),
        ))

    return out


def export_crossing_results_csv(results: List[CrossingResult], path_csv: str) -> None:
    df = pd.DataFrame([{
        "cruce_id": r.crossing_id,
        "nombre": r.name,
        "s_m": r.s_m,
        "hmin_m": r.hmin_m,
        "hmax_m": r.hmax_m,
        "cav_any": r.cav_any,
    } for r in results])
    df.to_csv(path_csv, index=False)
