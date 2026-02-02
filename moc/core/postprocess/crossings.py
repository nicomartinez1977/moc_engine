from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

from moc.core.postprocess.summary import ProfileSummary


@dataclass(frozen=True)
class CrossingResult:
    crossing_id: str
    name: str
    s_m: float
    s_used_m: float
    hmin_m: float
    hmax_m: float
    cav_any: bool


def _nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.abs(arr - value).argmin())


def evaluate_crossings_nearest(profile: ProfileSummary, crossings_csv: str) -> List[CrossingResult]:
    """
    Evaluates crossings using nearest-point sampling on the profile.
    (Good v0.1; later we can do linear interpolation.)
    """
    df = pd.read_csv(crossings_csv)

    required = {"cruce_id", "nombre", "s_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Crossings CSV missing columns: {sorted(missing)}")

    out: List[CrossingResult] = []

    for _, r in df.iterrows():
        cid = str(r["cruce_id"]).strip()
        name = str(r["nombre"]).strip()
        s = float(r["s_m"])

        i = _nearest_index(profile.s_m, s)
        s_used = float(profile.s_m[i])

        out.append(CrossingResult(
            crossing_id=cid,
            name=name,
            s_m=s,
            s_used_m=s_used,
            hmin_m=float(profile.hmin_m[i]),
            hmax_m=float(profile.hmax_m[i]),
            cav_any=bool(profile.cav_any[i]),
        ))

    return out


def export_crossing_results_csv(results: List[CrossingResult], path_csv: str) -> None:
    df = pd.DataFrame([{
        "cruce_id": r.crossing_id,
        "nombre": r.name,
        "s_m": r.s_m,
        "s_used_m": r.s_used_m,
        "hmin_m": r.hmin_m,
        "hmax_m": r.hmax_m,
        "cav_any": r.cav_any,
    } for r in results])

    df.to_csv(path_csv, index=False)
