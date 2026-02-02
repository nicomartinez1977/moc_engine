from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

from moc.core.postprocess.crossings import CrossingResult
from moc.core.postprocess.geometry import GeometryProfile


@dataclass(frozen=True)
class CrossingEngineeringRow:
    cruce_id: str
    nombre: str
    s: float

    z: float

    hmin: float
    hmax: float

    pmin: float
    pmax: float

    margen_cavitacion: float
    cavita: bool

    criterio_gobierno: str


def _interp_linear(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    if xq < float(x[0]) or xq > float(x[-1]):
        raise ValueError(
            f"El punto s_m={xq} está fuera del rango del perfil [{float(x[0])}, {float(x[-1])}] m."
        )

    j = int(np.searchsorted(x, xq, side="right"))
    j0 = max(0, j - 1)
    j1 = min(len(x) - 1, j)

    x0, x1 = float(x[j0]), float(x[j1])
    y0, y1 = float(y[j0]), float(y[j1])

    if x1 == x0:
        return y0

    w = (xq - x0) / (x1 - x0)
    return y0 * (1.0 - w) + y1 * w


def crossings_engineering_report_mca(
    crossings: List[CrossingResult],
    geom: GeometryProfile,
    *,
    hvap_m: float,
    top_n: int = 1,
) -> List[CrossingEngineeringRow]:
    """
    Reporte de ingeniería por cruce en m/mca.

    - z: interpolado desde GeometryProfile
    - p = H - z  (mca)
    - margen_cavitacion = pmin - hvap_m  (mca)

    criterio_gobierno:
      - sobrepresion: top_n por pmax (mayor)
      - depresion: top_n por pmin (menor)
      - cavitacion: top_n por margen_cavitacion (menor) entre cavitantes
    """
    if not crossings:
        return []

    n = max(1, int(top_n))

    tmp = []
    for c in crossings:
        zc = _interp_linear(geom.s_m, geom.z_m, float(c.s_m))
        pmin = float(c.hmin_m) - zc
        pmax = float(c.hmax_m) - zc

        margen = pmin - float(hvap_m)
        cavita = bool(c.cav_any) or (margen < 0.0)

        tmp.append({
            "cruce": c,
            "z": zc,
            "pmin": pmin,
            "pmax": pmax,
            "margen": margen,
            "cavita": cavita,
        })

    over_set = {t["cruce"].crossing_id for t in sorted(tmp, key=lambda r: r["pmax"], reverse=True)[:n]}
    under_set = {t["cruce"].crossing_id for t in sorted(tmp, key=lambda r: r["pmin"])[:n]}

    cav_tmp = [t for t in tmp if t["cavita"]]
    cav_set = set()
    if cav_tmp:
        cav_set = {t["cruce"].crossing_id for t in sorted(cav_tmp, key=lambda r: r["margen"])[:n]}

    rows: List[CrossingEngineeringRow] = []
    for t in tmp:
        c = t["cruce"]
        flags = []
        if c.crossing_id in over_set:
            flags.append("sobrepresion")
        if c.crossing_id in under_set:
            flags.append("depresion")
        if c.crossing_id in cav_set:
            flags.append("cavitacion")
        criterio = "+".join(flags) if flags else "ninguno"

        rows.append(CrossingEngineeringRow(
            cruce_id=c.crossing_id,
            nombre=c.name,
            s=float(c.s_m),

            z=float(t["z"]),

            hmin=float(c.hmin_m),
            hmax=float(c.hmax_m),

            pmin=float(t["pmin"]),
            pmax=float(t["pmax"]),

            margen_cavitacion=float(t["margen"]),
            cavita=bool(t["cavita"]),

            criterio_gobierno=criterio,
        ))

    return rows


_UNITS_REPORT = {
    "cruce_id": "-",
    "nombre": "-",
    "s": "m",
    "z": "m",
    "hmin": "m",
    "hmax": "m",
    "pmin": "mca",
    "pmax": "mca",
    "margen_cavitacion": "mca",
    "cavita": "-",
    "criterio_gobierno": "-",
}


def _df_with_units_row(df: pd.DataFrame, units: dict) -> pd.DataFrame:
    unit_row = {col: units.get(col, "-") for col in df.columns}
    return pd.concat([pd.DataFrame([unit_row]), df], ignore_index=True)


def export_crossings_engineering_csv(rows: List[CrossingEngineeringRow], path_csv: str) -> None:
    df = pd.DataFrame([r.__dict__ for r in rows])
    df_out = _df_with_units_row(df, _UNITS_REPORT)
    df_out.to_csv(path_csv, index=False)


def export_crossings_engineering_excel(rows: List[CrossingEngineeringRow], path_xlsx: str, sheet_name: str = "cruces") -> None:
    import openpyxl  # ensures dependency exists
    df = pd.DataFrame([r.__dict__ for r in rows])
    df_out = _df_with_units_row(df, _UNITS_REPORT)
    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)
