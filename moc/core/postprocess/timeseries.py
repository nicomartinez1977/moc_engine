from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from moc.core.models.network import Network
from moc.core.postprocess.summary import ProfileSummary
from moc.core.postprocess.geometry import GeometryProfile
from moc.core.solver.moc_solver import MocResults


# -------------------------
# Helpers (interpolation)
# -------------------------
def _interp_spatial_scalar(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """
    Linear interpolation y(xq) given monotonic x.
    Raises if outside range (estricto, ingeniería).
    """
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


def _interp_series_spatial(res: MocResults, s: np.ndarray, s_q: float) -> np.ndarray:
    """
    Interpola H(t, s_q) espacialmente para todos los tiempos.
    """
    if s_q < float(s[0]) or s_q > float(s[-1]):
        raise ValueError(f"El punto s_m={s_q} está fuera del rango [{float(s[0])}, {float(s[-1])}] m.")

    j = int(np.searchsorted(s, s_q, side="right"))
    j0 = max(0, j - 1)
    j1 = min(len(s) - 1, j)

    s0, s1 = float(s[j0]), float(s[j1])
    if s1 == s0:
        return res.H[:, j0].copy()

    w = (s_q - s0) / (s1 - s0)
    return (1.0 - w) * res.H[:, j0] + w * res.H[:, j1]


# -------------------------
# Public API
# -------------------------
@dataclass(frozen=True)
class CrossingSpec:
    cruce_id: str
    nombre: str
    s_m: float


def read_crossings_csv(path_csv: str) -> List[CrossingSpec]:
    df = pd.read_csv(path_csv)
    required = {"cruce_id", "nombre", "s_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El archivo de cruces no contiene las columnas requeridas: {sorted(missing)}")

    out: List[CrossingSpec] = []
    for _, r in df.iterrows():
        out.append(CrossingSpec(
            cruce_id=str(r["cruce_id"]).strip(),
            nombre=str(r["nombre"]).strip(),
            s_m=float(r["s_m"]),
        ))
    return out


def extract_timeseries_at_s_mca(
    network: Network,
    res: MocResults,
    prof: ProfileSummary,
    geom: GeometryProfile,
    *,
    s_m: float,
) -> pd.DataFrame:
    """
    Serie tiempo en posición s_m con columnas:
      t, H, z, P
    Unidades:
      t [s], H [m], z [m], P [mca] (P = H - z)

    Nota: aquí no usamos Pa/bar; todo en m/mca.
    """
    # H(t)
    H_ts = _interp_series_spatial(res, prof.s_m, float(s_m))

    # z(s)
    z_q = _interp_spatial_scalar(geom.s_m, geom.z_m, float(s_m))

    # Presión manométrica en mca
    P_ts = H_ts - z_q

    df = pd.DataFrame({
        "t": res.times,
        "H": H_ts,
        "z": np.full_like(H_ts, z_q, dtype=float),
        "P": P_ts,
    })
    return df


def extract_timeseries_for_crossings_mca(
    network: Network,
    res: MocResults,
    prof: ProfileSummary,
    geom: GeometryProfile,
    crossings: List[CrossingSpec],
) -> Dict[str, pd.DataFrame]:
    """
    Retorna dict: cruce_id -> DataFrame (t, H, z, P) en m/mca.
    """
    out: Dict[str, pd.DataFrame] = {}
    for c in crossings:
        out[c.cruce_id] = extract_timeseries_at_s_mca(network, res, prof, geom, s_m=c.s_m)
    return out


# -------------------------
# Export helpers (2-row header: names + units)
# -------------------------
_UNITS_TS = {"t": "s", "H": "m", "z": "m", "P": "mca"}


def _df_with_units_row(df: pd.DataFrame, units: Dict[str, str]) -> pd.DataFrame:
    """
    Returns a new DataFrame where the first row is units.
    Column names remain the variable names (no units in name).
    """
    unit_row = {col: units.get(col, "-") for col in df.columns}
    df2 = pd.concat([pd.DataFrame([unit_row]), df], ignore_index=True)
    return df2


def export_timeseries_to_excel(
    series_by_id: Dict[str, pd.DataFrame],
    path_xlsx: str,
    *,
    name_by_id: Optional[Dict[str, str]] = None,
    max_sheetname_len: int = 31,
) -> None:
    """
    Exporta cada serie como hoja distinta, con fila 1 = variables, fila 2 = unidades.
    """
    import openpyxl  # ensures dependency exists

    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
        for cid, df in series_by_id.items():
            base = cid
            if name_by_id and cid in name_by_id and name_by_id[cid]:
                base = f"{cid}_{name_by_id[cid]}"

            sheet = "".join(ch for ch in base if ch not in r'[]:*?/\\').strip()
            sheet = (sheet or cid)[:max_sheetname_len]

            df_out = _df_with_units_row(df, _UNITS_TS)
            df_out.to_excel(writer, sheet_name=sheet, index=False)


def export_timeseries_to_csv_folder(
    series_by_id: Dict[str, pd.DataFrame],
    folder: str,
) -> List[str]:
    """
    Exporta un CSV por cruce, con 2 filas header+unidad (igual que Excel).
    """
    import os
    os.makedirs(folder, exist_ok=True)

    out_paths: List[str] = []
    for cid, df in series_by_id.items():
        path = os.path.join(folder, f"serie_{cid}.csv")
        df_out = _df_with_units_row(df, _UNITS_TS)
        df_out.to_csv(path, index=False)
        out_paths.append(path)

    return out_paths
