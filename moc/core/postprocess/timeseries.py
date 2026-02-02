from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from moc.core.postprocess.summary import ProfileSummary
from moc.core.solver.moc_solver import MocResults


def _interp_series_spatial(res: MocResults, s: np.ndarray, s_q: float) -> np.ndarray:
    """
    Interpola H(t, s_q) en el dominio espacial, para todos los tiempos.
    Usa interpolación lineal entre puntos vecinos.
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


def extract_timeseries_at_s(
    res: MocResults,
    profile: ProfileSummary,
    *,
    s_m: float,
    rho: float = 1000.0,
    g: float = 9.81,
) -> pd.DataFrame:
    """
    Devuelve DataFrame con:
      t_s, H_m, P_bar
    en la posición s_m (distancia acumulada).
    """
    H_ts = _interp_series_spatial(res, profile.s_m, float(s_m))
    P_bar = (rho * g * H_ts) / 1e5

    return pd.DataFrame({
        "t_s": res.times,  # res.times en tu MocResults
        "H_m": H_ts,
        "P_bar": P_bar,
    })


def extract_timeseries_at_index(
    res: MocResults,
    *,
    idx: int,
    rho: float = 1000.0,
    g: float = 9.81,
) -> pd.DataFrame:
    """
    Devuelve DataFrame t_s, H_m, P_bar en un índice espacial exacto (punto de malla).
    Útil para extremos (0 o -1) sin interpolación.
    """
    if idx < 0 or idx >= res.H.shape[1]:
        raise ValueError(f"Índice espacial fuera de rango: idx={idx}")

    H_ts = res.H[:, idx]
    P_bar = (rho * g * H_ts) / 1e5

    return pd.DataFrame({
        "t_s": res.times,
        "H_m": H_ts,
        "P_bar": P_bar,
    })


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


def extract_timeseries_for_crossings(
    res: MocResults,
    profile: ProfileSummary,
    crossings: List[CrossingSpec],
    *,
    rho: float = 1000.0,
    g: float = 9.81,
) -> Dict[str, pd.DataFrame]:
    """
    Retorna dict: cruce_id -> DataFrame (t_s, H_m, P_bar)
    """
    out: Dict[str, pd.DataFrame] = {}
    for c in crossings:
        out[c.cruce_id] = extract_timeseries_at_s(res, profile, s_m=c.s_m, rho=rho, g=g)
    return out


def export_timeseries_to_excel(
    series_by_id: Dict[str, pd.DataFrame],
    path_xlsx: str,
    *,
    name_by_id: Optional[Dict[str, str]] = None,
    max_sheetname_len: int = 31,
) -> None:
    """
    Exporta cada serie como una hoja distinta en un Excel.
    """
    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
        for cid, df in series_by_id.items():
            base = cid
            if name_by_id and cid in name_by_id and name_by_id[cid]:
                base = f"{cid}_{name_by_id[cid]}"
            # sanitizar nombre de hoja
            sheet = "".join(ch for ch in base if ch not in r'[]:*?/\\').strip()
            if not sheet:
                sheet = cid
            sheet = sheet[:max_sheetname_len]
            df.to_excel(writer, sheet_name=sheet, index=False)


def export_timeseries_to_csv_folder(
    series_by_id: Dict[str, pd.DataFrame],
    folder: str,
) -> List[str]:
    """
    Exporta CSV por cada serie. Retorna lista de paths generados.
    """
    import os
    os.makedirs(folder, exist_ok=True)
    out_paths = []
    for cid, df in series_by_id.items():
        path = os.path.join(folder, f"serie_{cid}.csv")
        df.to_csv(path, index=False)
        out_paths.append(path)
    return out_paths
