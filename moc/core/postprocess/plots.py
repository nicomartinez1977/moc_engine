from __future__ import annotations

from typing import Dict, Optional
import os

import pandas as pd
import matplotlib.pyplot as plt


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    # match exact
    for c in candidates:
        if c in df.columns:
            return c
    # match case-insensitive
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def plot_timeseries_pressure(
    df: pd.DataFrame,
    *,
    title: str,
    out_png: str,
) -> None:
    """
    Grafica P_bar vs t_s.
    Acepta alias típicos: t_s/t/time_s/tiempo_s y P_bar/p_bar/pressure_bar.
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    df = _normalize_columns(df)

    t_col = _find_col(df, ["t_s", "t", "time_s", "tiempo_s", "Tiempo", "tiempo"])
    p_col = _find_col(df, ["P_bar", "p_bar", "pressure_bar", "presion_bar", "presión_bar"])

    if t_col is None or p_col is None:
        raise KeyError(
            f"Columnas requeridas no encontradas. "
            f"Se buscó tiempo en {['t_s','t','time_s','tiempo_s', 'Tiempo','tiempo']} "
            f"y presión en {['P_bar','p_bar','pressure_bar','presion_bar','presión_bar']}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    plt.figure()
    plt.plot(df[t_col], df[p_col])
    plt.xlabel("t [s]")
    plt.ylabel("P [bar]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_all_crossings(
    series_by_id: Dict[str, pd.DataFrame],
    *,
    out_folder: str = "plots",
    name_by_id: Optional[Dict[str, str]] = None,
) -> None:
    """
    Genera un PNG por cruce, con presión vs tiempo.
    """
    os.makedirs(out_folder, exist_ok=True)
    for cid, df in series_by_id.items():
        name = name_by_id.get(cid, "") if name_by_id else ""
        title = f"{cid} {name}".strip()
        safe = "".join(ch for ch in title if ch not in r'[]:*?/\\').strip().replace(" ", "_")
        if not safe:
            safe = cid
        out_png = os.path.join(out_folder, f"{safe}.png")
        plot_timeseries_pressure(df, title=title, out_png=out_png)
