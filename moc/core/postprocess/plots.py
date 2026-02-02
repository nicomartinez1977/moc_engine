from __future__ import annotations

from typing import Dict, Optional
import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_timeseries_pressure(
    df: pd.DataFrame,
    *,
    title: str,
    out_png: str,
) -> None:
    """
    Grafica P_bar vs t_s.
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["P_bar"])
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
