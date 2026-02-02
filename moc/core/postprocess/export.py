from __future__ import annotations

from typing import Optional
import pandas as pd

from moc.core.postprocess.summary import ProfileSummary, PipeRange
from moc.core.models.network import Network


def export_profile_csv(
    profile: ProfileSummary,
    path_csv: str,
) -> None:
    """
    Export accumulated-distance critical profile to CSV.
    Columns:
      s_m, hmin_m, hmax_m, cav_any
    """
    df = pd.DataFrame({
        "s_m": profile.s_m,
        "hmin_m": profile.hmin_m,
        "hmax_m": profile.hmax_m,
        "cav_any": profile.cav_any.astype(bool),
    })
    df.to_csv(path_csv, index=False)


def export_profile_excel(
    profile: ProfileSummary,
    path_xlsx: str,
    sheet_name: str = "perfil_critico",
) -> None:
    """
    Export accumulated-distance critical profile to Excel.
    """
    df = pd.DataFrame({
        "s_m": profile.s_m,
        "hmin_m": profile.hmin_m,
        "hmax_m": profile.hmax_m,
        "cav_any": profile.cav_any.astype(bool),
    })
    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def export_pipe_summary_csv(
    network: Network,
    ranges: list[PipeRange],
    profile: ProfileSummary,
    path_csv: str,
) -> None:
    """
    Export per-pipe summary:
      pipe_uid, s_start_m, s_end_m, hmin_m, hmax_m, cav_any
    """
    rows = []
    for r in ranges:
        i0, i1 = r.i_start, r.i_end
        rows.append({
            "pipe_uid": r.pipe_uid,
            "s_start_m": r.s_start_m,
            "s_end_m": r.s_end_m,
            "hmin_m": float(profile.hmin_m[i0:i1 + 1].min()),
            "hmax_m": float(profile.hmax_m[i0:i1 + 1].max()),
            "cav_any": bool(profile.cav_any[i0:i1 + 1].any()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(path_csv, index=False)
