from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd

from moc.core.postprocess.crossings import CrossingResult


@dataclass(frozen=True)
class CrossingEngineeringRow:
    cruce_id: str
    nombre: str
    s_m: float

    hmin_m: float
    hmax_m: float

    pmin_mca: float
    pmax_mca: float
    pmin_bar: float
    pmax_bar: float

    margen_cavitacion_m: float
    cavita: bool

    criterio_gobierno: str


def crossings_engineering_report(
    crossings: List[CrossingResult],
    *,
    hvap_m: float,
    rho: float = 1000.0,
    g: float = 9.81,
    top_n: int = 1
) -> List[CrossingEngineeringRow]:
    """
    Build an engineering-style report for crossings.

    criterio_gobierno:
      - sobrepresion: in top_n highest Hmax
      - depresion: in top_n lowest Hmin
      - cavitacion: cav_any True and in top_n lowest (Hmin - Hvap)
      - combinations joined with '+'
      - 'ninguno' otherwise
    """
    if not crossings:
        return []

    def m_to_bar(h_m: float) -> float:
        return (rho * g * h_m) / 1e5

    n = max(1, int(top_n))

    # Overpressure: highest Hmax
    sorted_over = sorted(crossings, key=lambda c: c.hmax_m, reverse=True)
    over_set = {c.crossing_id for c in sorted_over[:n]}

    # Depression: lowest Hmin
    sorted_under = sorted(crossings, key=lambda c: c.hmin_m)
    under_set = {c.crossing_id for c in sorted_under[:n]}

    # Cavitation governing: among cavitating points, lowest margin first
    cav_cross = [c for c in crossings if bool(c.cav_any)]
    cav_set = set()
    if cav_cross:
        cav_sorted = sorted(cav_cross, key=lambda c: (c.hmin_m - hvap_m))
        cav_set = {c.crossing_id for c in cav_sorted[:n]}

    rows: List[CrossingEngineeringRow] = []
    for c in crossings:
        margen = c.hmin_m - hvap_m
        cavita = bool(c.cav_any) or (margen < 0.0)

        # v0.1: treat head as mca proxy directly
        pmin_mca = float(c.hmin_m)
        pmax_mca = float(c.hmax_m)

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
            s_m=float(c.s_m),

            hmin_m=float(c.hmin_m),
            hmax_m=float(c.hmax_m),

            pmin_mca=pmin_mca,
            pmax_mca=pmax_mca,
            pmin_bar=float(m_to_bar(pmin_mca)),
            pmax_bar=float(m_to_bar(pmax_mca)),

            margen_cavitacion_m=float(margen),
            cavita=bool(cavita),

            criterio_gobierno=criterio,
        ))

    return rows


def export_crossings_engineering_csv(rows: List[CrossingEngineeringRow], path_csv: str) -> None:
    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(path_csv, index=False)
