from __future__ import annotations

from csv import writer
from csv import writer
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

def _downsample_df_by_time(df: pd.DataFrame, t_col: str, every_s: float) -> pd.DataFrame:
    """Devuelve df muestreado cada every_s (aprox), manteniendo el primer y último punto."""
    if every_s is None or every_s <= 0:
        return df

    t = df[t_col].to_numpy(dtype=float)
    if t.size < 2:
        return df

    t0 = float(t[0])
    t1 = float(t[-1])

    targets = np.arange(t0, t1 + 1e-12, every_s)
    idx = np.searchsorted(t, targets, side="left")
    idx = np.clip(idx, 0, len(t) - 1)
    idx = np.unique(idx)

    # asegurar primer y último
    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)

    return df.iloc[idx].reset_index(drop=True)


def export_timeseries_to_excel(
    series_by_id: Dict[str, pd.DataFrame],
    path_xlsx: str,
    write_every_s: float = 0.5,
    *,
    name_by_id: Optional[Dict[str, str]] = None,
    max_sheetname_len: int = 31,
) -> None:
    
    """
    Exporta cada serie como una hoja distinta en un Excel, con:
      - 2 filas de encabezado: nombres + unidades
      - formato: negrita y centrado en encabezados
      - gráfico: H vs t (título = nombre de hoja)
    """
    from openpyxl.styles import Font, Alignment, Border, Side, fills, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.chart import LineChart, Reference, ScatterChart, Series
    from openpyxl.chart.axis import DisplayUnitsLabel, DisplayUnitsLabelList
    from openpyxl.chart.text import RichText
    from openpyxl.drawing.text import Paragraph, ParagraphProperties, CharacterProperties, Font as DrawingFont

    # Unidades por columna (ajusta si cambias nombres)
    UNITS = {
        "t": "s",
        "H": "mca",
        "z": "m",
        "P_bar": "bar",
        "P": "m",
    }

    def _sheet_name(cid: str) -> str:
        base = cid
        if name_by_id and cid in name_by_id and name_by_id[cid]:
            base = f"{name_by_id[cid]}"
        sheet = "".join(ch for ch in base if ch not in r'[]:*?/\\').strip()
        if not sheet:
            sheet = cid
        return sheet[:max_sheetname_len]

    header_font = Font(name="Verdana", bold=True,size=10, color="FFFFFF")  # o bold=True si prefieres
    units_font = Font(bold=True,size=10,name="Verdana", color="FFFFFF")  # o bold=True si prefieres
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    data_font = Font(bold=False,size=10,name="Verdana")

    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
        for cid, df in series_by_id.items():
            sheet = _sheet_name(cid)

            # Asegurar orden de columnas típico si existe
            preferred = [c for c in ["t", "H", "z", "P", "P"] if c in df.columns]
            rest = [c for c in df.columns if c not in preferred]
            df_out = df[preferred + rest] if preferred else df

            
            # Downsample para no dejar el Excel pesado
            df_out = _downsample_df_by_time(df_out, "t", every_s=write_every_s)

            # Escribir DATA desde la fila 3 (startrow=2), sin header
            df_out.to_excel(writer, sheet_name=sheet, index=False, header=False, startrow=2)

            ws = writer.sheets[sheet]
            for row in ws.iter_rows(min_row=3, max_row=ws.max_row):
                for cell in row:
                    cell.font = data_font
                    cell.alignment = center
                    cell.number_format = '0.00'  # formato numérico simple
                    cell.border = Border(left=Side(style="thin"), right=Side(style="thin"), top=Side(style="thin"), bottom=Side(style="thin"))
            # Fila 1: nombres de columnas
            for j, col in enumerate(df_out.columns, start=1):
                cell = ws.cell(row=1, column=j, value=str(col))
                cell.font = header_font
                cell.alignment = center
                cell.border = Border(left=Side(style="thin"), right=Side(style="thin"), top=Side(style="thin"), bottom=Side(style=None))
                cell.fill = PatternFill(patternType="solid", fgColor="FF9900")  

            # Fila 2: unidades (si no existe, deja vacío)
            for j, col in enumerate(df_out.columns, start=1):
                unit = UNITS.get(str(col), "")
                cell = ws.cell(row=2, column=j, value=f"[{unit}]" if unit else "")
                cell.font = units_font
                cell.alignment = center
                cell.border = Border(left=Side(style="thin"), right=Side(style="thin"), top=Side(style="thin"), bottom=Side(style="thin"))
                cell.fill = PatternFill(patternType="solid", fgColor="FF9900")

            # Congelar paneles bajo las 2 primeras filas
            ws.freeze_panes = "A3"

            # Auto-ancho simple de columnas
            for j, col in enumerate(df_out.columns, start=1):
                col_letter = get_column_letter(j)
                # ancho basado en header + una muestra de datos
                sample = [str(col), UNITS.get(str(col), "")]
                if len(df_out) > 0:
                    sample += [str(df_out.iloc[0, j-1])]
                width = min(max(len(s) for s in sample) + 4, 28)
                ws.column_dimensions[col_letter].width = width

            # ------------------------------------------------------------
            # Gráfico H vs t (si existen columnas)
            # ------------------------------------------------------------
            
            if "t" in df_out.columns and "H" in df_out.columns:
                col_t = list(df_out.columns).index("t") + 1
                col_h = list(df_out.columns).index("H") + 1
                #col_P = list(df_out.columns).index("P") + 1

                nrows = len(df_out)
                first_data_row = 3
                last_data_row = first_data_row + nrows - 1

                # X values: t_s
                xvalues = Reference(ws, min_col=col_t, min_row=3,max_row=63) #min_row=first_data_row, max_row=last_data_row)
                # Y values: H_m
                yvalues = Reference(ws, min_col=col_h, min_row=3,max_row=63) #min_row=first_data_row, max_row=last_data_row)
                #Formatos de gráfico
                chart = ScatterChart(scatterStyle="smooth")
                chart.title = sheet  # título = nombre hoja
                #chart.x_axis.delete = Falsegit
                #chart.y_axis.delete = False
                #chart.width = 15
                #chart.height = 10
                chart.x_axis.title = "Tiempo [s]"
                chart.y_axis.title = "H [mca]"    
                #chart.x_axis.delete = False
                #chart.y_axis.delete = False


                Serie1 = Series(yvalues, xvalues, title="H")
                Serie1.marker.symbol = "none"
                chart.series.append(Serie1)
                chart.legend.position ="b"
                # Ubicación del gráfico
                ws.add_chart(chart, "G4")
            


def new_func():
    return True



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
