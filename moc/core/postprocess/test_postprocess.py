# test_postprocess.py
from __future__ import annotations

# ============================================================
# BUILD / IO
# ============================================================

from moc.adapters.excel.read_excel import load_network_from_excel

from moc.core.build.validate import validate_network, raise_on_errors
from moc.core.build.wave_speed import compute_wave_speed
from moc.core.build.discretize import discretize_network
from moc.core.build.timestep import compute_global_dt
from moc.core.build.config import ModelConfig

# ============================================================
# SOLVER (RUN NUEVO)
# ============================================================

from moc.core.solver.moc_solver import run_moc_v01

# ============================================================
# POSTPROCESS – PERFIL / GEOMETRIA / CRUCES
# ============================================================

from moc.core.postprocess.summary import (
    summarize_profile_accumulated,
    build_pipe_ranges_accumulated,
)
from moc.core.postprocess.geometry import build_z_profile_linear
from moc.core.postprocess.pressure_profile import build_pressure_profile

from moc.core.postprocess.crossings import (
    evaluate_crossings_linear,
    export_crossing_results_csv,
)
from moc.core.postprocess.report_crossings import (
    crossings_engineering_report,
    export_crossings_engineering_csv,
    export_crossings_engineering_excel,
)

# ============================================================
# POSTPROCESS – TIMESERIES (NUEVO)
# ============================================================

from moc.core.postprocess.timeseries import (
    read_crossings_csv,
    extract_timeseries_for_crossings_mca,
    export_timeseries_to_excel,
    export_timeseries_to_csv_folder,
)

from moc.core.postprocess.plots import plot_all_crossings


# ============================================================
# INPUTS
# ============================================================

EXCEL_INPUT = r"C:/Users/Legion/Programas_Python/moc_engine/moc/examples/excel/moc_input_v0_1.xlsx"
CRUCES_CSV = "cruces.csv"

# Outputs
OUT_PROFILE_CSV = "perfil_critico.csv"
OUT_PROFILE_XLSX = "perfil_critico.xlsx"
OUT_PIPE_SUMMARY = "resumen_por_tubo.csv"

OUT_CRUCES_RESUMEN = "resumen_cruces.csv"
OUT_CRUCES_ING_CSV = "reporte_cruces_ingenieria.csv"
OUT_CRUCES_ING_XLSX = "reporte_cruces_ingenieria.xlsx"

OUT_SERIES_XLSX = "series_cruces.xlsx"
OUT_SERIES_FOLDER = "series_csv"
OUT_PLOTS_FOLDER = "plots_cruces"


# ============================================================
# PIPELINE
# ============================================================

# --- Leer Excel
net, cfg, _ = load_network_from_excel(EXCEL_INPUT)
raise_on_errors(validate_network(net))

# --- Configuración agregada (RUN / INIT / HEADLOSS)
cfg_model = ModelConfig.from_dict(
    cfg,
    network_gravity=getattr(net, "gravity", None),
)

# --- Preproceso hidráulico
ws = compute_wave_speed(net, default_thickness_m=0.01)
disc = discretize_network(net, default_dx_m=10.0)
ts = compute_global_dt(net, disc, ws)

# ============================================================
# PARÁMETROS DE ESCENARIO
# ============================================================

q0_m3s = float(cfg.get("q0_m3s", 0.05))
hvap_m = float(cfg.get("hvap_m", -9.5))

# ============================================================
# SOLVER (NUEVO)
# ============================================================

res = run_moc_v01(
    net,
    disc,
    ws,
    ts,
    model_cfg=cfg_model,
    q0_m3s=q0_m3s,
    hvap_m=hvap_m,
)

print("SOLVER OK")

# ============================================================
# PERFIL + GEOMETRIA + PRESION
# ============================================================

prof = summarize_profile_accumulated(net, disc, res)
geom = build_z_profile_linear(net, disc)
pprof = build_pressure_profile(prof, geom)

print("PROFILE + Z OK")
print(
    "Pmin [mca] =",
    float(pprof.pmin_mca.min()),
    "| Pmax [mca] =",
    float(pprof.pmax_mca.max()),
)

# ============================================================
# CRUCES
# ============================================================

crossings = evaluate_crossings_linear(prof, CRUCES_CSV)
export_crossing_results_csv(crossings, OUT_CRUCES_RESUMEN)
print("CRUCES OK")

rows = crossings_engineering_report(
    crossings,
    geom,
    hvap_m=hvap_m,
    rho_kg_m3=getattr(net, "density", 1000.0),
    g_m_s2=cfg_model.headloss.g_m_s2,
    top_n=1,
)

export_crossings_engineering_csv(rows, OUT_CRUCES_ING_CSV)
export_crossings_engineering_excel(rows, OUT_CRUCES_ING_XLSX)
print("REPORTE CRUCES INGENIERIA OK")

# ============================================================
# SERIES DE TIEMPO (m / mca)
# ============================================================

cross_specs = read_crossings_csv(CRUCES_CSV)

series_by_id = extract_timeseries_for_crossings_mca(
    net,
    res,
    prof,
    geom,
    cross_specs,
)

name_by_id = {c.cruce_id: c.nombre for c in cross_specs}

# --- Excel multi-hoja (con downsample y gráfico H vs t)
export_timeseries_to_excel(
    series_by_id,
    OUT_SERIES_XLSX,
    write_every_s=cfg_model.run.dt_out_s,
    name_by_id=name_by_id,
)

# --- CSV por cruce
export_timeseries_to_csv_folder(series_by_id, OUT_SERIES_FOLDER)

# --- Gráficos automáticos
plot_all_crossings(
    series_by_id,
    out_folder=OUT_PLOTS_FOLDER,
    name_by_id=name_by_id,
)

print("SERIES + EXCEL + CSV + PLOTS OK")

# ============================================================
# EXPORTS ADICIONALES
# ============================================================

ranges = build_pipe_ranges_accumulated(net, disc)

from moc.core.postprocess.export import (
    export_profile_csv,
    export_profile_excel,
    export_pipe_summary_csv,
)

export_profile_csv(prof, OUT_PROFILE_CSV)
export_profile_excel(prof, OUT_PROFILE_XLSX)
export_pipe_summary_csv(net, ranges, prof, OUT_PIPE_SUMMARY)

print("EXPORT OK")
