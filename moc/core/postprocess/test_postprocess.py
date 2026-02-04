from moc.adapters.excel.read_excel import load_network_from_excel

from moc.core.build.validate import validate_network, raise_on_errors
from moc.core.build.wave_speed import compute_wave_speed
from moc.core.build.discretize import discretize_network
from moc.core.build.timestep import compute_global_dt

from moc.core.solver.moc_solver import run_moc_v01, MocRunConfig

from moc.core.postprocess.summary import summarize_profile_accumulated
from moc.core.postprocess.geometry import build_z_profile_linear

from moc.core.postprocess.crossings import evaluate_crossings_linear, export_crossing_results_csv
from moc.core.postprocess.report_crossings import (
    crossings_engineering_report_mca,
    export_crossings_engineering_excel,
)

from moc.core.postprocess.timeseries import (
    read_crossings_csv,
    extract_timeseries_for_crossings_mca,
    export_timeseries_to_excel,
)

from moc.core.postprocess.plots import plot_all_crossings


# =========================
# CONFIG
# =========================
path = r"C:/Users/Legion/Programas_Python/moc_engine/moc/examples/excel/moc_input_v0_1.xlsx"
cruces_csv = "cruces.csv"

# =========================
# LOAD + BUILD
# =========================
net, cfg, _ = load_network_from_excel(path)
raise_on_errors(validate_network(net))

ws = compute_wave_speed(net, default_thickness_m=0.01)
disc = discretize_network(net, default_dx_m=10.0)
ts = compute_global_dt(net, disc, ws)

q0 = float(cfg.get("q0_m3s", 0.05))
t_end = float(cfg.get("t_end_s", 30.0))
hvap = float(cfg.get("hvap_m", -9.5))

res = run_moc_v01(
    net, disc, ws, ts,
    MocRunConfig(q0_m3s=q0, t_end_s=t_end, hvap_m=hvap)
)

print("SOLVER OK")


# =========================
# PROFILE + GEOMETRY
# =========================
prof = summarize_profile_accumulated(net, disc, res)
geom = build_z_profile_linear(net, disc)

print("PROFILE + Z OK  |  s_total_m =", float(prof.s_m[-1]))


# =========================
# CRUCES (Hmin/Hmax interpolados)
# =========================
# Nota: esta exportación es el "resumen simple" de cruces (sin unidades especiales).
# Si no lo quieres, puedes comentarlo.
crossings = evaluate_crossings_linear(prof, cruces_csv)
export_crossing_results_csv(crossings, "resumen_cruces.csv")
print("CRUCES OK")


# =========================
# REPORTE INGENIERIA (mca, con z)
# =========================
rows = crossings_engineering_report_mca(crossings, geom, hvap_m=hvap, top_n=1)
export_crossings_engineering_excel(rows, "reporte_cruces_ingenieria.xlsx", sheet_name="cruces")
print("REPORTE CRUCES INGENIERIA (mca) OK")


# =========================
# SERIES TIEMPO por cruce (mca) + Excel multi-hoja + plots
# =========================
cross_specs = read_crossings_csv(cruces_csv)
series_by_id = extract_timeseries_for_crossings_mca(net, res, prof, geom, cross_specs)

name_by_id = {c.cruce_id: c.nombre for c in cross_specs}
export_timeseries_to_excel(series_by_id, "series_cruces.xlsx", name_by_id=name_by_id)

#plot_all_crossings(series_by_id, out_folder="plots_cruces", name_by_id=name_by_id)

print("SERIES (EXCEL) + PLOTS OK")
