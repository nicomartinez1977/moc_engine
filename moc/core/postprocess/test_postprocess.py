from moc.adapters.excel.read_excel import load_network_from_excel
from moc.core.build.validate import validate_network, raise_on_errors
from moc.core.build.wave_speed import compute_wave_speed
from moc.core.build.discretize import discretize_network
from moc.core.build.timestep import compute_global_dt
from moc.core.postprocess.timeseries import extract_timeseries_at_s
from moc.core.solver.moc_solver import run_moc_v01, MocRunConfig
from moc.core.postprocess.summary import (
    summarize_profile_accumulated,
    build_pipe_ranges_accumulated,
)
from moc.core.postprocess.export import (
    export_profile_csv,
    export_profile_excel,
    export_pipe_summary_csv,
)

path = r"C:/Users/Legion/Programas_Python/moc_engine/moc/examples/excel/moc_input_v0_1.xlsx"

# --- PIPELINE ---
net, cfg, _ = load_network_from_excel(path)
raise_on_errors(validate_network(net))

ws = compute_wave_speed(net, default_thickness_m=0.01)
disc = discretize_network(net, default_dx_m=10.0)
ts = compute_global_dt(net, disc, ws)

q0 = float(cfg.get("q0_m3s", 0.05))
t_end = float(cfg.get("t_end_s", 30.0))
hv = float(cfg.get("hvap_m", -9.5))

res = run_moc_v01(
    net, disc, ws, ts,
    MocRunConfig(q0_m3s=q0, t_end_s=t_end, hvap_m=hv)
)

# --- POSTPROCESO ---
prof = summarize_profile_accumulated(net, disc, res)
ranges = build_pipe_ranges_accumulated(net, disc)

# --- EXPORT ---
export_profile_csv(prof, "perfil_critico.csv")
export_profile_excel(prof, "perfil_critico.xlsx")
export_pipe_summary_csv(net, ranges, prof, "resumen_por_tubo.csv")

print("EXPORT OK")
from moc.core.postprocess.crossings import evaluate_crossings_linear, export_crossing_results_csv

crossings = evaluate_crossings_linear(prof, "cruces.csv")
export_crossing_results_csv(crossings, "resumen_cruces.csv")
print("CRUCES OK")

s_cruce = 2100.0

df_ts = extract_timeseries_at_s(res, prof, s_m=s_cruce)
df_ts.to_csv("serie_tiempo_cruce_C2.csv", index=False)

print("SERIE TIEMPO OK")

from moc.core.postprocess.timeseries import (
    read_crossings_csv,
    extract_timeseries_for_crossings,
    export_timeseries_to_excel,
    export_timeseries_to_csv_folder,
)
from moc.core.postprocess.plots import plot_all_crossings

crossings_specs = read_crossings_csv("cruces.csv")

series_by_id = extract_timeseries_for_crossings(res, prof, crossings_specs)

# Export D) Excel multi-hoja
name_by_id = {c.cruce_id: c.nombre for c in crossings_specs}
export_timeseries_to_excel(series_by_id, "series_cruces.xlsx", name_by_id=name_by_id)

# Export A) CSV por cruce
export_timeseries_to_csv_folder(series_by_id, "series_csv")

# Export B) Gráficos automáticos
plot_all_crossings(series_by_id, out_folder="plots_cruces", name_by_id=name_by_id)

print("SERIES + EXCEL + CSV + PLOTS OK")


from moc.core.postprocess.timeseries import extract_timeseries_at_index

df_up = extract_timeseries_at_index(res, idx=0)
df_dn = extract_timeseries_at_index(res, idx=res.H.shape[1]-1)

df_up.to_csv("serie_inicio.csv", index=False)
df_dn.to_csv("serie_fin.csv", index=False)
