from moc.adapters.excel.read_excel import load_network_from_excel
from moc.core.build.validate import validate_network, raise_on_errors
from moc.core.build.wave_speed import compute_wave_speed
from moc.core.build.discretize import discretize_network
from moc.core.build.timestep import compute_global_dt
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
