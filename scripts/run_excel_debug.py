import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
print(sys.path[0])

from moc.adapters.excel.read_excel import load_network_from_excel
from moc_engine.moc.core.hydraulics.friction import compute_friction_from_q0
from moc.core.build.discretize import discretize_network
from moc.core.build.wave_speed import compute_wave_speed
from moc.core.build.timestep import compute_global_dt
from moc.core.solver.moc_solver import run_moc_v01, MocRunConfig



# 1) Cargar Excel
network, config, ids = load_network_from_excel(r"C:/Users/Legion/Programas_Python/moc_engine/moc/examples/excel/moc_input_v0_1.xlsx")

# 2) Config solver
cfg = MocRunConfig(
    q0_m3s=float(config["q0_m3s"]),
    t_end_s=40.0,
    enable_cavitation_clamp=True,
    hvap_m=-9.5,
    fail_on_nan=True,
    max_abs_H=5e6,   # subido
    max_abs_Q=1e3,
    #init_mode="steady_friction",  # "steady" o "flat"
)

# 3) Fricción (ε + ν → Re → f)
nu = float(config.get("Viscosidad Cinemática", 1e-6))
network, fr = compute_friction_from_q0(
    network,
    q0_m3s=cfg.q0_m3s,
    nu_m2s=nu,
)
"""
# ==========================
# ✅ CHECKLIST AQUÍ
# ==========================

# Check 1: f y rugosidad
print("\n--- Fricción por tubo ---")
for p in network.pipes.values():
    print(p.name, "eps[m]=", p.roughness, "f=", p.friction, "poisson=", p.poisson, "E[Pa]=", p.elasticity)

# Check 2: Reynolds
print("\n--- Reynolds ---")
for uid, Re in fr.Re_by_pipe.items():
    print(uid, Re)

# Check 3: Celeridad de onda
ws = compute_wave_speed(network)

print("\n--- Celeridad de onda ---")
for uid, p in network.pipes.items():
    print(p.name, "c[m/s]=", ws.wave_speed_by_pipe[uid])
"""

# ==========================

# 4) Build
ws = compute_wave_speed(network)
disc = discretize_network(network)
ts = compute_global_dt(network, disc, ws)

# 5) Solver
res = run_moc_v01(network, disc, ws, ts, cfg)

print("Solver OK")
print("t_end logrado:", res.times[-1], "nt:", len(res.times), "npts:", res.H.shape[1])
print("meta:", res.meta)
# ==========================
from moc.core.solver.moc_solver import _build_chain_order
up_uid, chain, dn_uid = _build_chain_order(network)
print("\n----DIAGNÓSTICO CADENA (A)----")
print("UP:", network.nodes[up_uid].name, up_uid)
for p in chain:
    print(p.name, ":", p.node_from, "->", p.node_to, "L=", p.length, "D=", p.diameter)
print("DN:", network.nodes[dn_uid].name, dn_uid)

# ==========================
print("\n----DIAGNÓSTICO SEGMENTOS (B)----")
# construir s (distancia acumulada) consistente con la malla global
import numpy as np
s = np.zeros(res.H.shape[1])
# dx_seg es largo n_cells_total; si no lo tienes en el script, reconstruyelo con disc + chain
dxs = []
for p in chain:
    m = disc.meshes_by_pipe[p.uid]
    dxs += [m.dx]*m.n_cells
dxs = np.array(dxs, dtype=float)
s[1:] = np.cumsum(dxs)

t_check = 20.0
it = int(np.argmin(np.abs(res.times - t_check)))
Hs = res.H[it, :]

print("H min/max:", Hs.min(), Hs.max())
# chequeo monotonicidad (debería bajar casi siempre)
dH = Hs[:-1] - Hs[1:]
print("dH min/max:", dH.min(), dH.max())
print("segmentos con dH < 0 (sube H):", np.sum(dH < -1e-6))

# ==========================


# ==========================
# ✅ CHECKLIST POST-SOLVER
# ==========================

import numpy as np
import pandas as pd
from moc.core.solver.moc_solver import _build_chain_order

up_uid, chain, dn_uid = _build_chain_order(network)

t_check = 20.0
it = int(np.argmin(np.abs(res.times - t_check)))
print(f"DH a t={res.times[it]:.3f}s")

rows = []
i0 = 0
for p in chain:
    m = disc.meshes_by_pipe[p.uid]
    n = m.n_cells
    i1 = i0 + n

    H_ini = float(res.H[it, i0])
    H_fin = float(res.H[it, i1])
    DH = H_ini - H_fin

    rows.append({
        "pipe": p.name,
        "i0": i0,
        "i1": i1,
        "H_inicio_m": H_ini,
        "H_fin_m": H_fin,
        "DH_m": DH,
        "L_m": float(p.length),
        "f": float(p.friction) if p.friction is not None else None,
    })

    i0 = i1

df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
