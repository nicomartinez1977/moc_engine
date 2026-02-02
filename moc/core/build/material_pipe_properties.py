from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

from moc.core.models.network import Network
from moc.core.models.pipe import Pipe

# ============================================================
# Canonicalización de materiales de tuberías
# ============================================================
# Entrada (Excel / CAD / GIS)  ->  Canonical (core)
#
# Regla:
#  - keys en lowercase
#  - sin tildes
#  - sin espacios extremos
#  - valores canónicos estables
# ============================================================

MATERIAL_CANONICAL_MAP = {

    # ----------------
    # HDPE / PEAD
    # ----------------
    "hdpe": "hdpe",
    "pead": "hdpe",
    "pe": "hdpe",
    "polietileno": "hdpe",
    "polietileno alta densidad": "hdpe",
    "polietileno de alta densidad": "hdpe",

    # ----------------
    # PVC
    # ----------------
    "pvc": "pvc",
    "upvc": "pvc",
    "pvc-u": "pvc",
    "pvc presion": "pvc",
    "pvc presión": "pvc",

    # ----------------
    # ACERO
    # ----------------
    "acero": "steel",
    "steel": "steel",
    "carbon steel": "steel",
    "acero carbono": "steel",
    "acero galvanizado": "steel",
    "galvanizado": "steel",

    # ----------------
    # HIERRO DÚCTIL
    # ----------------
    "hierro ductil": "ductile_iron",
    "hierro dúctil": "ductile_iron",
    "ductile iron": "ductile_iron",
    "di": "ductile_iron",
    "fundicion ductil": "ductile_iron",
    "fundición dúctil": "ductile_iron",

    # ----------------
    # PRFV / GRP / FRP
    # ----------------
    "prfv": "frp",
    "grp": "frp",
    "frp": "frp",
    "fibra de vidrio": "frp",
    "plastico reforzado con fibra de vidrio": "frp",
    "plástico reforzado con fibra de vidrio": "frp",
}


DEFAULT_ELASTICITY_BY_MATERIAL: Dict[str, float] = {
    "hdpe": 1.0e9,
    "pvc": 3.0e9,
    "steel": 2.10e11,
    "ductile_iron": 1.65e11,
    "frp": 2.0e10,
}

DEFAULT_POISSON_BY_MATERIAL: Dict[str, float] = {
    "hdpe": 0.45,
    "pvc": 0.38,
    "steel": 0.30,
    "ductile_iron": 0.28,
    "frp": 0.25,
}


def apply_material_defaults(network: Network) -> Network:
    """
    Completa propiedades físicas faltantes (E, nu) usando material canónico.
    Retorna un nuevo Network (no muta el original).
    """
    new_pipes: Dict[str, Pipe] = {}

    for puid, p in network.pipes.items():
        E = p.elasticity
        nu = p.poisson

        if p.material:
            mat = p.material

            if E is None and mat in DEFAULT_ELASTICITY_BY_MATERIAL:
                E = DEFAULT_ELASTICITY_BY_MATERIAL[mat]

            if nu is None and mat in DEFAULT_POISSON_BY_MATERIAL:
                nu = DEFAULT_POISSON_BY_MATERIAL[mat]

        # Solo reemplazamos si cambió algo
        if E != p.elasticity or nu != p.poisson:
            new_pipes[puid] = replace(p, elasticity=E, poisson=nu)
        else:
            new_pipes[puid] = p

    return Network(
        nodes=network.nodes,
        pipes=new_pipes,
        events=network.events,
        gravity=network.gravity,
        density=network.density,
    )
