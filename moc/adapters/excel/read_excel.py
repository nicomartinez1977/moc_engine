from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import pandas as pd

from moc.core.models.node import Node
from moc.core.models.pipe import Pipe
from moc.core.models.event import Event
from moc.core.models.network import Network


# -----------------------------
# Excel contract (Spanish)
# -----------------------------
SHEET_NODO = "nodo"
SHEET_TUBERIA = "tuberia"
SHEET_EVENTOS = "eventos"
SHEET_CONFIG = "config"

# Required columns (snake_case)
REQ_NODO = {"nodo_id", "nombre", "cota_m", "tipo_cond_borde", "valor_cond_borde"}
REQ_TUBO = {"tubo_id", "nombre", "nodo_inicio", "nodo_fin", "longitud_m", "diametro_int_m"}
REQ_EVENTO = {"evento_id", "tipo_evento", "objetivo_tipo", "objetivo_id", "t_inicio_s", "t_fin_s"}
REQ_CONFIG = {"clave", "valor"}

# Mappings: Spanish -> Core (English)
BC_MAP = {
    "none": "none",
    "estanque": "reservoir",
    "demanda": "demand",
    "valvula": "valve",
    "bomba": "pump",
}

EVENT_MAP = {
    "cierre_valvula": "valve_close",
    "apertura_valvula": "valve_open",
    "paro_bomba": "pump_trip",
    "arranque_bomba": "pump_start",
}

ALLOWED_OBJETIVO_TIPO = {"nodo", "tubo"}


@dataclass(frozen=True)
class ExcelIds:
    """Holds mapping from Excel IDs to internal UIDs."""
    node_uid_by_excel_id: Dict[str, str]
    pipe_uid_by_excel_id: Dict[str, str]


def _norm_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def _norm_lower(x: Any) -> str:
    return _norm_str(x).lower()


def _require_columns(df: pd.DataFrame, required: set[str], sheet: str) -> None:
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Sheet '{sheet}' is missing required columns: {missing}")


def _as_float(x: Any, field: str, sheet: str, row_hint: str) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == ""):
            raise ValueError("empty")
        return float(x)
    except Exception as e:
        raise ValueError(f"Invalid numeric value for '{field}' in sheet '{sheet}' ({row_hint}): {x!r}") from e


def _maybe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == ""):
            return None
        return float(x)
    except Exception:
        return None


def _parse_tags(tags_cell: Any) -> Dict[str, Any]:
    """
    Optional tags column: "a;b;c" -> {"tags": ["a","b","c"]}
    """
    s = _norm_str(tags_cell)
    if not s:
        return {}
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return {"tags": parts} if parts else {}


def _parse_params_json(cell: Any) -> Dict[str, Any]:
    """
    Optional JSON parameters cell. Empty -> {}.
    """
    s = _norm_str(cell)
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in 'parametros_json': {s!r}") from e


def load_network_from_excel(path: str) -> Tuple[Network, Dict[str, Any], ExcelIds]:
    """
    Reads 'nodo', 'tuberia', 'eventos', 'config' from an Excel file (Spanish contract)
    and returns:
      - Network (canonical core model, in English)
      - config dict from 'config' sheet (keys normalized)
      - ExcelIds mapping (Excel ids -> internal uids)
    """

    # Read sheets
    df_nodo = pd.read_excel(path, sheet_name=SHEET_NODO, engine="openpyxl")
    df_tubo = pd.read_excel(path, sheet_name=SHEET_TUBERIA, engine="openpyxl")
    df_evento = pd.read_excel(path, sheet_name=SHEET_EVENTOS, engine="openpyxl")
    df_config = pd.read_excel(path, sheet_name=SHEET_CONFIG, engine="openpyxl")

    # Validate columns
    _require_columns(df_nodo, REQ_NODO, SHEET_NODO)
    _require_columns(df_tubo, REQ_TUBO, SHEET_TUBERIA)
    _require_columns(df_evento, REQ_EVENTO, SHEET_EVENTOS)
    _require_columns(df_config, REQ_CONFIG, SHEET_CONFIG)

    # -----------------------------
    # Config
    # -----------------------------
    config: Dict[str, Any] = {}
    for i, r in df_config.iterrows():
        key = _norm_lower(r["clave"])
        if not key:
            continue
        val = r["valor"]

        # Try to coerce to float if looks numeric
        if isinstance(val, str):
            v = val.strip()
            if v == "":
                continue
            try:
                val_num = float(v)
                config[key] = val_num
                continue
            except Exception:
                config[key] = v
                continue

        if isinstance(val, float) and pd.isna(val):
            continue

        config[key] = val

    # -----------------------------
    # Nodes
    # -----------------------------
    node_uid_by_excel_id: Dict[str, str] = {}
    nodes: Dict[str, Node] = {}

    # Ensure unique nodo_id
    nodo_ids = [_norm_str(x) for x in df_nodo["nodo_id"].tolist() if _norm_str(x)]
    dup_nodos = sorted({x for x in nodo_ids if nodo_ids.count(x) > 1})
    if dup_nodos:
        raise ValueError(f"Duplicate nodo_id in sheet '{SHEET_NODO}': {dup_nodos}")

    for idx, r in df_nodo.iterrows():
        nodo_id = _norm_str(r["nodo_id"])
        if not nodo_id:
            continue  # allow blank rows
        name = _norm_str(r["nombre"]) or nodo_id
        z = _as_float(r["cota_m"], "cota_m", SHEET_NODO, f"nodo_id={nodo_id}")

        bc_sp = _norm_lower(r["tipo_cond_borde"]) or "none"
        if bc_sp not in BC_MAP:
            raise ValueError(
                f"Invalid tipo_cond_borde in '{SHEET_NODO}' (nodo_id={nodo_id}): "
                f"{bc_sp!r}. Allowed: {sorted(BC_MAP.keys())}"
            )
        bc_type = BC_MAP[bc_sp]

        bc_value = r["valor_cond_borde"]
        bc_value_norm = _maybe_float(bc_value)
        if bc_type != "none" and bc_value_norm is None:
            raise ValueError(
                f"Missing valor_cond_borde in '{SHEET_NODO}' for nodo_id={nodo_id} "
                f"when tipo_cond_borde={bc_sp!r}"
            )

        external_id = _norm_str(r.get("external_id", "")) or None
        meta = {}
        meta.update(_parse_tags(r.get("tags", "")))

        uid = str(uuid.uuid4())
        node_uid_by_excel_id[nodo_id] = uid
        nodes[uid] = Node(
            uid=uid,
            name=name,
            z=z,
            bc_type=bc_type,      # type: ignore[arg-type]
            bc_value=bc_value_norm,
            external_id=external_id,
            metadata=meta,
        )

    # -----------------------------
    # Pipes
    # -----------------------------
    pipe_uid_by_excel_id: Dict[str, str] = {}
    pipes: Dict[str, Pipe] = {}

    tubo_ids = [_norm_str(x) for x in df_tubo["tubo_id"].tolist() if _norm_str(x)]
    dup_tubos = sorted({x for x in tubo_ids if tubo_ids.count(x) > 1})
    if dup_tubos:
        raise ValueError(f"Duplicate tubo_id in sheet '{SHEET_TUBERIA}': {dup_tubos}")

    for idx, r in df_tubo.iterrows():
        tubo_id = _norm_str(r["tubo_id"])
        if not tubo_id:
            continue

        name = _norm_str(r["nombre"]) or tubo_id
        n_from_excel = _norm_str(r["nodo_inicio"])
        n_to_excel = _norm_str(r["nodo_fin"])

        if n_from_excel not in node_uid_by_excel_id:
            raise ValueError(f"Unknown nodo_inicio '{n_from_excel}' in '{SHEET_TUBERIA}' (tubo_id={tubo_id})")
        if n_to_excel not in node_uid_by_excel_id:
            raise ValueError(f"Unknown nodo_fin '{n_to_excel}' in '{SHEET_TUBERIA}' (tubo_id={tubo_id})")

        node_from = node_uid_by_excel_id[n_from_excel]
        node_to = node_uid_by_excel_id[n_to_excel]

        L = _as_float(r["longitud_m"], "longitud_m", SHEET_TUBERIA, f"tubo_id={tubo_id}")
        D = _as_float(r["diametro_int_m"], "diametro_int_m", SHEET_TUBERIA, f"tubo_id={tubo_id}")

        material = _norm_str(r.get("material", "")) or None
        thickness = _maybe_float(r.get("espesor_m", None))
        elasticity = _maybe_float(r.get("e_pa", None)) or _maybe_float(r.get("E_Pa", None))  # tolerate casing
        poisson = _maybe_float(r.get("nu", None))
        wave_speed = _maybe_float(r.get("a_mps", None))
        friction = _maybe_float(r.get("f_darcy", None))
        dx_target = _maybe_float(r.get("dx_obj_m", None))

        external_id = _norm_str(r.get("external_id", "")) or None

        uid = str(uuid.uuid4())
        pipe_uid_by_excel_id[tubo_id] = uid
        pipes[uid] = Pipe(
            uid=uid,
            name=name,
            node_from=node_from,
            node_to=node_to,
            length=L,
            diameter=D,
            material=material,
            thickness=thickness,
            elasticity=elasticity,
            poisson=poisson,
            wave_speed=wave_speed,
            friction=friction,
            dx_target=dx_target,
            external_id=external_id,
            metadata={"excel_tubo_id": tubo_id, "excel_from": n_from_excel, "excel_to": n_to_excel},
        )

    # -----------------------------
    # Events
    # -----------------------------
    events: list[Event] = []
    evento_ids = [_norm_str(x) for x in df_evento["evento_id"].tolist() if _norm_str(x)]
    dup_eventos = sorted({x for x in evento_ids if evento_ids.count(x) > 1})
    if dup_eventos:
        raise ValueError(f"Duplicate evento_id in sheet '{SHEET_EVENTOS}': {dup_eventos}")

    for idx, r in df_evento.iterrows():
        evento_id = _norm_str(r["evento_id"])
        if not evento_id:
            continue

        tipo_sp = _norm_lower(r["tipo_evento"])
        if tipo_sp not in EVENT_MAP:
            raise ValueError(
                f"Invalid tipo_evento in '{SHEET_EVENTOS}' (evento_id={evento_id}): "
                f"{tipo_sp!r}. Allowed: {sorted(EVENT_MAP.keys())}"
            )
        event_type = EVENT_MAP[tipo_sp]

        obj_tipo = _norm_lower(r["objetivo_tipo"])
        if obj_tipo not in ALLOWED_OBJETIVO_TIPO:
            raise ValueError(
                f"Invalid objetivo_tipo in '{SHEET_EVENTOS}' (evento_id={evento_id}): "
                f"{obj_tipo!r}. Allowed: {sorted(ALLOWED_OBJETIVO_TIPO)}"
            )

        obj_id = _norm_str(r["objetivo_id"])
        if obj_tipo == "nodo":
            if obj_id not in node_uid_by_excel_id:
                raise ValueError(f"Unknown objetivo_id '{obj_id}' (nodo) in '{SHEET_EVENTOS}' (evento_id={evento_id})")
            target_uid = node_uid_by_excel_id[obj_id]
        else:
            if obj_id not in pipe_uid_by_excel_id:
                raise ValueError(f"Unknown objetivo_id '{obj_id}' (tubo) in '{SHEET_EVENTOS}' (evento_id={evento_id})")
            target_uid = pipe_uid_by_excel_id[obj_id]

        t_start = _as_float(r["t_inicio_s"], "t_inicio_s", SHEET_EVENTOS, f"evento_id={evento_id}")
        t_end = _as_float(r["t_fin_s"], "t_fin_s", SHEET_EVENTOS, f"evento_id={evento_id}")
        if t_end < t_start:
            raise ValueError(f"t_fin_s < t_inicio_s in '{SHEET_EVENTOS}' (evento_id={evento_id})")

        params = _parse_params_json(r.get("parametros_json", ""))

        uid = str(uuid.uuid4())
        events.append(Event(
            uid=uid,
            event_type=event_type,   # type: ignore[arg-type]
            target_uid=target_uid,
            t_start=t_start,
            t_end=t_end,
            parameters=params,
        ))

    network = Network(nodes=nodes, pipes=pipes, events=events)
    excel_ids = ExcelIds(node_uid_by_excel_id=node_uid_by_excel_id, pipe_uid_by_excel_id=pipe_uid_by_excel_id)

    return network, config, excel_ids
