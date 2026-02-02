from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Iterable

import pandas as pd

from moc.core.models.node import Node
from moc.core.models.pipe import Pipe
from moc.core.models.event import Event
from moc.core.models.network import Network


# =========================
# Nombres de hojas
# =========================
SHEET_NODO = "nodo"
SHEET_TUBERIA = "tuberia"
SHEET_EVENTOS = "eventos"
SHEET_CONFIG = "config"

# =========================
# Columnas requeridas (fila 1 del Excel)
# =========================
REQ_NODO = {"nodo_id", "Nombre", "Cota", "Condicion de Borde", "Valor Condcion de Borde"}
REQ_TUBO = {"tubo_id", "Nombre Tramo", "nodo inicial", "nodo final", "Longitud", "Diametro Nominal", "Material", "Espesor"}
REQ_EVENTO = {"evento_id", "Tipo de Evento", "Tipo de Objetivo", "objetivo_id", "Tiempo Inicio", "Tiempo Fin"}
REQ_CONFIG = {"Variable", "Valor"}

# =========================
# Mapeos (Excel español -> core inglés)
# =========================
BC_MAP = {
    "Ninguno": "none",
    "Estanque": "reservoir",
    "Demanda": "demand",
    "Valvula": "valve",
    "Bomba": "pump",
}

EVENT_MAP = {
    "Cierre de Valvula": "valve_close",
    "Apertura de Valvula": "valve_open",
    "Parada de Bomba": "pump_trip",
    "Partida de Bomba": "pump_start",
}

MATERIAL_MAP ={
    "HDPE": "hdpe",
    "PVC": "pvc",
    "Acero": "steel",
    "Hierro Ductil": "ductile_iron",
    "PRFV": "frp",
}



ALLOWED_OBJETIVO_TIPO = {"nodo", "tubo"}


# =========================
# Unidades esperadas (fila 2 del Excel)
# =========================
# Nota: permitimos algunos sinónimos (ej: '-' o '' para adimensional)
EXPECTED_UNITS_NODO = {
    "nodo_id": {"-", "", None},
    "nombre": {"-", "", None},
    "cota_m": {"m"},
    "tipo_cond_borde": {"-", "", None},
    # valor_cond_borde: en tu modelo representa una "carga" H (m)
    "valor_cond_borde": {"m", "mca"},  # aceptamos m o mca para ser tolerantes
}

EXPECTED_UNITS_TUBO = {
    "tubo_id": {"-", "", None},
    "nombre": {"-", "", None},
    "nodo_inicio": {"-", "", None},
    "nodo_fin": {"-", "", None},
    "longitud_m": {"m"},
    "diametro_int_mm": {"mm"},
    # opcional
    "espesor_mm": {"mm"},
    "E_Pa": {"Pa"},
    "nu": {"-", "", None},
    "a_mps": {"m/s", "mps", "m/s "},  # tolerante
    "f_darcy": {"-", "", None},
    "dx_obj_m": {"m"},
    "external_id": {"-", "", None},
}

EXPECTED_UNITS_EVENTOS = {
    "evento_id": {"-", "", None},
    "tipo_evento": {"-", "", None},
    "objetivo_tipo": {"-", "", None},
    "objetivo_id": {"-", "", None},
    "t_inicio_s": {"s"},
    "t_fin_s": {"s"},
    "parametros_json": {"-", "", None},
}

EXPECTED_UNITS_CONFIG = {
    "clave": {"-", "", None},
    # valor puede ser numérico o texto; unidad puede variar, aceptamos '-'
    "valor": {"-", "", None, "m", "mca", "s", "mm"},
}


@dataclass(frozen=True)
class ExcelIds:
    """Mapeos Excel ID -> UID interno"""
    node_uid_by_excel_id: Dict[str, str]
    pipe_uid_by_excel_id: Dict[str, str]


# =========================
# Utils
# =========================
def _norm_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def _norm_lower(x: Any) -> str:
    return _norm_str(x).lower()


def _maybe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == ""):
            return None
        return float(x)
    except Exception:
        return None


def _as_float(x: Any, field: str, sheet: str, row_hint: str) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == ""):
            raise ValueError("vacío")
        return float(x)
    except Exception as e:
        raise ValueError(f"[{sheet}] Valor numérico inválido en '{field}' ({row_hint}): {x!r}") from e


def _require_columns(df: pd.DataFrame, required: set[str], sheet: str) -> None:
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"[{sheet}] Faltan columnas requeridas: {missing}")


def _parse_tags(tags_cell: Any) -> Dict[str, Any]:
    s = _norm_str(tags_cell)
    if not s:
        return {}
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return {"tags": parts} if parts else {}


def _parse_params_json(cell: Any) -> Dict[str, Any]:
    s = _norm_str(cell)
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido en 'parametros_json': {s!r}") from e


def _unit_ok(unit_read: Any, allowed: Iterable[Any]) -> bool:
    u = _norm_str(unit_read)
    # permitimos None/''
    if unit_read is None and None in allowed:
        return True
    return (u in {""} and "" in allowed) or (u in allowed)


def _read_sheet_with_units(
    path: str,
    sheet_name: str,
    expected_units: Optional[Dict[str, set]] = None,
) -> pd.DataFrame:
    """
    Lee una hoja con 2 filas de encabezado:
      fila 1: nombres de columnas (snake_case)
      fila 2: unidades

    Devuelve DataFrame con columnas = fila 1 (sin unidades).
    Valida unidades si expected_units se entrega (solo para columnas presentes).
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=[0, 1], engine="openpyxl")

    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        raise ValueError(
            f"[{sheet_name}] La hoja debe tener 2 filas de encabezado: "
            f"(1) nombres de variables y (2) unidades."
        )

    # Extraer nombres y unidades
    names = [str(c[0]).strip() for c in df.columns]
    units = {str(c[0]).strip(): c[1] for c in df.columns}

    # Aplicar nombres simples
    df.columns = names

    # Validar unidades (solo si el usuario entregó expected_units)
    if expected_units:
        for col, allowed in expected_units.items():
            if col in df.columns:
                u_read = units.get(col, None)
                if not _unit_ok(u_read, allowed):
                    raise ValueError(
                        f"[{sheet_name}] Unidad incorrecta en columna '{col}': "
                        f"se esperaba una de {sorted([str(a) for a in allowed])}, "
                        f"se leyó '{_norm_str(u_read)}'."
                    )

    return df


# =========================
# API principal
# =========================
def load_network_from_excel(path: str) -> Tuple[Network, Dict[str, Any], ExcelIds]:
    """
    Lee Excel con 2 filas de encabezado (variables + unidades) en:
      - nodo
      - tuberia
      - eventos
      - config

    Retorna:
      - Network (core)
      - config dict (claves normalizadas a lower)
      - ExcelIds (mapeos Excel ID -> UID)
    """
    # Leer hojas con unidades
    df_nodo = _read_sheet_with_units(path, SHEET_NODO, EXPECTED_UNITS_NODO)
    df_tubo = _read_sheet_with_units(path, SHEET_TUBERIA, EXPECTED_UNITS_TUBO)
    df_evento = _read_sheet_with_units(path, SHEET_EVENTOS, EXPECTED_UNITS_EVENTOS)
    df_config = _read_sheet_with_units(path, SHEET_CONFIG, EXPECTED_UNITS_CONFIG)

    # Validar columnas requeridas
    _require_columns(df_nodo, REQ_NODO, SHEET_NODO)
    _require_columns(df_tubo, REQ_TUBO, SHEET_TUBERIA)
    _require_columns(df_evento, REQ_EVENTO, SHEET_EVENTOS)
    _require_columns(df_config, REQ_CONFIG, SHEET_CONFIG)

    # -----------------------------
    # CONFIG
    # -----------------------------
    config: Dict[str, Any] = {}
    for _, r in df_config.iterrows():
        key = _norm_lower(r["clave"])
        if not key:
            continue
        val = r["valor"]

        # coerce numéricos si corresponde
        if isinstance(val, str):
            v = val.strip()
            if v == "":
                continue
            try:
                config[key] = float(v)
            except Exception:
                config[key] = v
        else:
            if isinstance(val, float) and pd.isna(val):
                continue
            config[key] = val

    # -----------------------------
    # NODES
    # -----------------------------
    node_uid_by_excel_id: Dict[str, str] = {}
    nodes: Dict[str, Node] = {}

    nodo_ids = [_norm_str(x) for x in df_nodo["nodo_id"].tolist() if _norm_str(x)]
    dup_nodos = sorted({x for x in nodo_ids if nodo_ids.count(x) > 1})
    if dup_nodos:
        raise ValueError(f"[{SHEET_NODO}] nodo_id duplicado: {dup_nodos}")

    for _, r in df_nodo.iterrows():
        nodo_id = _norm_str(r["nodo_id"])
        if not nodo_id:
            continue

        name = _norm_str(r["nombre"]) or nodo_id
        z = _as_float(r["cota_m"], "cota_m", SHEET_NODO, f"nodo_id={nodo_id}")

        bc_sp = _norm_lower(r["tipo_cond_borde"]) or "none"
        if bc_sp not in BC_MAP:
            raise ValueError(
                f"[{SHEET_NODO}] tipo_cond_borde inválido en nodo_id={nodo_id}: {bc_sp!r}. "
                f"Permitidos: {sorted(BC_MAP.keys())}"
            )
        bc_type = BC_MAP[bc_sp]

        bc_value_norm = _maybe_float(r["valor_cond_borde"])
        if bc_type != "none" and bc_value_norm is None:
            raise ValueError(
                f"[{SHEET_NODO}] Falta valor_cond_borde para nodo_id={nodo_id} cuando tipo_cond_borde={bc_sp!r}"
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
    # PIPES
    # -----------------------------
    pipe_uid_by_excel_id: Dict[str, str] = {}
    pipes: Dict[str, Pipe] = {}

    tubo_ids = [_norm_str(x) for x in df_tubo["tubo_id"].tolist() if _norm_str(x)]
    dup_tubos = sorted({x for x in tubo_ids if tubo_ids.count(x) > 1})
    if dup_tubos:
        raise ValueError(f"[{SHEET_TUBERIA}] tubo_id duplicado: {dup_tubos}")

    for _, r in df_tubo.iterrows():
        tubo_id = _norm_str(r["tubo_id"])
        if not tubo_id:
            continue

        name = _norm_str(r["nombre"]) or tubo_id

        n_from_excel = _norm_str(r["nodo_inicio"])
        n_to_excel = _norm_str(r["nodo_fin"])

        if n_from_excel not in node_uid_by_excel_id:
            raise ValueError(f"[{SHEET_TUBERIA}] nodo_inicio desconocido '{n_from_excel}' (tubo_id={tubo_id})")
        if n_to_excel not in node_uid_by_excel_id:
            raise ValueError(f"[{SHEET_TUBERIA}] nodo_fin desconocido '{n_to_excel}' (tubo_id={tubo_id})")

        node_from = node_uid_by_excel_id[n_from_excel]
        node_to = node_uid_by_excel_id[n_to_excel]

        L = _as_float(r["longitud_m"], "longitud_m", SHEET_TUBERIA, f"tubo_id={tubo_id}")

        # mm -> m
        D_mm = _as_float(r["diametro_int_mm"], "diametro_int_mm", SHEET_TUBERIA, f"tubo_id={tubo_id}")
        D = D_mm / 1000.0

        e_mm = _maybe_float(r.get("espesor_mm", None))
        thickness = (e_mm / 1000.0) if e_mm is not None else None

        material = _norm_str(r.get("material", "")) or None

        elasticity = _maybe_float(r.get("E_Pa", None))
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
    # EVENTS
    # -----------------------------
    events: list[Event] = []

    evento_ids = [_norm_str(x) for x in df_evento["evento_id"].tolist() if _norm_str(x)]
    dup_eventos = sorted({x for x in evento_ids if evento_ids.count(x) > 1})
    if dup_eventos:
        raise ValueError(f"[{SHEET_EVENTOS}] evento_id duplicado: {dup_eventos}")

    for _, r in df_evento.iterrows():
        evento_id = _norm_str(r["evento_id"])
        if not evento_id:
            continue

        tipo_sp = _norm_lower(r["tipo_evento"])
        if tipo_sp not in EVENT_MAP:
            raise ValueError(
                f"[{SHEET_EVENTOS}] tipo_evento inválido en evento_id={evento_id}: {tipo_sp!r}. "
                f"Permitidos: {sorted(EVENT_MAP.keys())}"
            )
        event_type = EVENT_MAP[tipo_sp]

        obj_tipo = _norm_lower(r["objetivo_tipo"])
        if obj_tipo not in ALLOWED_OBJETIVO_TIPO:
            raise ValueError(
                f"[{SHEET_EVENTOS}] objetivo_tipo inválido en evento_id={evento_id}: {obj_tipo!r}. "
                f"Permitidos: {sorted(ALLOWED_OBJETIVO_TIPO)}"
            )

        obj_id = _norm_str(r["objetivo_id"])
        if obj_tipo == "nodo":
            if obj_id not in node_uid_by_excel_id:
                raise ValueError(f"[{SHEET_EVENTOS}] objetivo_id nodo desconocido '{obj_id}' (evento_id={evento_id})")
            target_uid = node_uid_by_excel_id[obj_id]
        else:
            if obj_id not in pipe_uid_by_excel_id:
                raise ValueError(f"[{SHEET_EVENTOS}] objetivo_id tubo desconocido '{obj_id}' (evento_id={evento_id})")
            target_uid = pipe_uid_by_excel_id[obj_id]

        t_start = _as_float(r["t_inicio_s"], "t_inicio_s", SHEET_EVENTOS, f"evento_id={evento_id}")
        t_end = _as_float(r["t_fin_s"], "t_fin_s", SHEET_EVENTOS, f"evento_id={evento_id}")
        if t_end < t_start:
            raise ValueError(f"[{SHEET_EVENTOS}] t_fin_s < t_inicio_s en evento_id={evento_id}")

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

    # -----------------------------
    # Network
    # -----------------------------
    network = Network(nodes=nodes, pipes=pipes, events=events)
    excel_ids = ExcelIds(node_uid_by_excel_id=node_uid_by_excel_id, pipe_uid_by_excel_id=pipe_uid_by_excel_id)

    return network, config, excel_ids
