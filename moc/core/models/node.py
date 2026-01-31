from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

BoundaryType = Literal["none", "reservoir", "valve", "pump", "demand"]


@dataclass(frozen=True, slots=True)
class Node:
    """
    Canonical hydraulic node (core model).

    Notes:
    - uid: internal stable id (recommended UUID string)
    - external_id: id coming from adapters (Excel/CAD), kept for traceability
    - bc_type/bc_value: boundary condition tag/value (physics handled elsewhere)

    """
    
    """
    Nodo hidráulico canónico (modelo central).

    Notas:
    - uid: id interno estable (se recomienda cadena UUID)
    - external_id: id proveniente de adaptadores (Excel/CAD), se mantiene para trazabilidad
    - bc_type/bc_value: etiqueta/valor de condición de frontera (la física se maneja en otro lugar)
    """
    uid: str
    name: str
    z: float  # elevation [m]

    bc_type: BoundaryType = "none"
    bc_value: Optional[float] = None

    external_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
