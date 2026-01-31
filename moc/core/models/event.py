from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Literal

EventType = Literal[
    "valve_close",
    "valve_open",
    "pump_trip",
    "pump_start",
]


@dataclass(frozen=True, slots=True)
class Event:
    """
    Canonical time event affecting a target (node or pipe).

    Notes:
    - target_uid refers to Node.uid or Pipe.uid depending on event_type/adapter.
    - parameters holds event-specific configuration (closure law, etc.)
    """
    uid: str
    event_type: EventType
    target_uid: str

    t_start: float  # [s]
    t_end: float    # [s]

    parameters: Dict[str, Any] = field(default_factory=dict)
