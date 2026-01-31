from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .node import Node
from .pipe import Pipe
from .event import Event


@dataclass(frozen=True, slots=True)
class Network:
    """
    Canonical network container (core model).
    """
    nodes: Dict[str, Node] = field(default_factory=dict)   # key: Node.uid
    pipes: Dict[str, Pipe] = field(default_factory=dict)   # key: Pipe.uid
    events: List[Event] = field(default_factory=list)

    gravity: float = 9.81
    density: float = 1000.0  # [kg/m3]

    def get_node(self, uid: str) -> Node:
        return self.nodes[uid]

    def get_pipe(self, uid: str) -> Pipe:
        return self.pipes[uid]

    def pipes_from(self, node_uid: str) -> List[Pipe]:
        return [p for p in self.pipes.values() if p.node_from == node_uid]

    def pipes_to(self, node_uid: str) -> List[Pipe]:
        return [p for p in self.pipes.values() if p.node_to == node_uid]
