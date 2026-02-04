from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True, slots=True)
class Pipe:
    """
    Canonical 1D pipe segment (core model).

    Notes:
    - node_from/node_to reference Node.uid
    - wave_speed (a) can be provided directly by adapters or computed later
    - friction is Darcy-Weisbach f (dimensionless), can be provided or computed later
    """
    """
    Segmento de tubería canónica 1D de Cano (modelo básico).

    Notas:
    - node_from/node_to hace referencia a Node.uid
    - la velocidad de onda (a) puede ser proporcionada directamente por adaptadores o calculada más adelante
    - la fricción es Darcy-Weisbach f (adimensional), puede ser proporcionada o calculada más adelante
    """
    
    uid: str
    name: str

    node_from: str
    node_to: str

    length: float      # [m]
    diameter: float    # internal diameter [m]

    material: Optional[str] = None
    thickness: Optional[float] = None          # [m]
    elasticity: Optional[float] = None         # E [Pa]
    poisson: Optional[float] = None            # nu [-]

    wave_speed: Optional[float] = None         # a [m/s]


    friction: Optional[float] = None      # Darcy f [-] (si None, se calcula)
    roughness: Optional[float] = None     # ε [m] rugosidad absoluta (si None, usar default)
    
    dx_target: Optional[float] = None          # preferred spatial step [m]

    external_id: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


    @property
    def area(self) -> float:
        import math
        return math.pi * (self.diameter ** 2) / 4.0
