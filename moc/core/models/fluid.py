@dataclass(frozen=True, slots=True)
class Fluid:
    rho: float      # kg/m3
    mu: float       # Pa*s  (o almacenar nu)
    K: float        # Pa    bulk modulus
    nu: float       # m2/s  kinematic viscosity
    T_C: float      # degrees CelsiusTemperatura del Fluido