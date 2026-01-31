def load_network_from_excel(path: str) -> tuple[Network, dict]:
    """Returns (network, config_dict)."""

"""
    Network: construido con uid internos (UUID)
    config_dict: valores de CONFIG (Q0, t_end, Hvap, etc.)
    Mantener diccionarios de mapeo:
    nodo_id → Node.uid
    tubo_id → Pipe.uid
"""