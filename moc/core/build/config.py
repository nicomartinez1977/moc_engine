# moc/core/build/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal


# ============================================================
# InitConfig (condición inicial)
# ============================================================

InitMode = Literal["flat", "steady_friction"]


@dataclass(frozen=True)
class InitConfig:
    """
    Configuración de condición inicial del MOC.
    """
    mode: InitMode = "steady_friction"

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "InitConfig":
        mode = str(
            cfg.get("init_mode",
            cfg.get("init",
            cfg.get("initial_condition", "steady_friction")))
        ).strip().lower()

        out = InitConfig(mode=mode)
        out.validate()
        return out

    def validate(self) -> None:
        if self.mode not in ("flat", "steady_friction"):
            raise ValueError(f"InitConfig.mode inválido: {self.mode!r}")


# ============================================================
# MocRunConfig (tiempos / output)
# ============================================================

@dataclass(frozen=True)
class MocRunConfig:
    """
    Configuración de corrida MOC (tiempo y muestreo).
    """
    t_end_s: float = 30.0
    dt_out_s: float = 0.0

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "MocRunConfig":
        t_end = cfg.get("t_end_s", cfg.get("t_end", 30.0))
        dt_out = cfg.get("dt_out_s", cfg.get("dt_out", cfg.get("write_every_s", 0.0)))

        out = MocRunConfig(
            t_end_s=float(t_end),
            dt_out_s=float(dt_out),
        )
        out.validate()
        return out

    def validate(self) -> None:
        if self.t_end_s <= 0:
            raise ValueError(f"MocRunConfig.t_end_s debe ser > 0 (recibido {self.t_end_s})")
        if self.dt_out_s < 0:
            raise ValueError(f"MocRunConfig.dt_out_s debe ser >= 0 (recibido {self.dt_out_s})")


# ============================================================
# HeadlossConfig
# ============================================================

HeadlossModelName = Literal["darcy_weisbach", "chezy"]


@dataclass(frozen=True)
class HeadlossConfig:
    """
    Configuración de pérdidas de carga.
    """
    model: HeadlossModelName = "darcy_weisbach"
    include_minor: bool = False
    g_m_s2: float = 9.80665

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "HeadlossConfig":
        model = str(
            cfg.get("headloss_model",
            cfg.get("headloss",
            cfg.get("hl_model", "darcy_weisbach")))
        ).strip().lower()

        include_minor = cfg.get(
            "include_minor",
            cfg.get("headloss_include_minor", cfg.get("hl_minor", False)),
        )

        g = cfg.get("g_m_s2", cfg.get("g", cfg.get("gravity", 9.80665)))

        if isinstance(include_minor, str):
            include_minor = include_minor.strip().lower() in ("1", "true", "yes", "si", "sí", "y")
        else:
            include_minor = bool(include_minor)

        out = HeadlossConfig(
            model=model,
            include_minor=include_minor,
            g_m_s2=float(g),
        )
        out.validate()
        return out

    def validate(self) -> None:
        if self.model not in ("darcy_weisbach", "chezy"):
            raise ValueError(f"HeadlossConfig.model inválido: {self.model!r}")
        if not (0.0 < self.g_m_s2 < 20.0):
            raise ValueError(f"HeadlossConfig.g_m_s2 fuera de rango: {self.g_m_s2}")


# ============================================================
# ModelConfig (agregador)
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    run: MocRunConfig
    init: InitConfig
    headloss: HeadlossConfig
    version: int = 1

    @staticmethod
    def from_dict(cfg: Dict[str, Any], *, network_gravity: float | None = None) -> "ModelConfig":
        run = MocRunConfig.from_dict(cfg)
        init = InitConfig.from_dict(cfg)

        headloss = HeadlossConfig.from_dict(cfg)
        if network_gravity is not None:
            headloss = HeadlossConfig(
                model=headloss.model,
                include_minor=headloss.include_minor,
                g_m_s2=float(network_gravity),
            )

        out = ModelConfig(
            run=run,
            init=init,
            headloss=headloss,
            version=int(cfg.get("config_version", cfg.get("version", 1))),
        )
        out.validate()
        return out

    def validate(self) -> None:
        if self.version <= 0:
            raise ValueError(f"ModelConfig.version debe ser > 0 (recibido {self.version})")

        self.run.validate()
        self.init.validate()
        self.headloss.validate()
