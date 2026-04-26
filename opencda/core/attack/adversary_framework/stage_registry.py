"""Registry for discoverable attack stages."""

from __future__ import annotations

import inspect
import logging
from typing import Any, TypeVar

from .attack_stage_protocol import AttackStage

logger = logging.getLogger("cavise.opencda.opencda.core.attack.adversary_framework.stage_registry")

AttackStageT = TypeVar("AttackStageT", bound=AttackStage)


class AttackStageRegistry:
    """Registry for attack stage classes."""

    _registry: dict[str, type[AttackStage]] = {}

    @classmethod
    def register(cls, stage_cls: type[AttackStageT]) -> type[AttackStageT]:
        """Register a concrete attack stage class."""
        if inspect.isabstract(stage_cls):
            raise ValueError(f"Cannot register abstract attack stage class '{stage_cls.__name__}'.")

        if (stage_name := getattr(stage_cls, "stage_name", None)) is None:
            raise ValueError(f"Attack stage class '{stage_cls.__name__}' must define 'stage_name'.")

        if stage_name in cls._registry:
            raise ValueError(f"Duplicate attack stage registration for stage='{stage_name}'.")

        cls._registry[stage_name] = stage_cls
        logger.info("Registered attack stage class '%s' as '%s'.", stage_cls.__name__, stage_name)
        return stage_cls

    @classmethod
    def get_stage_class(cls, stage_name: str) -> type[AttackStage]:
        """Return an attack stage class for the given stage name."""
        if stage_name not in cls._registry:
            available = cls.list_stages()
            raise KeyError(f"Unknown attack stage '{stage_name}'. Available: {available}")
        return cls._registry[stage_name]

    @classmethod
    def create_stage(cls, stage_name: str, **kwargs: Any) -> AttackStage:
        """Instantiate an attack stage by name."""
        stage_cls = cls.get_stage_class(stage_name=stage_name)
        return stage_cls(**kwargs)

    @classmethod
    def list_stages(cls) -> list[str]:
        """List registered attack stages."""
        return sorted(cls._registry)
