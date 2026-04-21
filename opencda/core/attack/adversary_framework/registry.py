"""Registry for discoverable attacks."""

from __future__ import annotations

import inspect
import logging
from typing import Any, TypeVar

from .attack_protocol import Attack

logger = logging.getLogger("cavise.opencda.opencda.core.attack.adversary_framework.registry")

AttackT = TypeVar("AttackT", bound=Attack)


class AttackRegistry:
    """Registry for attack classes."""

    _registry: dict[str, type[Attack]] = {}

    @classmethod
    def register(cls, attack_cls: type[AttackT]) -> type[AttackT]:
        """Register a concrete attack class."""
        if inspect.isabstract(attack_cls):
            raise ValueError(f"Cannot register abstract attack class '{attack_cls.__name__}'.")

        if (attack_name := getattr(attack_cls, "attack_name", None)) is None:
            raise ValueError(f"Attack class '{attack_cls.__name__}' must define 'attack_name'.")

        if attack_name in cls._registry:
            raise ValueError(f"Duplicate attack registration for attack='{attack_name}'.")

        cls._registry[attack_name] = attack_cls
        logger.info("Registered attack class '%s' as '%s'.", attack_cls.__name__, attack_name)
        return attack_cls

    @classmethod
    def get_attack_class(cls, attack_name: str) -> type[Attack]:
        """Return an attack class for the given attack name."""
        if attack_name not in cls._registry:
            available = cls.list_attacks()
            raise KeyError(f"Unknown attack '{attack_name}'. Available: {available}")
        return cls._registry[attack_name]

    @classmethod
    def create_attack(cls, attack_name: str, **kwargs: Any) -> Attack:
        """Instantiate an attack by name."""
        attack_cls = cls.get_attack_class(attack_name=attack_name)
        return attack_cls(**kwargs)

    @classmethod
    def list_attacks(cls) -> list[str]:
        """List registered attacks."""
        return sorted(cls._registry)
