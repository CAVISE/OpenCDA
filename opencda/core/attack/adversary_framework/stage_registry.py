"""Registry for discoverable attack stages."""

from __future__ import annotations

from collections.abc import Collection
import inspect
import logging
from typing import Any, TypeVar, cast

from opencda.core.application.behavior.capability import Capability

from .attack_stage_protocol import AttackStage

logger = logging.getLogger("cavise.opencda.opencda.core.attack.adversary_framework.stage_registry")

AttackStageT = TypeVar("AttackStageT")


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

        cls._registry[stage_name] = cast(type[AttackStage], stage_cls)
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
        configured_capabilities = kwargs.get("capabilities")
        supported_capabilities = tuple(getattr(stage_cls, "supported_capabilities", ()))
        default_capabilities = tuple(getattr(stage_cls, "default_capabilities", ()))

        if not supported_capabilities:
            raise ValueError(f"Attack stage '{stage_name}' must define non-empty 'supported_capabilities'.")

        if not default_capabilities:
            raise ValueError(f"Attack stage '{stage_name}' must define non-empty 'default_capabilities'.")

        normalized_default_capabilities = cls._normalize_capabilities(default_capabilities)
        cls._validate_supported_capabilities(
            stage_name=stage_name,
            capabilities=normalized_default_capabilities,
            supported_capabilities=supported_capabilities,
            source_label="default",
        )

        if configured_capabilities is None:
            normalized_capabilities = normalized_default_capabilities
        else:
            normalized_capabilities = cls._normalize_capabilities(configured_capabilities)
            cls._validate_supported_capabilities(
                stage_name=stage_name,
                capabilities=normalized_capabilities,
                supported_capabilities=supported_capabilities,
                source_label="configured",
            )

        kwargs["capabilities"] = normalized_capabilities
        return stage_cls(**kwargs)

    @classmethod
    def _validate_supported_capabilities(
        cls,
        stage_name: str,
        capabilities: tuple[Capability, ...],
        supported_capabilities: tuple[Capability, ...],
        source_label: str,
    ) -> None:
        unsupported_capabilities = sorted(
            set(capabilities).difference(supported_capabilities),
            key=lambda capability: capability.value,
        )
        if unsupported_capabilities:
            unsupported = ", ".join(capability.value for capability in unsupported_capabilities)
            supported = ", ".join(capability.value for capability in supported_capabilities)
            raise ValueError(
                f"Attack stage '{stage_name}' does not support {source_label} capabilities [{unsupported}]. Supported capabilities: [{supported}]."
            )

    @classmethod
    def list_stages(cls) -> list[str]:
        """List registered attack stages."""
        return sorted(cls._registry)

    @staticmethod
    def _normalize_capabilities(capabilities: Collection[Capability] | Collection[str]) -> tuple[Capability, ...]:
        normalized_capabilities = tuple(
            capability if isinstance(capability, Capability) else Capability(str(capability)) for capability in capabilities
        )
        if not normalized_capabilities:
            raise ValueError("Configured stage capabilities cannot be empty.")
        return normalized_capabilities
