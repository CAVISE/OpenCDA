"""Protocol for attack pipeline stages."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol, Sequence, runtime_checkable

from opencda.core.application.behavior.capability import Capability
from opencda.core.application.behavior.behavior_service_protocol import BehaviorService

from .models import AttackStageResult


@runtime_checkable
class AttackStage(Protocol):
    """Single executable attack stage.

    Stages with ongoing effects may declare ``persistent_effect = True`` and
    provide an idempotent ``deactivate()`` method. Installation success then
    leaves the stage ACTIVE until its stop trigger or attack termination.
    Other stages complete after execute() unless a stage stop trigger is set.
    """

    stage_name: str
    supported_capabilities: Collection[Capability]
    default_capabilities: Collection[Capability]
    capabilities: Collection[Capability]
    description: str  # noqa: DC01

    def execute(self, services: Sequence[BehaviorService[Any, Any]]) -> AttackStageResult:
        """Execute a single stage against the provided service pool."""
