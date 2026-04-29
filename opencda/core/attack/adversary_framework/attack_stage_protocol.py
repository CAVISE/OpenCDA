"""Protocol for attack pipeline stages."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol, Sequence, runtime_checkable

from opencda.core.application.behavior.capability import Capability
from opencda.core.application.behavior.behavior_service_protocol import BehaviorService

from .models import AttackStageResult


@runtime_checkable
class AttackStage(Protocol):
    """Single executable attack stage."""

    stage_name: str
    required_capabilities: Collection[Capability]
    description: str

    def execute(self, services: Sequence[BehaviorService[Any, Any]]) -> AttackStageResult:
        """Execute a single stage against the provided service pool."""
