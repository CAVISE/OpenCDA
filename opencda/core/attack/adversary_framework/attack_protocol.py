"""Protocol for event-driven adversary attacks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.scenario_testing.types import SimulationSnapshot

if TYPE_CHECKING:
    from .attack_stage_protocol import AttackStage

ServiceResolver: TypeAlias = Callable[[str, str], BehaviorService[Any, Any] | None]


@runtime_checkable
class Attack(Protocol):
    """Event-driven attack contract."""

    attack_name: str
    stages: Sequence[AttackStage]
    is_active: bool

    def should_start(
        self,
        previous_snapshot: SimulationSnapshot | None,
        current_snapshot: SimulationSnapshot,
    ) -> bool:
        """Return whether the attack should start on the current tick."""

    def resolve_targets(
        self,
        current_snapshot: SimulationSnapshot,
        service_resolver: ServiceResolver,
    ) -> Sequence[BehaviorService[Any, Any]]:
        """Resolve live target services for the current snapshot."""

    def mark_active(self) -> None:
        """Mark the attack as active after a successful start."""
