"""Orchestration manager for event-driven service attacks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.scenario_testing.types import SimulationSnapshot

from .attack_protocol import Attack, ServiceResolver
from .attack_result import AttackResult, AttackStageResult, Status


class AttackManager:
    """Manage attack execution inside the current simulation."""

    def __init__(self) -> None:
        self.previous_snapshot: SimulationSnapshot | None = None

    def evaluate(
        self,
        attacks: Iterable[Attack],
        current_snapshot: SimulationSnapshot,
        *,
        service_resolver: ServiceResolver,
    ) -> tuple[AttackResult, ...]:
        """Evaluate event-driven attacks against the current simulation snapshot."""
        previous_snapshot = self.previous_snapshot
        results: list[AttackResult] = []

        for attack in attacks:
            if not attack.should_start(previous_snapshot, current_snapshot):
                continue

            target_services = tuple(attack.resolve_targets(current_snapshot, service_resolver))
            if not target_services:
                results.append(
                    AttackResult(
                        attack_name=attack.attack_name,
                        status=Status.FAIL,
                        reason="Attack trigger fired, but no target services were resolved.",
                    )
                )
                continue

            result = self._run_attack(attack, target_services)
            if result.status == Status.SUCCESS:
                attack.mark_active()
            results.append(result)

        self.previous_snapshot = current_snapshot
        return tuple(results)

    def _run_attack(
        self,
        attack: Attack,
        available_services: tuple[BehaviorService[Any, Any], ...],
    ) -> AttackResult:
        """Run attack stages against an already resolved service set."""
        stages = tuple(attack.stages)
        if not stages:
            raise RuntimeError(f"Attack '{attack.attack_name}' does not define any stages.")

        stage_history: list[AttackStageResult] = []

        for stage in stages:
            stage_result = stage.execute(available_services)
            stage_history.append(stage_result)

            if stage_result.status == Status.STOP:
                return AttackResult(
                    attack_name=attack.attack_name,
                    status=Status.STOP,
                    reason=stage_result.reason,
                    stage_history=tuple(stage_history),
                )

            if stage_result.status == Status.FAIL:
                return AttackResult(
                    attack_name=attack.attack_name,
                    status=Status.FAIL,
                    reason=stage_result.reason,
                    stage_history=tuple(stage_history),
                )

        return AttackResult(
            attack_name=attack.attack_name,
            status=Status.SUCCESS,
            reason="Attack pipeline finished successfully.",
            stage_history=tuple(stage_history),
        )
