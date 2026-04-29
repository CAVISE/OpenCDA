"""Orchestration manager for event-driven service attacks."""

from __future__ import annotations

from collections.abc import Iterable

from opencda.scenario_testing.types import SimulationSnapshot

from .attack import Attack
from .models import AttackResult, AttackStageResult, RuntimeStatus, Status
from .utils import ServiceResolver


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
            if attack.is_active:
                if attack.should_stop(previous_snapshot, current_snapshot):
                    stage_history = attack.get_stage_history()
                    attack.deactivate()
                    attack.mark_stopped()
                    results.append(
                        AttackResult(
                            attack_name=attack.attack_name,
                            status=Status.STOP,
                            reason="Attack stop trigger fired.",
                            stage_history=stage_history,
                        )
                    )
                    continue

                target_services = tuple(attack.resolve_targets(current_snapshot, service_resolver))
                stage_results = attack.run_stage_lifecycle(previous_snapshot, current_snapshot, target_services)
                if stage_results:
                    results.append(self._build_attack_result(attack, stage_results))
                continue

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

            attack.mark_started()
            stage_results = attack.run_stage_lifecycle(previous_snapshot, current_snapshot, target_services)
            if attack.status == RuntimeStatus.STARTED:
                attack.mark_active()
            if stage_results:
                results.append(self._build_attack_result(attack, stage_results))

        self.previous_snapshot = current_snapshot
        return tuple(results)

    @staticmethod
    def _build_attack_result(
        attack: Attack,
        stage_results: tuple[AttackStageResult, ...],
    ) -> AttackResult:
        last_result = stage_results[-1]
        if attack.status == RuntimeStatus.FAIL:
            return AttackResult(
                attack_name=attack.attack_name,
                status=Status.FAIL,
                reason=last_result.reason,
                stage_history=attack.get_stage_history(),
            )

        if attack.status == RuntimeStatus.STOPPED or last_result.status == Status.STOP:
            return AttackResult(
                attack_name=attack.attack_name,
                status=Status.STOP,
                reason=last_result.reason,
                stage_history=attack.get_stage_history(),
            )

        return AttackResult(
            attack_name=attack.attack_name,
            status=Status.SUCCESS,
            reason=last_result.reason,
            stage_history=attack.get_stage_history(),
        )
