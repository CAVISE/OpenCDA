"""Orchestration manager for event-driven service attacks."""

from __future__ import annotations

from collections.abc import Iterable
import logging

from opencda.scenario_testing.types import SimulationSnapshot

from .attack import Attack
from .models import AttackResult, AttackStageResult, RuntimeStatus, Status
from .utils import ServiceResolver

logger = logging.getLogger("cavise.opencda.opencda.core.attack.adversary_framework.attack_manager")


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
                    attack_result = AttackResult(
                        attack_name=attack.attack_name,
                        status=Status.STOP,
                        reason="Attack stop trigger fired.",
                        stage_history=stage_history,
                    )
                    self._log_attack_result(attack_result)
                    results.append(attack_result)
                    continue

                target_services = tuple(attack.resolve_targets(current_snapshot, service_resolver))
                stage_results = attack.run_stage_lifecycle(previous_snapshot, current_snapshot, target_services)
                if stage_results:
                    attack_result = self._build_attack_result(attack, stage_results)
                    self._log_attack_result(attack_result)
                    results.append(attack_result)
                continue

            if not attack.should_start(previous_snapshot, current_snapshot):
                continue

            target_services = tuple(attack.resolve_targets(current_snapshot, service_resolver))
            if not target_services:
                attack_result = AttackResult(
                    attack_name=attack.attack_name,
                    status=Status.FAIL,
                    reason="Attack trigger fired, but no target services were resolved.",
                )
                self._log_attack_result(attack_result)
                results.append(attack_result)
                continue

            attack.mark_started()
            stage_results = attack.run_stage_lifecycle(previous_snapshot, current_snapshot, target_services)
            if attack.status == RuntimeStatus.STARTED:
                attack.mark_active()
            if stage_results:
                attack_result = self._build_attack_result(attack, stage_results)
                self._log_attack_result(attack_result)
                results.append(attack_result)

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

    @staticmethod
    def _log_attack_result(attack_result: AttackResult) -> None:
        log_message = "Attack '%s' result: status=%s, reason=%s."
        log_args = (
            attack_result.attack_name,
            attack_result.status.value,
            attack_result.reason,
        )
        if attack_result.status == Status.FAIL:
            logger.warning(log_message, *log_args)
            return
        logger.info(log_message, *log_args)
