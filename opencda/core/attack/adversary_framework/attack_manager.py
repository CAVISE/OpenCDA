"""Orchestration manager for capability-based service attacks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from opencda.core.application.behavior.capability import Capability
from opencda.core.application.behavior.behavior_service_protocol import BehaviorService

from .attack_protocol import Attack
from .attack_result import AttackResult, AttackStageResult, Status
from .utils import match_services


class AttackManager:
    """Manage attack execution inside the current simulation."""

    def __init__(self, services: Iterable[BehaviorService[Any, Any]] = ()) -> None:
        self._services = list(services)

    def run(
        self,
        attack: Attack,
        *,
        target_service: BehaviorService[Any, Any] | None = None,
        service_filter: Callable[[BehaviorService[Any, Any]], bool] | None = None,
    ) -> AttackResult:
        """Run a single attack as a linear stage pipeline."""
        available_services = self._match_services(
            attack,
            target_service=target_service,
            service_filter=service_filter,
        )
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

    def _match_services(
        self,
        attack: Attack,
        *,
        target_service: BehaviorService[Any, Any] | None = None,
        service_filter: Callable[[BehaviorService[Any, Any]], bool] | None = None,
    ) -> tuple[BehaviorService[Any, Any], ...]:
        candidate_services = [target_service] if target_service is not None else list(self._services)
        if service_filter is not None:
            candidate_services = [service for service in candidate_services if service_filter(service)]

        required_capabilities = self._collect_required_capabilities(attack)
        matched_services = match_services(candidate_services, required_capabilities)
        if not matched_services:
            raise RuntimeError(
                f"No matching service found for attack '{attack.attack_name}' and required "
                f"capabilities {sorted(cap.value for cap in required_capabilities)}."
            )

        return tuple(matched_services)

    @staticmethod
    def _collect_required_capabilities(attack: Attack) -> tuple[Capability, ...]:
        required_capabilities: list[Capability] = []
        seen_capabilities: set[Capability] = set()
        for stage in attack.stages:
            for capability in stage.required_capabilities:
                if capability in seen_capabilities:
                    continue
                seen_capabilities.add(capability)
                required_capabilities.append(capability)
        return tuple(required_capabilities)
