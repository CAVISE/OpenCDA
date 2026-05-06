"""Replay previously observed service responses."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.core.application.behavior.capability import Capability
from opencda.core.attack.adversary_framework.models import AttackStageResult, Status
from opencda.core.attack.adversary_framework.stage_registry import AttackStageRegistry
from opencda.core.attack.adversary_framework.utils import RestoreCallback, install_output_interceptor


@AttackStageRegistry.register
class ResponseReplayerStage:
    """Replay the previous output batch produced by `response.submit` handlers."""

    stage_name = "response_replayer"
    required_capabilities = (Capability.RESPONSE_SUBMIT,)
    description = "Replays the previous response-submit output batch for the intercepted service."  # noqa: DC01

    def __init__(self) -> None:
        self._restore_callbacks: list[RestoreCallback] = []
        self._previous_outputs_by_service: dict[int, Any] = {}

    def execute(self, services: Sequence[BehaviorService[Any, Any]]) -> AttackStageResult:
        self.deactivate()

        if not services:
            return AttackStageResult(
                stage_name=self.stage_name,
                status=Status.FAIL,
                reason="ResponseReplayerStage received no target services.",
            )

        for service in services:
            restore_callback = install_output_interceptor(
                service,
                Capability.RESPONSE_SUBMIT,
                rewrite_result=partial(self._replay_output, service),
            )
            self._restore_callbacks.append(restore_callback)

        return AttackStageResult(
            stage_name=self.stage_name,
            status=Status.SUCCESS,
            reason=f"Installed response replayers on {len(services)} service(s).",
        )

    def deactivate(self) -> None:
        """Remove active interceptors and clear replay history."""
        while self._restore_callbacks:
            restore_callback = self._restore_callbacks.pop()
            restore_callback()

        self._previous_outputs_by_service.clear()

    def _replay_output(
        self,
        service: BehaviorService[Any, Any],
        output: Any,
    ) -> Any:
        service_key = id(service)
        previous_output = self._previous_outputs_by_service.get(service_key)
        self._previous_outputs_by_service[service_key] = deepcopy(output)

        if previous_output is None:
            return output

        return deepcopy(previous_output)
