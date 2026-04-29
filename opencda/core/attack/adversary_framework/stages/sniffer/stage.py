"""Passive sniffer stage for observing service responses."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.core.application.behavior.capability import Capability
from opencda.core.attack.adversary_framework.models import AttackStageResult, Status
from opencda.core.attack.adversary_framework.stage_registry import AttackStageRegistry
from opencda.core.attack.adversary_framework.stages.sniffer.types import ObservedOutput
from opencda.core.attack.adversary_framework.utils import RestoreCallback, install_output_interceptor


@AttackStageRegistry.register
class SnifferStage:
    """Passively observe outputs produced by services exposing `response.observe`."""

    stage_name = "sniffer"
    required_capabilities = (Capability.RESPONSE_OBSERVE,)
    description = "Passively captures outputs returned by response-observe capability handlers."

    def __init__(self) -> None:
        self.observed_outputs: list[ObservedOutput] = []
        self._restore_callbacks: list[RestoreCallback] = []

    def execute(self, services: Sequence[BehaviorService[Any, Any]]) -> AttackStageResult:
        self.deactivate()

        if not services:
            return AttackStageResult(
                stage_name=self.stage_name,
                status=Status.FAIL,
                reason="SnifferStage received no target services.",
            )

        for service in services:
            restore_callback = install_output_interceptor(
                service,
                Capability.RESPONSE_OBSERVE,
                rewrite_result=partial(self._observe_output, service),
            )
            self._restore_callbacks.append(restore_callback)

        return AttackStageResult(
            stage_name=self.stage_name,
            status=Status.SUCCESS,
            reason=f"Installed response observers on {len(services)} service(s).",
        )

    def deactivate(self) -> None:
        """Remove all active observers installed by this stage."""
        while self._restore_callbacks:
            restore_callback = self._restore_callbacks.pop()
            restore_callback()

    def _observe_output(
        self,
        service: BehaviorService[Any, Any],
        output: Any,
    ) -> Any:
        self.observed_outputs.append(
            ObservedOutput(
                service=service,
                output=deepcopy(output),
            )
        )
        return output
