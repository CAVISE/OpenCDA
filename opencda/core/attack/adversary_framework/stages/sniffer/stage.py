"""Passive sniffer stage for observing service responses."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.core.application.behavior.capability import Capability
from opencda.core.attack.adversary_framework.models import AttackStageResult, Status
from opencda.core.attack.adversary_framework.stage_registry import AttackStageRegistry
from opencda.core.attack.adversary_framework.stages.sniffer.types import ObservedOutput
from opencda.core.attack.adversary_framework.utils import RestoreCallback, install_output_interceptor, safe_clone


@AttackStageRegistry.register
class SnifferStage:
    """Passively observe outputs produced by selected service capabilities."""

    stage_name = "sniffer"
    supported_capabilities = (
        Capability.REQUEST_OBSERVE,
        Capability.REQUEST_SUBMIT,
        Capability.RESPONSE_OBSERVE,
        Capability.RESPONSE_SUBMIT,
        Capability.COMMAND_SUBMIT,
        Capability.STATE_OBSERVE,
    )
    default_capabilities = (Capability.RESPONSE_OBSERVE,)
    description = "Passively captures outputs returned by the configured capability handlers."  # noqa: DC01

    def __init__(self, capabilities: Sequence[Capability]) -> None:
        self.capabilities = tuple(capabilities)
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
            for capability in self.capabilities:
                restore_callback = install_output_interceptor(
                    service,
                    capability,
                    rewrite_result=partial(self._observe_output, service, capability),
                )
                self._restore_callbacks.append(restore_callback)

        return AttackStageResult(
            stage_name=self.stage_name,
            status=Status.SUCCESS,
            reason=f"Installed observers for {len(self.capabilities)} capability handler(s) on {len(services)} service(s).",
        )

    def deactivate(self) -> None:
        """Remove all active observers installed by this stage."""
        while self._restore_callbacks:
            restore_callback = self._restore_callbacks.pop()
            restore_callback()

    def _observe_output(
        self,
        service: BehaviorService[Any, Any],
        capability: Capability,
        output: Any,
    ) -> Any:
        self.observed_outputs.append(
            ObservedOutput(
                service=service,
                capability=capability,
                output=safe_clone(output),
            )
        )
        return output
