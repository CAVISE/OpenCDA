"""Replay previously observed service outputs."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.core.application.behavior.capability import Capability
from opencda.core.attack.adversary_framework.models import AttackStageResult, Status
from opencda.core.attack.adversary_framework.stage_registry import AttackStageRegistry
from opencda.core.attack.adversary_framework.utils import RestoreCallback, install_output_interceptor, safe_clone


@AttackStageRegistry.register
class ReplayerStage:
    """Replay the previous outputs produced by the configured capability handlers."""

    stage_name = "replayer"
    supported_capabilities = (
        Capability.REQUEST_SUBMIT,
        Capability.RESPONSE_SUBMIT,
        Capability.COMMAND_SUBMIT,
    )
    default_capabilities = (Capability.RESPONSE_SUBMIT,)
    description = "Replays the previous output observed on the configured capability handlers."  # noqa: DC01

    def __init__(self, capabilities: Sequence[Capability]) -> None:
        self.capabilities = tuple(capabilities)
        self._restore_callbacks: list[RestoreCallback] = []
        self._previous_outputs_by_binding: dict[tuple[int, Capability], Any] = {}

    def execute(self, services: Sequence[BehaviorService[Any, Any]]) -> AttackStageResult:
        self.deactivate()

        if not services:
            return AttackStageResult(
                stage_name=self.stage_name,
                status=Status.FAIL,
                reason="ReplayerStage received no target services.",
            )

        for service in services:
            for capability in self.capabilities:
                restore_callback = install_output_interceptor(
                    service,
                    capability,
                    rewrite_result=partial(self._replay_output, service, capability),
                )
                self._restore_callbacks.append(restore_callback)

        return AttackStageResult(
            stage_name=self.stage_name,
            status=Status.SUCCESS,
            reason=f"Installed replayers for {len(self.capabilities)} capability handler(s) on {len(services)} service(s).",
        )

    def deactivate(self) -> None:
        """Remove active interceptors and clear replay history."""
        while self._restore_callbacks:
            restore_callback = self._restore_callbacks.pop()
            restore_callback()

        self._previous_outputs_by_binding.clear()

    def _replay_output(
        self,
        service: BehaviorService[Any, Any],
        capability: Capability,
        output: Any,
    ) -> Any:
        binding_key = (id(service), capability)
        previous_output = self._previous_outputs_by_binding.get(binding_key)
        self._previous_outputs_by_binding[binding_key] = safe_clone(output)

        if previous_output is None:
            return output

        return safe_clone(previous_output)
