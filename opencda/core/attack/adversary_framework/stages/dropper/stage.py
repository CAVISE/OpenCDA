"""Drop service outputs produced by configured capability handlers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.core.application.behavior.capability import Capability
from opencda.core.attack.adversary_framework.models import AttackStageResult, Status
from opencda.core.attack.adversary_framework.stage_registry import AttackStageRegistry
from opencda.core.attack.adversary_framework.utils import RestoreCallback, install_output_interceptor


@AttackStageRegistry.register
class DropperStage:
    """Replace selected message-producing outputs with an empty batch."""

    stage_name = "dropper"
    supported_capabilities = (
        Capability.REQUEST_OBSERVE,
        Capability.REQUEST_SUBMIT,
        Capability.RESPONSE_OBSERVE,
        Capability.RESPONSE_SUBMIT,
        Capability.COMMAND_SUBMIT,
    )
    default_capabilities = (Capability.RESPONSE_SUBMIT,)
    description = "Drops outputs returned by the configured message capability handlers."  # noqa: DC01

    def __init__(self, capabilities: Sequence[Capability]) -> None:
        self.capabilities = tuple(capabilities)
        self._restore_callbacks: list[RestoreCallback] = []

    def execute(self, services: Sequence[BehaviorService[Any, Any]]) -> AttackStageResult:
        self.deactivate()

        if not services:
            return AttackStageResult(
                stage_name=self.stage_name,
                status=Status.FAIL,
                reason="DropperStage received no target services.",
            )

        for service in services:
            for capability in self.capabilities:
                restore_callback = install_output_interceptor(
                    service,
                    capability,
                    rewrite_result=self._drop_output,
                )
                self._restore_callbacks.append(restore_callback)

        return AttackStageResult(
            stage_name=self.stage_name,
            status=Status.SUCCESS,
            reason=f"Installed droppers for {len(self.capabilities)} capability handler(s) on {len(services)} service(s).",
        )

    def deactivate(self) -> None:
        """Remove all active interceptors installed by this stage."""
        while self._restore_callbacks:
            restore_callback = self._restore_callbacks.pop()
            restore_callback()

    def _drop_output(self, output: Any) -> tuple[()]:
        del output
        return ()
