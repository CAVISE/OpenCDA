"""Drop service outputs produced by configured capability handlers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import random
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
    _default_drop_rate = 1.0

    def __init__(self, capabilities: Sequence[Capability], params: Mapping[str, Any] | None = None) -> None:
        self.capabilities = tuple(capabilities)
        self._restore_callbacks: list[RestoreCallback] = []
        self._drop_rate = self._parse_drop_rate(params)

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
            reason=(
                f"Installed droppers for {len(self.capabilities)} capability handler(s) on "
                f"{len(services)} service(s) with drop_rate={self._drop_rate}."
            ),
        )

    def deactivate(self) -> None:
        """Remove all active interceptors installed by this stage."""
        while self._restore_callbacks:
            restore_callback = self._restore_callbacks.pop()
            restore_callback()

    def _drop_output(self, output: Any) -> Any:
        is_batch_output = self._is_batch_output(output)
        if self._drop_rate <= 0.0:
            if is_batch_output:
                return tuple(output)
            return output
        if self._drop_rate >= 1.0:
            return ()
        if is_batch_output:
            return tuple(item for item in output if not self._should_drop())
        if self._should_drop():
            return ()
        return output

    @staticmethod
    def _is_batch_output(output: Any) -> bool:
        return isinstance(output, Iterable) and not isinstance(output, (str, bytes, bytearray, Mapping))

    def _should_drop(self) -> bool:
        return random.random() < self._drop_rate

    @classmethod
    def _parse_drop_rate(cls, params: Mapping[str, Any] | None) -> float:
        if params is None:
            return cls._default_drop_rate

        raw_drop_rate = params.get("drop_rate", cls._default_drop_rate)
        if not isinstance(raw_drop_rate, float):
            raise ValueError("DropperStage 'params.drop_rate' must be a float from 0.0 to 1.0.")

        drop_rate = float(raw_drop_rate)
        if not 0.0 <= drop_rate <= 1.0:
            raise ValueError("DropperStage 'params.drop_rate' must be between 0.0 and 1.0.")
        return drop_rate
