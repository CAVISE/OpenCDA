"""Spoof selected fields inside TransportMessage outputs."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from copy import deepcopy
from dataclasses import dataclass, is_dataclass
from typing import Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.core.application.behavior.capability import Capability
from opencda.core.application.behavior.transport_message import TransportMessage
from opencda.core.attack.adversary_framework.models import AttackStageResult, Status
from opencda.core.attack.adversary_framework.stage_registry import AttackStageRegistry
from opencda.core.attack.adversary_framework.utils import RestoreCallback, install_output_interceptor


@dataclass(frozen=True, slots=True)
class RewriteRule:
    """Single message rewrite rule configured through stage params."""

    path: tuple[str, ...]
    operation: str
    value: Any


@AttackStageRegistry.register
class SpooferStage:
    """Rewrite configured fields on TransportMessage envelopes and payloads."""

    stage_name = "spoofer"
    supported_capabilities = (
        Capability.REQUEST_SUBMIT,
        Capability.RESPONSE_SUBMIT,
        Capability.COMMAND_SUBMIT,
    )
    default_capabilities = (Capability.REQUEST_SUBMIT,)
    description = "Spoofs TransportMessage envelope or payload fields according to configured rewrite rules."  # noqa: DC01
    _supported_operations = frozenset({"set", "add", "multiply"})

    def __init__(self, capabilities: Sequence[Capability], params: Mapping[str, Any] | None = None) -> None:
        self.capabilities = tuple(capabilities)
        self._restore_callbacks: list[RestoreCallback] = []
        self._rewrite_rules = self._parse_rewrite_rules(params)

    def execute(self, services: Sequence[BehaviorService[Any, Any]]) -> AttackStageResult:
        self.deactivate()

        if not services:
            return AttackStageResult(
                stage_name=self.stage_name,
                status=Status.FAIL,
                reason="SpooferStage received no target services.",
            )

        for service in services:
            for capability in self.capabilities:
                restore_callback = install_output_interceptor(
                    service,
                    capability,
                    rewrite_result=self._spoof_output,
                )
                self._restore_callbacks.append(restore_callback)

        return AttackStageResult(
            stage_name=self.stage_name,
            status=Status.SUCCESS,
            reason=f"Installed spoofers for {len(self.capabilities)} capability handler(s) on {len(services)} service(s).",
        )

    def deactivate(self) -> None:
        """Remove all active interceptors installed by this stage."""
        while self._restore_callbacks:
            restore_callback = self._restore_callbacks.pop()
            restore_callback()

    def _spoof_output(self, output: Any) -> Any:
        if isinstance(output, TransportMessage):
            return self._spoof_message(output)
        if isinstance(output, tuple):
            return tuple(self._spoof_output(item) for item in output)
        if isinstance(output, list):
            return [self._spoof_output(item) for item in output]
        return output

    def _spoof_message(self, message: TransportMessage[Any]) -> TransportMessage[Any]:
        rewritten_message: TransportMessage[Any] = message
        has_changes = False

        for rewrite_rule in self._rewrite_rules:
            if not self._path_exists(rewritten_message, rewrite_rule.path):
                continue

            if not has_changes:
                rewritten_message = deepcopy(message)
                has_changes = True

            current_value = self._resolve_path_value(rewritten_message, rewrite_rule.path)
            spoofed_value = self._apply_operation(current_value, rewrite_rule)
            self._assign_path_value(rewritten_message, rewrite_rule.path, spoofed_value)

        return rewritten_message

    @classmethod
    def _parse_rewrite_rules(cls, params: Mapping[str, Any] | None) -> tuple[RewriteRule, ...]:
        if params is None:
            raise ValueError("SpooferStage requires 'params.rewrites' configuration.")

        raw_rewrites = params.get("rewrites")
        if not isinstance(raw_rewrites, Sequence) or isinstance(raw_rewrites, (str, bytes, bytearray)):
            raise ValueError("SpooferStage requires 'params.rewrites' to be a sequence of mappings.")
        if not raw_rewrites:
            raise ValueError("SpooferStage requires at least one rewrite rule in 'params.rewrites'.")

        rewrite_rules: list[RewriteRule] = []
        for index, raw_rewrite in enumerate(raw_rewrites):
            if not isinstance(raw_rewrite, Mapping):
                raise ValueError(f"SpooferStage rewrite #{index} must be a mapping.")

            raw_path = raw_rewrite.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"SpooferStage rewrite #{index} must define a non-empty string 'path'.")

            path = tuple(raw_path.split("."))
            if any(not segment for segment in path):
                raise ValueError(f"SpooferStage rewrite #{index} has invalid path '{raw_path}'.")

            operation = str(raw_rewrite.get("operation", "set"))
            if operation not in cls._supported_operations:
                supported = ", ".join(sorted(cls._supported_operations))
                raise ValueError(f"SpooferStage rewrite #{index} has unsupported operation '{operation}'. Supported operations: [{supported}].")

            if "value" not in raw_rewrite:
                raise ValueError(f"SpooferStage rewrite #{index} must define 'value'.")

            rewrite_rules.append(
                RewriteRule(
                    path=path,
                    operation=operation,
                    value=raw_rewrite["value"],
                )
            )

        return tuple(rewrite_rules)

    @staticmethod
    def _path_exists(root: Any, path: tuple[str, ...]) -> bool:
        try:
            SpooferStage._resolve_path_value(root, path)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _resolve_path_value(root: Any, path: tuple[str, ...]) -> Any:
        current = root
        for segment in path:
            current = SpooferStage._read_child(current, segment)
        return current

    @staticmethod
    def _assign_path_value(root: Any, path: tuple[str, ...], value: Any) -> None:
        parent = root
        for segment in path[:-1]:
            parent = SpooferStage._read_child(parent, segment)
        SpooferStage._write_child(parent, path[-1], value)

    @staticmethod
    def _read_child(parent: Any, segment: str) -> Any:
        if isinstance(parent, Mapping):
            return parent[segment]
        if SpooferStage._is_sequence_index(segment) and isinstance(parent, Sequence) and not isinstance(parent, (str, bytes, bytearray)):
            return parent[int(segment)]
        return getattr(parent, segment)

    @staticmethod
    def _write_child(parent: Any, segment: str, value: Any) -> None:
        if isinstance(parent, MutableMapping):
            parent[segment] = value
            return
        if SpooferStage._is_sequence_index(segment):
            if isinstance(parent, MutableSequence):
                parent[int(segment)] = value
                return
            raise TypeError("SpooferStage does not support indexed writes to immutable sequences.")
        if is_dataclass(parent):
            object.__setattr__(parent, segment, value)
            return
        setattr(parent, segment, value)

    @staticmethod
    def _is_sequence_index(segment: str) -> bool:
        return segment.isdigit()

    @staticmethod
    def _apply_operation(current_value: Any, rewrite_rule: RewriteRule) -> Any:
        if rewrite_rule.operation == "set":
            return deepcopy(rewrite_rule.value)
        if rewrite_rule.operation == "add":
            return current_value + rewrite_rule.value
        if rewrite_rule.operation == "multiply":
            return current_value * rewrite_rule.value
        raise ValueError(f"Unsupported rewrite operation '{rewrite_rule.operation}'.")
