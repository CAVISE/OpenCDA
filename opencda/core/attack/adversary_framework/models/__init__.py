"""Dataclass-based models for adversary framework configuration and runtime."""

from .attack_context import AttackContext
from .attack_result import AttackResult, AttackStageResult, Status
from .attack_runtime import RuntimeStatus, StageRuntime
from .attack_spec import AttackSpec, ConditionSpec, StageSpec, TargetSpec, TriggerSourceSpec

__all__ = [
    "AttackContext",
    "AttackResult",
    "AttackSpec",
    "AttackStageResult",
    "ConditionSpec",
    "RuntimeStatus",
    "StageRuntime",
    "StageSpec",
    "Status",
    "TargetSpec",
    "TriggerSourceSpec",
]
