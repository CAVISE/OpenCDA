"""Public API for adversary framework primitives."""

from .stages import _import_builtin_attack_stages
from .attack_manager import AttackManager
from .attack import Attack
from .attack_stage_protocol import AttackStage
from .models import AttackContext, AttackResult, AttackSpec, AttackStageResult, ConditionSpec, RuntimeStatus, StageRuntime, StageSpec, Status, TargetSpec, TriggerSourceSpec
from .stage_registry import AttackStageRegistry
from .utils import (
    AttackResultRewriter,
    RestoreCallback,
    ServiceResolver,
    get_capability_binding,
    install_output_interceptor,
    match_services,
    resolve_targets,
    service_supports_capabilities,
    wrap_method_output,
)

__all__ = [
    "AttackContext",
    "Attack",
    "AttackManager",
    "AttackSpec",
    "ConditionSpec",
    "ServiceResolver",
    "AttackStageRegistry",
    "AttackResult",
    "AttackResultRewriter",
    "AttackStage",
    "AttackStageResult",
    "RuntimeStatus",
    "StageRuntime",
    "StageSpec",
    "Status",
    "TargetSpec",
    "TriggerSourceSpec",
    "RestoreCallback",
    "get_capability_binding",
    "install_output_interceptor",
    "match_services",
    "resolve_targets",
    "service_supports_capabilities",
    "wrap_method_output",
]

_import_builtin_attack_stages()
