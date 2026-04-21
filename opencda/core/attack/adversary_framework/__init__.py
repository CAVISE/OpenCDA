"""Public API for adversary framework primitives."""

from .attack_manager import AttackManager
from .attack_protocol import Attack
from .attack_result import AttackResult, AttackStageResult, Status
from .attack_stage_protocol import AttackStage
from .registry import AttackRegistry
from .utils import (
    AttackResultRewriter,
    RestoreCallback,
    get_capability_binding,
    install_output_interceptor,
    match_services,
    service_supports_capabilities,
    wrap_method_output,
)

__all__ = [
    "Attack",
    "AttackManager",
    "AttackRegistry",
    "AttackResult",
    "AttackResultRewriter",
    "AttackStage",
    "AttackStageResult",
    "Status",
    "RestoreCallback",
    "get_capability_binding",
    "install_output_interceptor",
    "match_services",
    "service_supports_capabilities",
    "wrap_method_output",
]
