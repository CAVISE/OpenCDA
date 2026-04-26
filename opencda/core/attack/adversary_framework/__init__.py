"""Public API for adversary framework primitives."""

from . import attacks as _builtin_attacks
from . import stages as _builtin_stages
from .attack_manager import AttackManager
from .attack_protocol import Attack, ServiceResolver
from .attack_result import AttackResult, AttackStageResult, Status
from .attack_stage_protocol import AttackStage
from .registry import AttackRegistry
from .stage_registry import AttackStageRegistry
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
    "ServiceResolver",
    "AttackStageRegistry",
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
