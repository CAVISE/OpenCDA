"""Results for attack stage execution and attack completion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    """Execution status emitted by an attack stage or the whole attack pipeline."""

    SUCCESS = "success"
    FAIL = "fail"
    STOP = "stop"
    UNSTABLE = "unstable"
    ERROR = "error"


@dataclass(slots=True)
class AttackStageResult:
    """Result of a single attack stage execution."""

    stage_name: str
    status: Status
    reason: str = ""


@dataclass(slots=True)
class AttackResult:
    """Final attack execution outcome."""

    attack_name: str
    status: Status
    reason: str = ""
    stage_history: tuple[AttackStageResult, ...] = field(default_factory=tuple)
