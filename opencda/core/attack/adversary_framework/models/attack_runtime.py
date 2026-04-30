"""Runtime state primitives for configured attacks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from opencda.core.attack.adversary_framework.models.attack_result import AttackStageResult
from opencda.core.attack.adversary_framework.models.attack_spec import StageSpec

if TYPE_CHECKING:
    from opencda.core.attack.adversary_framework.attack_stage_protocol import AttackStage


class RuntimeStatus(str, Enum):
    """Runtime lifecycle status for attacks and stages."""

    STARTED = "STARTED"
    INACTIVE = "inactive"
    ACTIVE = "active"
    SUCCESS = "success"
    FAIL = "fail"
    STOPPED = "stopped"


@dataclass(slots=True)
class StageRuntime:
    """Runtime state of a configured attack stage."""

    spec: StageSpec
    stage: AttackStage
    status: RuntimeStatus = RuntimeStatus.INACTIVE
    previous_status: RuntimeStatus = RuntimeStatus.INACTIVE
    last_result: AttackStageResult | None = None
