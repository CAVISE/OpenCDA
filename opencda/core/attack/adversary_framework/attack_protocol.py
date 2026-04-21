"""Protocol for adversary attacks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .attack_stage_protocol import AttackStage


@runtime_checkable
class Attack(Protocol):
    """Declarative attack contract."""

    attack_name: str
    stages: Sequence[AttackStage]
