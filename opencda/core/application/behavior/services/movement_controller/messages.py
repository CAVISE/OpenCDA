"""Input dataclasses for the AIM server behavior service."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opencda.core.application.behavior.types import Transform


@dataclass(frozen=True)
class MovementControllerRequestMessage:
    """CAV state and route context routed to the MovementController service."""

    target_position: Transform
