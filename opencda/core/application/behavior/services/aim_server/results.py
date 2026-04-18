"""Result dataclasses for the AIM server behavior service."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla
    from .messages import Location


@dataclass(frozen=True)
class AIMServerMessage:
    """Predicted next target position for a vehicle handled by AIM."""

    next_position: Location | carla.Location


@dataclass(frozen=True)
class AIMServerResult:
    """Batch prediction result returned by the AIM server service."""

    messages: tuple[AIMServerMessage, ...]
