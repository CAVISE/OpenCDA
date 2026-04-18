"""Input dataclasses for the AIM server behavior service."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import deque
    from .types import Transform, Location


@dataclass(frozen=True)
class AIMServerRequest:
    """CAV state and route context routed to the AIM server service."""

    vehicle_id: str
    position: Transform
    speed: float
    yaw: float
    waypoints: deque


@dataclass(frozen=True)
class AIMServerResponse:
    """Predicted next target position for a vehicle handled by AIM."""

    next_position: Location
