"""Input dataclasses for the AIM server behavior service."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla
    from collections import deque


@dataclass(frozen=True)
class AIMServerRequestMessage:
    """Simple text message routed to the AIM server service."""

    service_id: str
    text: str
    vehicle_id: str
    position: carla.Transform
    speed: float
    yaw: float
    waypoints: deque
