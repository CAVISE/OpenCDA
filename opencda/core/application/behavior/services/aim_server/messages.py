"""Input dataclasses for the AIM server behavior service."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla
    from collections import deque


@dataclass(frozen=True)
class Location:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Rotation:
    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class Transform:
    location: Location
    rotation: Rotation


@dataclass(frozen=True)
class AIMServerRequestMessage:
    """CAV state and route context routed to the AIM server service."""

    vehicle_id: str
    position: Transform | carla.Transform
    speed: float
    yaw: float
    waypoints: deque
