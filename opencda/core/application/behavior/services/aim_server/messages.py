"""Input dataclasses for the AIM server behavior service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from opencda.core.application.behavior.types import Location, Transform


@dataclass(frozen=True)
class AIMServerRequest:
    """CAV state and route context routed to the AIM server service."""

    vehicle_id: str
    position: Transform
    speed: float
    yaw: float
    waypoints: Sequence[Any]


@dataclass(frozen=True)
class AIMServerResponse:
    """Predicted next target position for a vehicle handled by AIM."""

    trajectory: Sequence[Location]
