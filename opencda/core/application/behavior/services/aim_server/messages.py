"""Input dataclasses for the AIM server behavior service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from opencda.core.application.behavior.types import Location, Transform


@dataclass(frozen=True)
class AIMServerRequest:
    """CAV state and route context routed to the AIM server service."""

    vehicle_id: str
    position: Location
    speed: float
    yaw: float
    waypoints: Sequence[Transform]


@dataclass(frozen=True)
class AIMServerResponse:
    """Predicted next target position for a vehicle handled by AIM."""

    trajectory: Sequence[Location]
    yaw: float | None = None
    speed: float | None = None
