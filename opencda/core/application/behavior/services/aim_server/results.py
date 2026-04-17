"""Result dataclasses for the AIM server behavior service."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla


@dataclass(frozen=True)
class AIMServerMessage:
    """Predicted next target position for a vehicle handled by AIM."""

    service_name: str
    vehicle_id: str
    next_position: carla.Location


@dataclass(frozen=True)
class AIMServerResult:
    """Batch prediction result returned by the AIM server service."""

    service_name: str
    owner_id: str
    messages: tuple[AIMServerMessage, ...]
