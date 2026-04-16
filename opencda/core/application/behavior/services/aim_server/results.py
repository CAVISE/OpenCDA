"""Result dataclasses for the dummy behavior service."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla


@dataclass(frozen=True)
class AIMServerMessage:
    """Echoed text produced by the dummy service."""

    service_id: str
    vehicle_id: str
    next_position: carla.Location


@dataclass(frozen=True)
class AIMServerResult:
    """Batch result returned by the dummy service."""

    service_id: str
    owner_id: str
    messages: tuple[AIMServerMessage, ...]
