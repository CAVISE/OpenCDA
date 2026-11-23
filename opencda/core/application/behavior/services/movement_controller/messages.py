"""Movement-command payload contract shared by command-producing services."""

from __future__ import annotations
from dataclasses import dataclass

from opencda.core.application.behavior.types import Location


@dataclass(frozen=True)
class MovementControllerRequestMessage:
    """CAV state and route context routed to the MovementController service."""

    target_speed: float | None
    target_location: Location | None
