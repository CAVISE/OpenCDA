from dataclasses import dataclass

from opencda.core.application.behavior.types import Location


@dataclass(frozen=True, slots=True)
class MovementControllerState:
    service_name: str
    owner_id: str | None
    is_attached: bool
    target_position: Location | None
