from dataclasses import dataclass

from opencda.core.application.behavior.types import Location


@dataclass(frozen=True, slots=True)
class MovementControllerState:
    service_type: str
    owner_id: str | None
    target_position: Location | None  # noqa: DC01
