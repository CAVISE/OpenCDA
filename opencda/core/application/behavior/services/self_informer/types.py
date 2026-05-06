from dataclasses import dataclass

from opencda.core.application.behavior.types import Location


@dataclass(frozen=True, slots=True)
class SelfInformerState:
    service_type: str
    owner_id: str | None
    location: Location | None
    speed: float | None
