from dataclasses import dataclass

from opencda.core.application.behavior.types import Location


@dataclass(frozen=True, slots=True)
class AIMClientState:
    service_name: str
    owner_id: str | None
    is_attached: bool  # noqa: DC01
    next_position: Location | None
