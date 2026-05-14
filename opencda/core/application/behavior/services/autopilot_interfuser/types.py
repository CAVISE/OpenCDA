from __future__ import annotations

from dataclasses import dataclass

from opencda.core.application.behavior.types import Location


@dataclass(frozen=True, slots=True)
class AutopilotInterfuserState:
    service_type: str
    owner_id: str | None
    model_loaded: bool
    step: int
    last_target_location: Location | None
    last_target_speed: float | None
    last_error: str | None
