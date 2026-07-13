from dataclasses import dataclass

from opencda.core.sensing.localization.types import LocalizationState


@dataclass(frozen=True, slots=True)
class SelfInformerState:
    service_type: str
    owner_id: str | None
    localization: LocalizationState | None
