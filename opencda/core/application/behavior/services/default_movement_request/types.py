from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DefaultMovementRequestState:
    service_type: str
    owner_id: str
