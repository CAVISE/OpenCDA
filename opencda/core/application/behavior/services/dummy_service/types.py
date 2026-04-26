from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DummyServiceState:
    service_name: str
    owner_id: str | None
    is_attached: bool
