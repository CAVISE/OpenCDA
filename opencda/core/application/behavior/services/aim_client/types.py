from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AIMClientState:
    service_name: str
    owner_id: str | None
    is_attached: bool  # noqa: DC01
