from __future__ import annotations

from dataclasses import dataclass

from opencda.core.application.behavior.types import Location


@dataclass(frozen=True, slots=True)
class AIMClientState:
    service_type: str
    owner_id: str | None
    trajectory: tuple[Location, ...]  # noqa: DC01
