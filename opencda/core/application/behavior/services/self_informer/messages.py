"""Message dataclasses for the self-informer behavior service."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opencda.core.application.behavior.types import Location


@dataclass(frozen=True)
class SelfInformerResponse:
    """Current vehicle state published by the self-informer service."""

    location: Location | None
    speed: float | None
