"""Shared localization data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from opencda.core.application.behavior.types import Transform


class LocalizationSource(str, Enum):
    """Source used to produce a localization estimate."""

    GT = "gt"
    SENSOR = "sensor"


@dataclass(frozen=True, slots=True)
class LocalizationState:
    """Immutable localization snapshot exposed to OpenCDA consumers."""

    transform: Transform
    speed_kmh: float
    source: LocalizationSource
    frame: int | None = None
    timestamp: float | None = None
