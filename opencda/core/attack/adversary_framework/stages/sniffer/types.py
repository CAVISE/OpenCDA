from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.core.application.behavior.capability import Capability


@dataclass(slots=True)
class ObservedOutput:
    """Observed output captured from a service capability call."""

    service: BehaviorService[Any, Any]
    capability: Capability
    output: Any
