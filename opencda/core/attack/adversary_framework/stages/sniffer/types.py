from __future__ import annotations

from typing import Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService


class ObservedOutput:
    """Observed output captured from a service capability call."""

    __slots__ = ("service", "output")

    def __init__(
        self,
        service: BehaviorService[Any, Any],
        output: Any,
    ) -> None:
        self.service = service
        self.output = output
