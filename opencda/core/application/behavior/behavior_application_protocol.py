"""Interface contracts for behavior applications."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .messages import BehaviorApplicationMessage
from .results import BehaviorApplicationResult


@runtime_checkable
class BehaviorApplication(Protocol):
    """Protocol implemented by any behavior application attached to a participant."""

    @property
    def application_id(self) -> str:
        """Return a stable identifier of the application instance."""

    def on_attach(self, owner_id: str) -> None:
        """Initialize the application for a particular participant instance."""

    def process(self, messages: Sequence[BehaviorApplicationMessage]) -> BehaviorApplicationResult:
        """Process typed input messages and return a typed result."""

    def on_detach(self) -> None:
        """Release application resources before the participant is destroyed."""
