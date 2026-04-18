"""Interface contracts for behavior services."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable
from .transport_message import TransportMessage

BehaviorServiceMessageT = TypeVar("BehaviorServiceMessageT", contravariant=True)
BehaviorServiceResultT = TypeVar("BehaviorServiceResultT", covariant=True)


@runtime_checkable
class BehaviorService(Protocol[BehaviorServiceMessageT, BehaviorServiceResultT]):
    """Protocol implemented by any behavior service attached to a participant."""

    def on_attach(self, owner: Any) -> None:
        """Initialize the service for a particular participant instance."""

    def process(self, messages: Sequence[TransportMessage[BehaviorServiceMessageT]]) -> TransportMessage[BehaviorServiceResultT]:
        """Process typed input messages and return a typed result."""

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
