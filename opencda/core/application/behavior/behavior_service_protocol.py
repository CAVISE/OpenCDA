"""Interface contracts for behavior services."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

from .transport_message import TransportMessage

BehaviorServiceRequestT = TypeVar("BehaviorServiceRequestT")
BehaviorServiceResponseT = TypeVar("BehaviorServiceResponseT")


@runtime_checkable
class BehaviorService(Protocol[BehaviorServiceRequestT, BehaviorServiceResponseT]):
    """Protocol implemented by any behavior service attached to a participant."""

    service_name: str

    def on_attach(self, owner: Any) -> None:
        """Initialize the service for a particular participant instance."""

    def process(
        self,
        messages: Sequence[TransportMessage[BehaviorServiceRequestT]],
    ) -> Sequence[TransportMessage[BehaviorServiceResponseT]]:
        """Process typed input messages and return a typed result."""

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
