"""Interface contracts for behavior services."""

from __future__ import annotations

from typing import Protocol, Sequence, TypeVar, runtime_checkable

BehaviorServiceMessageT = TypeVar("BehaviorServiceMessageT", contravariant=True)
BehaviorServiceResultT = TypeVar("BehaviorServiceResultT", covariant=True)


@runtime_checkable
class BehaviorService(Protocol[BehaviorServiceMessageT, BehaviorServiceResultT]):
    """Protocol implemented by any behavior service attached to a participant."""

    @property
    def service_id(self) -> str:
        """Return a stable identifier of the service instance."""

    def on_attach(self, owner_id: str) -> None:
        """Initialize the service for a particular participant instance."""

    def process(self, messages: Sequence[BehaviorServiceMessageT]) -> BehaviorServiceResultT:
        """Process typed input messages and return a typed result."""

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
