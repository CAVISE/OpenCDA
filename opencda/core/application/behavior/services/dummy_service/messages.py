"""Input dataclasses for the dummy behavior service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DummyServiceMessage:
    """Simple text message routed to the dummy service."""

    text: str
