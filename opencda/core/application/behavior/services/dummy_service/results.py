"""Result dataclasses for the dummy behavior service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DummyServiceEchoMessage:
    """Echoed text produced by the dummy service."""

    text: str


@dataclass(frozen=True)
class DummyServiceResult:
    """Batch result returned by the dummy service."""

    messages: tuple[DummyServiceEchoMessage, ...]
