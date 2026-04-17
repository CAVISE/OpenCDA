"""Result dataclasses for the dummy behavior service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DummyServiceEchoMessage:
    """Echoed text produced by the dummy service."""

    service_name: str
    text: str


@dataclass(frozen=True)
class DummyServiceResult:
    """Batch result returned by the dummy service."""

    service_name: str
    owner_id: str
    messages: tuple[DummyServiceEchoMessage, ...]
