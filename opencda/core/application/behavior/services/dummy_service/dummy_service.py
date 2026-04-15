"""Dummy behavior service implementation."""

from __future__ import annotations

from typing import Sequence

from opencda.core.application.behavior.registry import BehaviorServiceRegistry

from .messages import DummyServiceMessage
from .results import DummyServiceEchoMessage, DummyServiceResult


@BehaviorServiceRegistry.register
class DummyService:
    """A trivial service that echoes back text with a small modification."""

    service_name = "dummy_service"

    def __init__(self, service_id: str = "dummy_service", response_suffix: str = " [dummy processed]") -> None:
        self._service_id = service_id
        self._response_suffix = response_suffix
        self._owner_id = ""

    @property
    def service_id(self) -> str:
        return self._service_id

    def on_attach(self, owner_id: str) -> None:
        self._owner_id = owner_id

    def process(self, messages: Sequence[DummyServiceMessage]) -> DummyServiceResult:
        echoed_messages = tuple(
            DummyServiceEchoMessage(
                service_id=self.service_id,
                text=f"{message.text}{self._response_suffix}",
            )
            for message in messages
        )
        return DummyServiceResult(
            service_id=self.service_id,
            owner_id=self._owner_id,
            messages=echoed_messages,
        )

    def on_detach(self) -> None:
        self._owner_id = ""
