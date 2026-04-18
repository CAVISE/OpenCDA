"""Dummy behavior service implementation."""

from __future__ import annotations

import logging
from typing import Any, Sequence
import weakref

from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage

from .messages import DummyServiceMessage
from .results import DummyServiceEchoMessage, DummyServiceResult

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.dummy_service")


@BehaviorServiceRegistry.register
class DummyService:
    """A trivial service that echoes back text with a small modification."""

    service_name = "dummy_service"

    def __init__(
        self,
        response_suffix: str = " [dummy processed]",
    ) -> None:
        self._response_suffix = response_suffix
        self._owner_ref = None

    def on_attach(self, owner: Any) -> None:
        self._owner_ref = weakref.ref(owner)

    def process(self, messages: Sequence[TransportMessage[DummyServiceMessage]]) -> TransportMessage[DummyServiceResult]:
        if self._owner_ref() is None:
            raise RuntimeError("Cannot process messages without an attached owner.")
        echoed_messages = tuple(
            TransportMessage(
                src_owner_id=self._owner_ref().id,
                src_service_type=self.service_name,
                dst_owner_id=message.src_owner_id,
                dst_service_type=message.src_service_type,
                payload=DummyServiceEchoMessage(
                    text=f"{message.payload.text}{self._response_suffix}",
                ),
            )
            for message in messages
        )
        logger.info(
            "DummyService owner=%s service=%s processed_messages=%s",
            self._owner_ref().id,
            self.service_name,
            [message.text for message in echoed_messages],
        )
        payload = DummyServiceResult(
            messages=echoed_messages,
        )
        return TransportMessage(
            src_owner_id=self._owner_ref().id,
            src_service_type=self.service_name,
            dst_owner_id="broadcast",
            dst_service_type="",
            payload=payload,
        )

    def on_detach(self) -> None:
        self._owner_ref = None
