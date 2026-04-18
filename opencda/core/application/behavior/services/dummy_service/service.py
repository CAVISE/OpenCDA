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
        self._owner_ref: weakref.ReferenceType[Any] | None = None

    def _get_owner(self) -> Any:
        owner_ref = self._owner_ref
        if owner_ref is None:
            raise RuntimeError("DummyService is not attached to an owner.")

        owner = owner_ref()
        if owner is None:
            raise RuntimeError("DummyService owner is no longer available.")

        return owner

    def on_attach(self, owner: Any) -> None:
        self._owner_ref = weakref.ref(owner)

    def process(self, messages: Sequence[TransportMessage[DummyServiceMessage]]) -> Sequence[TransportMessage[DummyServiceResult]]:
        owner = self._get_owner()
        echoed_payloads = tuple(
            DummyServiceEchoMessage(
                text=f"{message.payload.text}{self._response_suffix}",
            )
            for message in messages
        )
        logger.info(
            "DummyService owner=%s service=%s processed_messages=%s",
            owner.id,
            self.service_name,
            [message.text for message in echoed_payloads],
        )
        payload = DummyServiceResult(messages=echoed_payloads)
        result_message = TransportMessage(
            src_owner_id=owner.id,
            src_service_type=self.service_name,
            dst_owner_id="broadcast",
            dst_service_type="",
            payload=payload,
        )
        return (result_message,)

    def on_detach(self) -> None:
        self._owner_ref = None
