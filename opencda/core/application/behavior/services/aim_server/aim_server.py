"""Dummy behavior service implementation."""

from __future__ import annotations

import logging
from typing import Sequence

from opencda.core.application.behavior.registry import BehaviorServiceRegistry

from .results import AIMServerMessage

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_server")


@BehaviorServiceRegistry.register
class AIMServer:
    """A trivial service that echoes back text with a small modification."""

    service_name = "aim_server"

    def __init__(
        self,
        service_id: str = "aim_server",
    ) -> None:
        self._service_id = service_id
        self._owner_id = ""

    @property
    def service_id(self) -> str:
        return self._service_id

    def on_attach(self, owner_id: str) -> None:
        self._owner_id = owner_id

    def process(self, messages: Sequence[AIMServerMessage]) -> None:
        pass

    def on_detach(self) -> None:
        self._owner_id = ""
