"""Default movement-request behavior service implementation."""

from __future__ import annotations

import logging
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from opencda.core.application.behavior.capability import CapabilityBindings
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from opencda.core.application.behavior.services.movement_controller import MovementControllerRequestMessage
from .types import DefaultMovementRequestState

if TYPE_CHECKING:
    from opencda.core.common.agent_manager import AgentManager


logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.default_movement_request")


@BehaviorServiceRegistry.register
class DefaultMovementRequest:
    """Behavior service that emits a no-op movement request as a fallback control input."""

    service_type: str = "default_movement_request"
    priority: int = 3

    @property
    def capability_bindings(self) -> CapabilityBindings:
        return {}

    def __init__(self, priority: int = 3) -> None:
        self.priority = priority
        self._owner_ref: weakref.ReferenceType[AgentManager] | None = None

    def _get_owner(self) -> AgentManager:
        owner_ref = self._owner_ref
        if owner_ref is None:
            raise RuntimeError("DefaultMovementRequest is not attached to an owner.")

        owner = owner_ref()
        if owner is None:
            raise RuntimeError("DefaultMovementRequest owner is no longer available.")

        return owner

    def on_attach(self, owner: AgentManager) -> None:
        self._owner_ref = weakref.ref(owner)

    def on_detach(self) -> None:
        self._owner_ref = None

    def get_state(self) -> DefaultMovementRequestState:
        owner = self._get_owner()
        return DefaultMovementRequestState(
            service_type=self.service_type,
            owner_id=owner.id,
        )

    def _build_default_movement_request(self) -> TransportMessage[MovementControllerRequestMessage]:
        owner = self._get_owner()
        payload = MovementControllerRequestMessage(target_speed=None, target_location=None)
        return TransportMessage(
            src_owner_id=owner.id,
            src_service_type=self.service_type,
            dst_owner_id=owner.id,
            dst_service_type="movement_controller",
            payload=payload,
        )

    def process(
        self,
        messages: Sequence[TransportMessage[Any]],
    ) -> tuple[TransportMessage[MovementControllerRequestMessage], ...]:
        return (self._build_default_movement_request(),)
