"""AIM server behavior service implementation."""

from __future__ import annotations

import weakref
import logging
from typing import Sequence, TYPE_CHECKING

from opencda.core.application.behavior.capability import CapabilityBindings
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from opencda.core.application.behavior.types import Location
from .messages import MovementControllerRequestMessage
from .types import MovementControllerState


if TYPE_CHECKING:
    from opencda.core.common.vehicle_manager import VehicleManager


logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_client")


@BehaviorServiceRegistry.register
class MovementController:
    """Behavior service that runs AIM predictions for a batch of CAV requests."""

    service_name = "movement_controller"
    priority = 100

    @property
    def capability_bindings(self) -> CapabilityBindings:
        return {}

    def __init__(self, priority: int = 100) -> None:
        """
        Initialize the AIM-backed behavior service.
        """
        self.priority = priority
        self._owner_ref: weakref.ReferenceType[VehicleManager] | None = None
        self._target_position: Location | None = None

    def _get_owner(self) -> VehicleManager:
        owner_ref = self._owner_ref
        if owner_ref is None:
            raise RuntimeError("AIM server is not attached to an owner.")

        owner = owner_ref()
        if owner is None:
            raise RuntimeError("AIM server owner is no longer available.")

        return owner

    def on_attach(self, owner: VehicleManager) -> None:
        """Initialize the service for a particular participant instance."""
        self._owner_ref = weakref.ref(owner)

    def get_state(self) -> MovementControllerState:
        owner = self._get_owner()
        return MovementControllerState(
            service_name=self.service_name,
            owner_id=owner.id,
            is_attached=owner is not None,
            target_position=self._target_position,
        )

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None
        self._target_position = None

    def _filter_messages(self, messages: Sequence[TransportMessage[MovementControllerRequestMessage]]) -> list[MovementControllerRequestMessage]:
        owner = self._get_owner()
        valid_messages = []
        for message in messages:
            if message.dst_owner_id == owner.id and message.src_owner_id == owner.id and message.dst_service_type == self.service_name:
                valid_messages.append(message.payload)
        return valid_messages

    def process(self, messages: Sequence[TransportMessage[MovementControllerRequestMessage]]) -> Sequence[TransportMessage]:
        owner = self._get_owner()
        valid_messages = self._filter_messages(messages)
        self._target_position = None

        if len(valid_messages) > 0:
            # TODO: think what to do if multiple messages with different target positions are received - for now we just take the last one
            request = valid_messages[-1]
            self._target_position = request.target_location
            owner.control(target_speed=request.target_speed, target_location=request.target_location)
        return ()
