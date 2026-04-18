"""AIM server behavior service implementation."""

from __future__ import annotations

import weakref
import logging
from typing import Sequence, TYPE_CHECKING

from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from .messages import MovementControllerRequestMessage


if TYPE_CHECKING:
    from opencda.core.common.vehicle_manager import VehicleManager


logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_client")


@BehaviorServiceRegistry.register
class MovementController:
    """Behavior service that runs AIM predictions for a batch of CAV requests."""

    service_name = "movement_controller"

    def __init__(
        self,
    ) -> None:
        """
        Initialize the AIM-backed behavior service.
        """
        self._owner_ref: weakref.ReferenceType[VehicleManager] | None = None

    def _require_owner(self) -> VehicleManager:
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

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None

    def _validate_messages(self, messages: Sequence[TransportMessage[MovementControllerRequestMessage]]) -> list[MovementControllerRequestMessage]:
        owner = self._require_owner()
        valid_msgs = []
        for message in messages:
            if message.dst_owner_id == owner.id and message.src_owner_id == owner.id and message.dst_service_type == self.service_name:
                valid_msgs.append(message.payload)
        return valid_msgs

    def process(self, messages: Sequence[TransportMessage[MovementControllerRequestMessage]]) -> None:
        owner = self._require_owner()
        valid_msgs = self._validate_messages(messages)

        if len(valid_msgs) > 0:
            next_pos = [
                message.target_position.location for message in valid_msgs
            ]  # TODO: think what to do if multiple messages with different target positions are received - for now we just take the first one

            current_location = owner.vehicle.get_location()
            owner.set_destination(current_location, next_pos[0], clean=True, end_reset=False)
