"""AIM server behavior service implementation."""

from __future__ import annotations

import weakref
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import traci

from opencda.core.application.behavior.capability import Capability, CapabilityBindings
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from opencda.core.application.behavior.services.aim_server import AIMServerRequest, AIMServerResponse
from opencda.core.application.behavior.services.movement_controller import MovementControllerRequestMessage
from opencda.core.application.behavior.types import Location, Rotation, Transform
from opencda.core.application.behavior.services.aim_client.types import AIMClientState

if TYPE_CHECKING:
    from opencda.core.common.vehicle_manager import VehicleManager


logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_client")


@BehaviorServiceRegistry.register
class AIMClient:
    """Behavior service that runs AIM predictions for a batch of CAV requests."""

    service_name = "aim_client"
    priority = 20

    @property
    def capability_bindings(self) -> CapabilityBindings:
        return {
            Capability.RESPONSE_OBSERVE: self._filter_messages,
            Capability.COMMAND_SUBMIT: self._build_movement_command_message,
            Capability.REQUEST_SUBMIT: self._build_aim_server_request_message,
        }

    def __init__(
        self,
    ) -> None:
        """
        Initialize the AIM-backed behavior service.
        """
        self._owner_ref: weakref.ReferenceType[VehicleManager] | None = None
        self._next_position: Location | None = None

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

    def get_state(self) -> AIMClientState:
        owner_ref = self._owner_ref
        owner = owner_ref() if owner_ref is not None else None
        return AIMClientState(
            service_name=self.service_name,
            owner_id=owner.id if owner is not None else None,
            is_attached=owner is not None,
            next_position=self._next_position,
        )

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None
        self._next_position = None

    def _filter_messages(self, messages: Sequence[TransportMessage[AIMServerResponse]]) -> list[AIMServerResponse]:
        owner = self._get_owner()
        valid_messages: list[AIMServerResponse] = []
        for message in messages:
            if message.dst_owner_id == owner.id and message.dst_service_type == self.service_name:
                valid_messages.append(message.payload)
        return valid_messages

    def _build_movement_command_message(self, target_position: Transform) -> TransportMessage[MovementControllerRequestMessage]:
        owner = self._get_owner()
        payload = MovementControllerRequestMessage(target_position=target_position)
        return TransportMessage(
            src_owner_id=owner.id,
            src_service_type=self.service_name,
            dst_owner_id=owner.id,
            dst_service_type="movement_controller",
            payload=payload,
        )

    def _build_aim_server_request_message(self) -> TransportMessage[AIMServerRequest]:
        owner = self._get_owner()
        position = owner.vehicle.get_transform()
        waypoints = owner.agent.get_local_planner().get_waypoint_buffer()
        payload = AIMServerRequest(
            vehicle_id=owner.id,
            position=position,
            speed=traci.vehicle.getSpeed(owner.id),
            yaw=traci.vehicle.getAngle(owner.id),
            waypoints=waypoints,
        )
        return TransportMessage(
            src_owner_id=owner.id,
            src_service_type=self.service_name,
            dst_owner_id="broadcast",
            dst_service_type="aim_server",
            payload=payload,
        )

    def process(
        self,
        messages: Sequence[TransportMessage[AIMServerResponse]],
    ) -> Sequence[TransportMessage[MovementControllerRequestMessage | AIMServerRequest]]:
        owner = self._get_owner()
        self._next_position = None
        res_messages: list[TransportMessage[MovementControllerRequestMessage | AIMServerRequest]] = []

        for message in self._filter_messages(messages):
            self._next_position = message.next_position
            target_position = Transform(message.next_position, Rotation(0, 0, 0))
            res_messages.append(self._build_movement_command_message(target_position))
        if len(res_messages) == 0:
            end_waypoint = owner.agent.end_waypoint
            if end_waypoint is None:
                raise RuntimeError("AIM client requires a valid end waypoint when no AIM response is available.")
            self._next_position = end_waypoint.transform.location
            target_position = Transform(end_waypoint.transform.location, Rotation(0, 0, 0))
            res_messages.append(self._build_movement_command_message(target_position))

        res_messages.append(self._build_aim_server_request_message())

        return res_messages
