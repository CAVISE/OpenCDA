"""AIM server behavior service implementation."""

from __future__ import annotations

import weakref
import traci
import logging
from typing import Sequence, TYPE_CHECKING

from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from opencda.core.application.behavior.services.aim_server import AIMServerRequest, AIMServerResponse
from opencda.core.application.behavior.services.movement_controller import MovementControllerRequestMessage
from opencda.core.application.behavior.types import Rotation, Transform

if TYPE_CHECKING:
    from opencda.core.common.vehicle_manager import VehicleManager


logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_client")


@BehaviorServiceRegistry.register
class AIMClient:
    """Behavior service that runs AIM predictions for a batch of CAV requests."""

    service_name = "aim_client"

    def __init__(
        self,
    ) -> None:
        """
        Initialize the AIM-backed behavior service.
        """
        self._owner_ref: weakref.ReferenceType[VehicleManager] | None = None

    def get_owner(self) -> VehicleManager:
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

    def _filter_messages(self, messages: Sequence[TransportMessage[AIMServerRequest]]) -> list[AIMServerRequest]:
        owner = self.get_owner()
        valid_messages = []
        for message in messages:
            if message.dst_owner_id == owner.id and message.dst_service_type == self.service_name:
                valid_messages.append(message.payload)
        return valid_messages

    def process(
        self, messages: Sequence[TransportMessage[AIMServerRequest]]
    ) -> Sequence[TransportMessage[AIMServerResponse | MovementControllerRequestMessage]]:
        owner = self.get_owner()
        res_messages = []

        for message in self._filter_messages(messages):
            target_position = Transform(message.next_position, Rotation(0, 0, 0))
            payload = MovementControllerRequestMessage(target_position=target_position)
            res_messages.append(
                TransportMessage(
                    src_owner_id=owner.id,
                    src_service_type=self.service_name,
                    dst_owner_id=owner.id,
                    dst_service_type="movement_controller",
                    payload=payload,
                )
            )
        if len(res_messages) == 0:
            target_position = Transform(owner.agent.end_waypoint.transform.location, Rotation(0, 0, 0))
            payload = MovementControllerRequestMessage(target_position=target_position)
            res_messages.append(
                TransportMessage(
                    src_owner_id=owner.id,
                    src_service_type=self.service_name,
                    dst_owner_id=owner.id,
                    dst_service_type="movement_controller",
                    payload=payload,
                )
            )

        pos = owner.vehicle.get_transform()
        waypoints = owner.agent.get_local_planner().get_waypoint_buffer()
        payload = AIMServerRequest(
            vehicle_id=owner.id,
            position=pos,
            speed=traci.vehicle.getSpeed(owner.id),
            yaw=traci.vehicle.getAngle(owner.id),
            waypoints=waypoints,
        )
        res_messages.append(
            TransportMessage(
                src_owner_id=owner.id,
                src_service_type=self.service_name,
                dst_owner_id="broadcast",
                dst_service_type="aim_server",
                payload=payload,
            )
        )

        return res_messages
