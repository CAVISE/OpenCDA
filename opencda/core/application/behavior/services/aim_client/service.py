"""AIM server behavior service implementation."""

from __future__ import annotations

import weakref
import logging
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import traci
import carla

from opencda.core.application.behavior.capability import Capability, CapabilityBindings
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from opencda.core.application.behavior.services.aim_server import AIMServerRequest, AIMServerResponse
from opencda.core.application.behavior.services.movement_controller import MovementControllerRequestMessage
from .types import AIMClientState

from .utils import get_speed, draw_trajetory_points, calculate_target_speeds

if TYPE_CHECKING:
    from opencda.core.application.behavior.types import Location, Transform
    from opencda.core.common.vehicle_manager import VehicleManager


logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_client")


@BehaviorServiceRegistry.register
class AIMClient:
    """Behavior service that runs AIM predictions for a batch of CAV requests."""

    service_name: str = "aim_client"
    priority: int = 20

    @property
    def capability_bindings(self) -> CapabilityBindings:
        return {
            Capability.RESPONSE_OBSERVE: self._filter_messages,
            Capability.COMMAND_SUBMIT: self._build_movement_command_message,
            Capability.REQUEST_SUBMIT: self._build_aim_server_request_message,
        }

    def __init__(self, priority: int = 20, debug: bool = False) -> None:
        """
        Initialize the AIM-backed behavior service.
        """
        self._owner_ref: weakref.ReferenceType[VehicleManager] | None = None
        self.priority = priority
        self.trajectory: deque[tuple[Location, float]] = deque()
        self.debug = debug

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
        owner_ref = self._get_owner()
        owner = owner_ref() if owner_ref is not None else None
        return AIMClientState(
            service_name=self.service_name,
            owner_id=owner.id if owner is not None else None,
            is_attached=owner is not None,
        )

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None
        self.trajectory.clear()

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
        res_messages: list[TransportMessage[MovementControllerRequestMessage | AIMServerRequest]] = []

        current_location = owner.vehicle.get_location()
        current_speed = get_speed(owner.vehicle)
        for message in self._filter_messages(messages):
            control_trajectory = message.trajectory[1:]  # drop first because it was calculated on previous tick
            target_speeds = calculate_target_speeds(control_trajectory, 0.05, current_location, current_speed, 111, 2.5, 4.5)
            self.trajectory = deque(zip(control_trajectory, target_speeds))
            if self.debug:
                draw_trajetory_points(
                    owner.agent._local_planner._vehicle.get_world(),
                    control_trajectory,
                    size=0.05,
                    color=carla.Color(255, 0, 0),
                    life_time=0.1,
                    _map=owner.agent._map,
                )
            target_location, target_speed = self.trajectory.popleft()
            payload = MovementControllerRequestMessage(target_location=target_location, target_speed=target_speed)
            res_messages.append(
                TransportMessage(
                    src_owner_id=owner.id,
                    src_service_type=self.service_name,
                    dst_owner_id=owner.id,
                    dst_service_type="movement_controller",
                    payload=payload,
                )
            )
        if len(res_messages) == 0 and self.trajectory:
            target_location, target_speed = self.trajectory.popleft()
            payload = MovementControllerRequestMessage(target_location=target_location, target_speed=target_speed)
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
        server_request_payload = AIMServerRequest(
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
                payload=server_request_payload,
            )
        )

        return res_messages
