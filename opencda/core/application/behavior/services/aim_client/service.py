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
    from opencda.core.application.behavior.types import Location
    from opencda.core.common.vehicle_manager import VehicleManager


logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_client")


@BehaviorServiceRegistry.register
class AIMClient:
    """Behavior service that runs AIM predictions for a batch of CAV requests."""

    service_type: str = "aim_client"
    priority: int = 20

    @property
    def capability_bindings(self) -> CapabilityBindings:
        return {
            Capability.RESPONSE_OBSERVE: self._observe_aim_responses,
            Capability.COMMAND_SUBMIT: self._build_movement_command_messages,
            Capability.REQUEST_SUBMIT: self._build_aim_server_request_messages,
        }

    def __init__(self, priority: int = 20, debug: bool = False) -> None:
        """
        Initialize the AIM-backed behavior service.
        """
        self._owner_ref: weakref.ReferenceType[VehicleManager] | None = None
        self.priority = priority
        self.trajectory: deque[tuple[Location, float]] = deque()
        self.server_id: str = "broadcast"
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
        return AIMClientState(
            service_type=self.service_type,
            owner_id=owner_ref.id if owner_ref is not None else None,
            is_attached=owner_ref is not None,
            trajectory=tuple(location for location, _ in self.trajectory),
        )

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None
        self.trajectory.clear()

    def _observe_aim_responses(
        self,
        messages: Sequence[TransportMessage[AIMServerResponse]],
    ) -> tuple[TransportMessage[AIMServerResponse], ...]:
        owner = self._get_owner()
        observed_messages: list[TransportMessage[AIMServerResponse]] = []

        for message in messages:
            if (
                message.dst_owner_id == owner.id
                and message.dst_service_type == self.service_type
                and (self.server_id == "broadcast" or self.server_id == message.src_owner_id)
            ):
                observed_messages.append(message)

        return tuple(observed_messages)

    def _build_movement_command_message(
        self,
        target_location: Location | None,
        target_speed: float | None = None,
    ) -> TransportMessage[MovementControllerRequestMessage]:
        owner = self._get_owner()
        payload = MovementControllerRequestMessage(target_location=target_location, target_speed=target_speed)
        return TransportMessage(
            src_owner_id=owner.id,
            src_service_type=self.service_type,
            dst_owner_id=owner.id,
            dst_service_type="movement_controller",
            payload=payload,
        )

    def _build_movement_command_messages(
        self,
        observed_responses: Sequence[TransportMessage[AIMServerResponse]],
    ) -> tuple[TransportMessage[MovementControllerRequestMessage], ...]:
        owner = self._get_owner()
        movement_commands: list[TransportMessage[MovementControllerRequestMessage]] = []
        current_location = owner.vehicle.get_location()
        current_speed = get_speed(owner.vehicle)

        if len(observed_responses) > 1:
            logger.warning(f"Received more than one aim_server messages. Amount: {len(observed_responses)}")

        if len(observed_responses) > 0:
            response = observed_responses[0]
            if self.server_id == "broadcast":
                self.server_id = response.src_owner_id
            control_trajectory = response.payload.trajectory[1:]  # drop first because it was calculated on previous tick
            target_speeds = calculate_target_speeds(control_trajectory, 0.05, current_location, current_speed, 111, 2.5, 4.5)
            self.trajectory = deque(zip(control_trajectory, target_speeds))

            if self.debug:
                self._draw_control_trajectory(owner, control_trajectory)

            target_location, target_speed = self.trajectory.popleft()
            movement_commands.append(self._build_movement_command_message(target_location, target_speed))

        if len(movement_commands) == 0:
            if self.trajectory:
                target_location, target_speed = self.trajectory.popleft()
                movement_commands.append(self._build_movement_command_message(target_location, target_speed))
            else:
                self.server_id = "broadcast"

        return tuple(movement_commands)

    def _build_aim_server_request_message(self, dst_owner_id: str = "broadcast") -> tuple[TransportMessage[AIMServerRequest], ...]:
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
        return (
            TransportMessage(
                src_owner_id=owner.id,
                src_service_type=self.service_type,
                dst_owner_id=dst_owner_id,
                dst_service_type="aim_server",
                payload=payload,
            ),
        )

    def _draw_control_trajectory(
        self,
        owner: VehicleManager,
        control_trajectory: Sequence[Location],
    ) -> None:
        """Visualize the currently active AIM trajectory in debug mode."""
        draw_trajetory_points(
            owner.agent._local_planner._vehicle.get_world(),
            control_trajectory,
            size=0.05,
            color=carla.Color(255, 0, 0),
            life_time=0.1,
        )

    def process(
        self,
        messages: Sequence[TransportMessage[AIMServerResponse]],
    ) -> tuple[TransportMessage[MovementControllerRequestMessage | AIMServerRequest], ...]:
        observed_responses = self._observe_aim_responses(messages)
        movement_commands = self._build_movement_command_messages(observed_responses)
        aim_requests = self._build_aim_server_request_messages()
        return (*movement_commands, *aim_requests)