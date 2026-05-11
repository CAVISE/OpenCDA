"""AIM server behavior service implementation."""

from __future__ import annotations

import weakref
import logging
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import traci
import carla

from opencda.core.application.behavior.capability import Capability, CapabilityBindings
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage, BROADCAST_OWNER_ID, BROADCAST_SERVICE_TYPE
from opencda.core.application.behavior.services.aim_server import AIMServerRequest, AIMServerResponse
from opencda.core.application.behavior.services.movement_controller import MovementControllerRequestMessage
from opencda.core.application.behavior.services.self_informer import SelfInformerResponse
from .types import AIMClientState

from .utils import draw_trajetory_points, calculate_target_speeds

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
        self.trajectory: deque[tuple[Location, float, float | None]] = deque()
        self.server_id: str = BROADCAST_OWNER_ID
        self.debug = debug
        self._self_informer_data: SelfInformerResponse | None = None

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
        owner = self._get_owner()
        return AIMClientState(
            service_type=self.service_type,
            owner_id=owner.id,
            trajectory=tuple(location for location, _, _ in self.trajectory),
        )

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None
        self.trajectory.clear()

    def _observe_aim_responses(
        self,
        messages: Sequence[TransportMessage[AIMServerResponse | SelfInformerResponse]],
    ) -> tuple[tuple[TransportMessage[AIMServerResponse], ...], tuple[SelfInformerResponse, ...]]:
        owner = self._get_owner()
        aim_server_responses: list[TransportMessage[AIMServerResponse]] = []
        self_informer_responses: list[SelfInformerResponse] = []

        for message in messages:
            if isinstance(message.payload, AIMServerResponse):
                if (
                    message.dst_owner_id == owner.id
                    and message.dst_service_type in (self.service_type, BROADCAST_SERVICE_TYPE)
                    and (self.server_id == BROADCAST_OWNER_ID or self.server_id == message.src_owner_id)
                ):
                    aim_server_responses.append(cast(TransportMessage[AIMServerResponse], message))
            elif isinstance(message.payload, SelfInformerResponse):
                if (
                    message.dst_owner_id == owner.id
                    and message.src_owner_id == owner.id
                    and message.dst_service_type in (self.service_type, BROADCAST_SERVICE_TYPE)
                ):
                    self_informer_responses.append(message.payload)

        return tuple(aim_server_responses), tuple(self_informer_responses)

    def _handle_self_informer_responses(
        self,
        self_informer_responses: Sequence[SelfInformerResponse],
    ) -> None:
        if len(self_informer_responses) <= 0:
            raise RuntimeError("No self_informer_responses received")
        else:
            if len(self_informer_responses) > 1:
                logger.warning(f"Received more than one self_informer responses. Amount: {len(self_informer_responses)}")
            self._self_informer_data = self_informer_responses[0]

    def _build_movement_command_message(
        self,
        target_location: Location | None,
        target_speed: float | None = None,
        target_yaw: float | None = None,
    ) -> TransportMessage[MovementControllerRequestMessage]:
        owner = self._get_owner()
        payload = MovementControllerRequestMessage(
            target_location=target_location,
            target_speed=target_speed,
            target_yaw=target_yaw,
        )
        return TransportMessage(
            src_owner_id=owner.id,
            src_service_type=self.service_type,
            dst_owner_id=owner.id,
            dst_service_type="movement_controller",
            payload=payload,
        )

    def _build_movement_command_messages(
        self,
        aim_server_responses: Sequence[TransportMessage[AIMServerResponse]],
    ) -> tuple[TransportMessage[MovementControllerRequestMessage], ...]:
        owner = self._get_owner()
        movement_commands: list[TransportMessage[MovementControllerRequestMessage]] = []

        if len(aim_server_responses) > 1:
            logger.warning(f"Received more than one aim_server messages. Amount: {len(aim_server_responses)}")

        if len(aim_server_responses) > 0:
            response = aim_server_responses[0]
            if self.server_id == BROADCAST_OWNER_ID:
                self.server_id = response.src_owner_id
            # control_trajectory = response.payload.trajectory[1:]  # drop first because it was calculated on previous tick
            control_trajectory = response.payload.trajectory

            if response.payload.speed is None:
                if self._self_informer_data is None:
                    raise RuntimeError("_self_informer_data not set")
                current_location = self._self_informer_data.location
                current_speed = self._self_informer_data.speed
                target_speeds = calculate_target_speeds(control_trajectory, 0.05, current_location, current_speed, 111, 2.5, 4.5)
                self.trajectory = deque(zip(control_trajectory, target_speeds, None))
                self.trajectory = deque((location, speed, None) for location, speed in zip(control_trajectory, target_speeds))

            else:
                target_speeds = [response.payload.speed for _ in control_trajectory]
                self.trajectory = deque((location, speed, response.payload.yaw) for location, speed in zip(control_trajectory, target_speeds))

            if self.debug:
                self._draw_control_trajectory(owner, control_trajectory)

            target_location, target_speed, target_yaw = self.trajectory.popleft()
            movement_commands.append(self._build_movement_command_message(target_location, target_speed, target_yaw))
        if len(movement_commands) == 0:
            if self.trajectory:
                target_location, target_speed, target_yaw = self.trajectory.popleft()
                movement_commands.append(self._build_movement_command_message(target_location, target_speed, target_yaw))
            else:
                self.server_id = BROADCAST_OWNER_ID

        return tuple(movement_commands)

    def _build_aim_server_request_messages(self, dst_owner_id: str = BROADCAST_OWNER_ID) -> tuple[TransportMessage[AIMServerRequest], ...]:
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
        messages: Sequence[TransportMessage[AIMServerResponse | SelfInformerResponse]],
    ) -> tuple[TransportMessage[MovementControllerRequestMessage | AIMServerRequest], ...]:
        aim_server_responses, self_informer_responses = self._observe_aim_responses(messages)
        self._handle_self_informer_responses(self_informer_responses)
        movement_commands = self._build_movement_command_messages(aim_server_responses)
        aim_requests = self._build_aim_server_request_messages()
        return (*movement_commands, *aim_requests)
