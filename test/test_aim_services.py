from __future__ import annotations

import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


services_path = Path(__file__).resolve().parents[1] / "opencda" / "core" / "application" / "behavior" / "services"
behavior_services_stub = types.ModuleType("opencda.core.application.behavior.services")
behavior_services_stub.__all__ = []
behavior_services_stub.__path__ = [str(services_path)]
sys.modules.setdefault("opencda.core.application.behavior.services", behavior_services_stub)

traci_stub = types.ModuleType("traci")
traci_stub.vehicle = SimpleNamespace(
    getSpeed=lambda vehicle_id: 12.5,
    getAngle=lambda vehicle_id: 90.0,
)
sys.modules.setdefault("traci", traci_stub)


class Color:
    def __init__(self, r: int = 0, g: int = 0, b: int = 0, a: int = 255) -> None:
        self.r = r
        self.g = g
        self.b = b
        self.a = a


carla_stub = types.ModuleType("carla")
carla_stub.Color = Color
sys.modules.setdefault("carla", carla_stub)

aim_stub = types.ModuleType("AIM")


class AIMModel:
    pass


aim_stub.AIMModel = AIMModel
aim_stub.get_model = Mock(return_value=Mock(name="aim_model"))
sys.modules.setdefault("AIM", aim_stub)

numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = object
sys.modules.setdefault("numpy", numpy_stub)

aim_model_manager_stub = types.ModuleType("opencda.core.application.behavior.services.aim_server.aim_model_manager")


class AIMModelManager:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


aim_model_manager_stub.AIMModelManager = AIMModelManager
sys.modules.setdefault("opencda.core.application.behavior.services.aim_server.aim_model_manager", aim_model_manager_stub)

from opencda.core.application.behavior.services.aim_client.service import AIMClient
from opencda.core.application.behavior.services.aim_server.messages import AIMServerRequest, AIMServerResponse
from opencda.core.application.behavior.services.aim_server.service import AIMServer
from opencda.core.application.behavior.services.movement_controller.messages import MovementControllerRequestMessage
from opencda.core.application.behavior.transport_message import TransportMessage


class DummyVehicle:
    def __init__(self, location: object, transform: object) -> None:
        self._location = location
        self._transform = transform
        self._velocity = SimpleNamespace(x=0.0, y=0.0, z=0.0)

    def get_location(self) -> object:
        return self._location

    def get_transform(self) -> object:
        return self._transform

    def get_velocity(self) -> object:
        return self._velocity


class DummyPlanner:
    def __init__(self, waypoints: tuple[object, ...]) -> None:
        self._waypoints = waypoints

    def get_waypoint_buffer(self) -> tuple[object, ...]:
        return self._waypoints


class DummyAgent:
    def __init__(self, planner: DummyPlanner) -> None:
        self._planner = planner

    def get_local_planner(self) -> DummyPlanner:
        return self._planner


class DummyOwner:
    def __init__(self, owner_id: str, vehicle: DummyVehicle, agent: DummyAgent) -> None:
        self.id = owner_id
        self.vehicle = vehicle
        self.agent = agent


def _make_location(name: str) -> object:
    return SimpleNamespace(name=name, x=0.0, y=0.0, z=0.0)


def test_aim_client_process_routes_messages_through_capability_steps() -> None:
    current_location = _make_location("current")
    transform = SimpleNamespace(location=current_location)
    response_locations = tuple(_make_location(name) for name in ("origin", "target-1", "target-2"))

    client = AIMClient()
    owner = DummyOwner(
        owner_id="veh-1",
        vehicle=DummyVehicle(location=current_location, transform=transform),
        agent=DummyAgent(DummyPlanner(waypoints=("wp-1", "wp-2"))),
    )
    client.on_attach(owner)

    valid_response = TransportMessage(
        src_owner_id="rsu-1",
        src_service_type="aim_server",
        dst_owner_id="veh-1",
        dst_service_type="aim_client",
        payload=AIMServerResponse(trajectory=response_locations),
    )
    ignored_response = TransportMessage(
        src_owner_id="rsu-1",
        src_service_type="aim_server",
        dst_owner_id="veh-2",
        dst_service_type="aim_client",
        payload=AIMServerResponse(trajectory=response_locations),
    )

    with patch("opencda.core.application.behavior.services.aim_client.service.get_speed", return_value=7.0), patch(
        "opencda.core.application.behavior.services.aim_client.service.calculate_target_speeds",
        return_value=(3.5, 2.5),
    ):
        outputs = client.process((ignored_response, valid_response))

    assert len(outputs) == 2

    command_message, request_message = outputs
    assert command_message.payload == MovementControllerRequestMessage(
        target_location=response_locations[1],
        target_speed=3.5,
    )
    assert request_message.dst_service_type == "aim_server"
    assert request_message.payload == AIMServerRequest(
        vehicle_id="veh-1",
        position=transform,
        speed=12.5,
        yaw=90.0,
        waypoints=("wp-1", "wp-2"),
    )
    assert client.trajectory == deque(((response_locations[2], 2.5),))


def test_aim_client_process_uses_buffered_trajectory_without_new_responses() -> None:
    current_location = _make_location("current")
    transform = SimpleNamespace(location=current_location)
    buffered_target = _make_location("buffered-target")

    client = AIMClient()
    owner = DummyOwner(
        owner_id="veh-1",
        vehicle=DummyVehicle(location=current_location, transform=transform),
        agent=DummyAgent(DummyPlanner(waypoints=("wp-1", "wp-2"))),
    )
    client.on_attach(owner)
    client.trajectory = deque(((buffered_target, 4.0),))

    outputs = client.process(())

    assert len(outputs) == 2
    assert outputs[0].payload == MovementControllerRequestMessage(
        target_location=buffered_target,
        target_speed=4.0,
    )
    assert outputs[1].dst_service_type == "aim_server"
    assert not client.trajectory


def test_aim_server_process_passes_observed_requests_to_response_submit() -> None:
    server = AIMServer()
    observed_request = TransportMessage(
        src_owner_id="veh-1",
        src_service_type="aim_client",
        dst_owner_id="broadcast",
        dst_service_type="aim_server",
        payload=AIMServerRequest(
            vehicle_id="veh-1",
            position=SimpleNamespace(location=_make_location("current")),
            speed=12.5,
            yaw=90.0,
            waypoints=("wp-1",),
        ),
    )
    expected_response = TransportMessage(
        src_owner_id="rsu-1",
        src_service_type="aim_server",
        dst_owner_id="veh-1",
        dst_service_type="aim_client",
        payload=AIMServerResponse(trajectory=(_make_location("next"),)),
    )

    server.aim_model_manager = Mock()
    server.aim_model_manager.process.return_value = (expected_response,)

    outputs = server.process((observed_request,))

    server.aim_model_manager.process.assert_called_once_with((observed_request,))
    assert outputs == (expected_response,)
