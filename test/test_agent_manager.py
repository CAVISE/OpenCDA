"""Unit tests for the common AgentManager contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import Mock

import pytest

from opencda.core.application.behavior import (
    BROADCAST_OWNER_ID,
    BROADCAST_SERVICE_TYPE,
    TransportMessage,
)
from opencda.core.common.agent import AgentType
from opencda.core.common.agent_manager import AgentManager
from opencda.core.sensing.sensor_types import SensorActorBundle


class StubBehaviorService:
    def __init__(
        self,
        service_type: str,
        priority: int = 100,
        process: Callable[[Sequence[TransportMessage[Any]]], tuple[TransportMessage[Any], ...]] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.service_type = service_type
        self.priority = priority
        self._process = process or (lambda messages: ())
        self._events = events
        self.owner: AgentManager | None = None
        self.received_messages: list[tuple[TransportMessage[Any], ...]] = []
        self.state: Any = {"service_type": service_type}
        self.attach_error: Exception | None = None
        self.detach_error: Exception | None = None

    @property
    def capability_bindings(self) -> dict[Any, Any]:
        return {}

    def on_attach(self, owner: Any) -> None:
        if self.attach_error is not None:
            raise self.attach_error
        self.owner = owner
        if self._events is not None:
            self._events.append(f"attach:{self.service_type}")

    def get_state(self) -> Any:
        return self.state

    def process(self, messages: Sequence[TransportMessage[Any]]) -> tuple[TransportMessage[Any], ...]:
        message_batch = tuple(messages)
        self.received_messages.append(message_batch)
        return self._process(message_batch)

    def on_detach(self) -> None:
        if self._events is not None:
            self._events.append(f"detach:{self.service_type}")
        self.owner = None
        if self.detach_error is not None:
            raise self.detach_error


class StubAgent:
    use_carla_autopilot = False

    def __init__(self, events: list[str] | None = None, error: Exception | None = None) -> None:
        self._events = events
        self._error = error

    def destroy(self) -> None:
        if self._events is not None:
            self._events.append("destroy:agent")
        if self._error is not None:
            raise self._error


def message(
    *,
    source_owner: str = "peer",
    source_service: str = "source",
    destination_owner: str = "agent-1",
    destination_service: str = "first",
    payload: Any = None,
) -> TransportMessage[Any]:
    return TransportMessage(
        src_owner_id=source_owner,
        src_service_type=source_service,
        dst_owner_id=destination_owner,
        dst_service_type=destination_service,
        payload=payload,
    )


def test_initialization_sorts_and_attaches_behavior_services() -> None:
    events: list[str] = []
    late = StubBehaviorService("late", priority=20, events=events)
    early = StubBehaviorService("early", priority=10, events=events)
    agent = StubAgent()

    manager = AgentManager(agent, "agent-1", behavior_services=[late, early])

    assert manager.agent is agent
    assert manager.id == "agent-1"
    assert manager.behavior_services == (early, late)
    assert early.owner is manager
    assert late.owner is manager
    assert events == ["attach:early", "attach:late"]


@pytest.mark.parametrize("agent_id", ["", None, 1])
def test_initialization_rejects_invalid_agent_id(agent_id: Any) -> None:
    with pytest.raises(ValueError, match="agent_id must be a non-empty string"):
        AgentManager(StubAgent(), agent_id)


def test_initialization_rejects_duplicate_behavior_service_type() -> None:
    with pytest.raises(ValueError, match="Duplicate behavior service ID"):
        AgentManager(
            StubAgent(),
            "agent-1",
            behavior_services=[StubBehaviorService("duplicate"), StubBehaviorService("duplicate")],
        )


def test_initialization_rejects_non_integer_priority() -> None:
    service = StubBehaviorService("invalid")
    service.priority = "high"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="must define an integer priority"):
        AgentManager(StubAgent(), "agent-1", behavior_services=[service])


def test_initialization_rejects_object_outside_behavior_service_protocol() -> None:
    with pytest.raises(TypeError, match="must implement the BehaviorService protocol"):
        AgentManager(StubAgent(), "agent-1", behavior_services=[object()])  # type: ignore[list-item]


def test_create_cav_builds_vehicle_agent_through_common_factory(mocker, minimal_vehicle_config, mock_cav_world) -> None:
    actor = Mock()
    actor.id = 10
    carla_map = Mock()
    localizer = Mock()
    perception_manager = Mock()
    v2x_manager = Mock()
    map_manager = Mock()
    safety_manager = Mock()
    behavior_agent = Mock()
    controller = Mock()
    collision_sensor_actor = Mock()
    sensor_actors = SensorActorBundle(collision=collision_sensor_actor)

    localizer_factory = mocker.patch("opencda.core.common.agent_manager.create_localizer", return_value=localizer)
    perception_factory = mocker.patch("opencda.core.common.agent_manager.PerceptionManager", return_value=perception_manager)
    mocker.patch("opencda.core.common.agent_manager.V2XManager", return_value=v2x_manager)
    mocker.patch("opencda.core.common.agent_manager.MapManager", return_value=map_manager)
    safety_factory = mocker.patch("opencda.core.common.agent_manager.SafetyManager", return_value=safety_manager)
    mocker.patch("opencda.core.common.agent_manager.BehaviorAgent", return_value=behavior_agent)
    mocker.patch("opencda.core.common.agent_manager.ControlManager", return_value=controller)

    config = {**minimal_vehicle_config, "id": 7, "behavior_services": []}
    manager = AgentManager.create(
        actor=actor,
        config_yaml=config,
        carla_map=carla_map,
        cav_world=mock_cav_world,
        agent_type=AgentType.CAV,
        application=["single"],
        id_prefix="cav",
        sensor_actors=sensor_actors,
    )

    assert manager.id == "cav-7"
    assert manager.agent.actor is actor
    assert manager.agent.is_vehicle is True
    assert manager.agent.v2x_manager is v2x_manager
    localizer_factory.assert_called_once_with(
        actor,
        config["sensing"]["localization"],
        carla_map,
        use_imu=True,
        sensor_actors=sensor_actors,
    )
    assert perception_factory.call_args.kwargs["vehicle"] is actor
    assert perception_factory.call_args.kwargs["sensor_actors"] is sensor_actors
    assert safety_factory.call_args.kwargs["collision_sensor_actor"] is collision_sensor_actor
    mock_cav_world.update_agent_manager.assert_called_once_with(manager)


def test_create_rsu_uses_same_factory_without_vehicle_components(mocker, minimal_rsu_config, mock_cav_world) -> None:
    actor = Mock()
    actor.id = 20
    actor.get_world.return_value = "world"
    carla_map = Mock()
    localizer = Mock()
    perception_manager = Mock()

    localizer_factory = mocker.patch("opencda.core.common.agent_manager.create_localizer", return_value=localizer)
    perception_factory = mocker.patch("opencda.core.common.agent_manager.PerceptionManager", return_value=perception_manager)

    config = {**minimal_rsu_config, "id": 3}
    manager = AgentManager.create(
        actor=actor,
        config_yaml=config,
        carla_map=carla_map,
        cav_world=mock_cav_world,
        agent_type=AgentType.RSU,
    )

    assert manager.id == "rsu-3"
    assert manager.agent.actor is actor
    assert manager.agent.is_vehicle is False
    localizer_factory.assert_called_once_with(actor, config["sensing"]["localization"], carla_map, use_imu=False)
    assert perception_factory.call_args.kwargs["vehicle"] is None
    assert perception_factory.call_args.kwargs["carla_world"] == "world"
    mock_cav_world.update_agent_manager.assert_called_once_with(manager)


def test_agent_ids_are_unique_in_shared_registry() -> None:
    assert AgentManager._allocate_id(1, "cav", True) == "cav-1"
    assert AgentManager._allocate_id(1, "rsu", True) == "rsu-1"
    assert AgentManager._allocate_id(1, "cav", True) == "cav-2"


def test_attach_failure_detaches_already_attached_services_in_reverse_order() -> None:
    events: list[str] = []
    first = StubBehaviorService("first", priority=1, events=events)
    second = StubBehaviorService("second", priority=2, events=events)
    failing = StubBehaviorService("failing", priority=3, events=events)
    failing.attach_error = RuntimeError("attach failed")

    with pytest.raises(RuntimeError, match="attach failed"):
        AgentManager(StubAgent(), "agent-1", behavior_services=[failing, second, first])

    assert events == ["attach:first", "attach:second", "detach:second", "detach:first"]


def test_update_routes_direct_broadcast_self_and_outgoing_messages() -> None:
    outgoing = message(
        source_owner="agent-1",
        source_service="first",
        destination_owner="peer-2",
        destination_service="remote",
        payload="outgoing",
    )
    self_message = message(
        source_owner="agent-1",
        source_service="first",
        destination_owner="agent-1",
        destination_service="second",
        payload="self",
    )
    first = StubBehaviorService("first", priority=1, process=lambda messages: (self_message, outgoing))
    second = StubBehaviorService("second", priority=2)
    first.state = "first-state"
    second.state = "second-state"
    manager = AgentManager(StubAgent(), "agent-1", behavior_services=[second, first])

    direct = message(destination_service="first", payload="direct")
    broadcast = message(
        destination_owner=BROADCAST_OWNER_ID,
        destination_service=BROADCAST_SERVICE_TYPE,
        payload="broadcast",
    )
    ignored_other_owner = message(destination_owner="agent-2", destination_service="first", payload="other")
    ignored_own_broadcast = message(
        source_owner="agent-1",
        destination_owner=BROADCAST_OWNER_ID,
        destination_service=BROADCAST_SERVICE_TYPE,
        payload="own-broadcast",
    )

    manager.update_behavior_services([direct, broadcast, ignored_other_owner, ignored_own_broadcast])

    assert first.received_messages == [(direct, broadcast)]
    assert second.received_messages == [(self_message, broadcast)]
    assert manager.behavior_service_results == [outgoing]
    assert manager.behavior_service_states == {
        "first": "first-state",
        "second": "second-state",
    }


def test_update_clears_previous_outgoing_results() -> None:
    responses: list[tuple[TransportMessage[Any], ...]] = [
        (
            message(
                source_owner="agent-1",
                destination_owner="peer-2",
                destination_service="remote",
            ),
        ),
        (),
    ]
    service = StubBehaviorService("first", process=lambda messages: responses.pop(0))
    manager = AgentManager(StubAgent(), "agent-1", behavior_services=[service])

    manager.update_behavior_services([])
    assert len(manager.behavior_service_results) == 1

    manager.update_behavior_services([])
    assert manager.behavior_service_results == []


def test_update_rejects_unknown_local_service() -> None:
    manager = AgentManager(StubAgent(), "agent-1", behavior_services=[StubBehaviorService("known")])

    with pytest.raises(ValueError, match="unknown service_type 'unknown'"):
        manager.update_behavior_services([message(destination_service="unknown")])


def test_update_rejects_non_transport_message() -> None:
    manager = AgentManager(StubAgent(), "agent-1")

    with pytest.raises(TypeError, match="must be a TransportMessage"):
        manager.update_behavior_services([object()])  # type: ignore[list-item]


def test_destroy_detaches_services_then_agent_once() -> None:
    events: list[str] = []
    first = StubBehaviorService("first", priority=1, events=events)
    second = StubBehaviorService("second", priority=2, events=events)
    agent = StubAgent(events)
    manager = AgentManager(
        agent,
        "agent-1",
        behavior_services=[first, second],
    )
    events.clear()

    manager.destroy()
    manager.destroy()

    assert events == [
        "detach:second",
        "detach:first",
        "destroy:agent",
    ]


def test_destroy_attempts_all_cleanup_and_raises_first_error() -> None:
    events: list[str] = []
    service = StubBehaviorService("service", events=events)
    service.detach_error = RuntimeError("detach failed")
    agent = StubAgent(events, RuntimeError("agent destroy failed"))
    manager = AgentManager(
        agent,
        "agent-1",
        behavior_services=[service],
    )
    events.clear()

    with pytest.raises(RuntimeError, match="detach failed"):
        manager.destroy()

    assert events == [
        "detach:service",
        "destroy:agent",
    ]
