"""Movement-command contracts and interception before command execution."""

from dataclasses import replace
from typing import Any
from unittest.mock import Mock, call

import pytest

from opencda.core.application.behavior import BROADCAST_SERVICE_TYPE, Capability, TransportMessage
from opencda.core.application.behavior.services.default_movement_request import DefaultMovementRequest
from opencda.core.application.behavior.services.movement_controller import MovementController, MovementControllerRequestMessage
from opencda.core.application.behavior.types import Location
from opencda.core.attack.adversary_framework import AttackStageRegistry, Status
from opencda.core.common.agent_manager import AgentManager


class _Agent:
    use_carla_autopilot = False

    def __init__(self) -> None:
        self.control = Mock()


@pytest.fixture
def movement_services():
    agent = _Agent()
    fallback = DefaultMovementRequest()
    controller = MovementController()
    owner = AgentManager(agent, "node-1", behavior_services=(fallback, controller))
    return owner, agent, fallback, controller


def _command(speed: float = 20.0, location: Location = Location(x=1.0)) -> TransportMessage[MovementControllerRequestMessage]:
    return TransportMessage(
        src_owner_id="node-1",
        src_service_type="aim_client",
        dst_owner_id="node-1",
        dst_service_type="movement_controller",
        payload=MovementControllerRequestMessage(target_speed=speed, target_location=location),
    )


def _install_stage(request, name, service, capability, params: dict[str, Any] | None = None):
    options: dict[str, Any] = {"capabilities": (capability.value,)}
    if params is not None:
        options["params"] = params
    stage = AttackStageRegistry.create_stage(name, **options)
    request.addfinalizer(stage.deactivate)
    assert stage.execute((service,)).status == Status.SUCCESS
    return stage


def test_fallback_publishes_complete_local_command(movement_services) -> None:
    owner, _, fallback, _ = movement_services

    (message,) = fallback.process(())

    assert message.src_owner_id == owner.id
    assert message.dst_owner_id == owner.id
    assert message.src_service_type == "default_movement_request"
    assert message.dst_service_type == "movement_controller"
    assert message.payload == MovementControllerRequestMessage(target_speed=None, target_location=None)


def test_controller_observes_complete_messages_in_order(movement_services) -> None:
    _, _, _, controller = movement_services
    first = _command(10.0)
    second = replace(_command(20.0), dst_service_type=BROADCAST_SERVICE_TYPE)

    observed = controller.capability_bindings[Capability.COMMAND_OBSERVE]((first, second))

    assert isinstance(observed, tuple)
    assert observed == (first, second)
    assert observed[0] is first
    assert observed[1] is second


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"src_owner_id": "other-node"},
        {"dst_owner_id": "other-node"},
        {"dst_service_type": "other-service"},
        {"payload": "not a movement command"},
    ],
)
def test_controller_rejects_invalid_commands_before_selection(movement_services, invalid_fields) -> None:
    _, agent, _, controller = movement_services
    valid = _command(20.0)
    invalid = replace(_command(99.0), **invalid_fields)

    controller.process((valid, invalid))

    agent.control.assert_called_once_with(target_speed=20.0, target_location=valid.payload.target_location)


@pytest.mark.parametrize("reverse", [False, True])
def test_controller_keeps_last_command_wins(movement_services, reverse) -> None:
    _, agent, fallback, controller = movement_services
    (fallback_message,) = fallback.process(())
    aim_message = _command()
    messages = (aim_message, fallback_message) if reverse else (fallback_message, aim_message)
    expected = messages[-1].payload

    assert controller.process(messages) == ()

    agent.control.assert_called_once_with(target_speed=expected.target_speed, target_location=expected.target_location)
    assert controller.get_state().target_position == expected.target_location


def test_controller_does_not_control_without_commands(movement_services) -> None:
    _, agent, _, controller = movement_services

    assert controller.process(()) == ()

    agent.control.assert_not_called()
    assert controller.get_state().target_position is None


def test_fallback_suppression_routes_an_empty_batch_and_restores(request, movement_services) -> None:
    owner, agent, fallback, _ = movement_services
    stage = _install_stage(request, "dropper", fallback, Capability.COMMAND_SUBMIT, {"drop_rate": 1.0})

    assert fallback.process(()) == ()
    owner.update_behavior_services(())
    agent.control.assert_not_called()

    stage.deactivate()
    owner.update_behavior_services(())

    agent.control.assert_called_once_with(target_speed=None, target_location=None)


def test_spoofed_fallback_is_delivered_to_controller(request, movement_services) -> None:
    owner, agent, fallback, _ = movement_services
    _install_stage(
        request,
        "spoofer",
        fallback,
        Capability.COMMAND_SUBMIT,
        {"rewrites": [{"path": "payload.target_speed", "operation": "set", "value": 50.0}]},
    )

    owner.update_behavior_services(())

    agent.control.assert_called_once_with(target_speed=50.0, target_location=None)


def test_later_aim_command_overrides_spoofed_fallback(request, movement_services) -> None:
    _, agent, fallback, controller = movement_services
    _install_stage(
        request,
        "spoofer",
        fallback,
        Capability.COMMAND_SUBMIT,
        {"rewrites": [{"path": "payload.target_speed", "operation": "set", "value": 50.0}]},
    )
    aim_message = _command(20.0)

    controller.process((*fallback.process(()), aim_message))

    agent.control.assert_called_once_with(target_speed=20.0, target_location=aim_message.payload.target_location)


def test_controller_spoofing_changes_control_without_mutating_input(request, movement_services) -> None:
    _, agent, _, controller = movement_services
    message = _command()
    _install_stage(
        request,
        "spoofer",
        controller,
        Capability.COMMAND_OBSERVE,
        {
            "rewrites": [
                {"path": "payload.target_speed", "operation": "set", "value": 50.0},
                {"path": "payload.target_location.x", "operation": "set", "value": 25.0},
            ]
        },
    )

    controller.process((message,))

    agent.control.assert_called_once_with(target_speed=50.0, target_location=Location(x=25.0))
    assert controller.get_state().target_position == Location(x=25.0)
    assert message.payload.target_speed == 20.0
    assert message.payload.target_location == Location(x=1.0)


def test_controller_suppression_skips_control_and_clears_target(request, movement_services) -> None:
    _, agent, _, controller = movement_services
    message = _command()
    controller.process((message,))
    agent.control.reset_mock()
    _install_stage(request, "dropper", controller, Capability.COMMAND_OBSERVE, {"drop_rate": 1.0})

    assert controller.process((message,)) == ()

    agent.control.assert_not_called()
    assert controller.get_state().target_position is None


def test_partial_suppression_can_leave_the_fallback_command(request, movement_services) -> None:
    _, agent, fallback, controller = movement_services
    _install_stage(request, "dropper", controller, Capability.COMMAND_OBSERVE, {"drop_rate": 0.5, "seed": 10})

    controller.process((*fallback.process(()), _command()))

    agent.control.assert_called_once_with(target_speed=None, target_location=None)


def test_controller_replays_previous_command_batch(request, movement_services) -> None:
    _, agent, _, controller = movement_services
    _install_stage(request, "replayer", controller, Capability.COMMAND_OBSERVE)
    first = _command(10.0, Location(x=1.0))
    second = _command(20.0, Location(x=2.0))
    third = _command(30.0, Location(x=3.0))

    controller.process((first,))
    controller.process((second,))
    controller.process((third,))

    assert agent.control.call_args_list == [
        call(target_speed=10.0, target_location=Location(x=1.0)),
        call(target_speed=10.0, target_location=Location(x=1.0)),
        call(target_speed=20.0, target_location=Location(x=2.0)),
    ]
    assert controller.get_state().target_position == Location(x=2.0)


@pytest.mark.parametrize("binding", ["fallback", "controller"])
def test_sniffer_observes_complete_command_batches_without_changing_control(request, movement_services, binding) -> None:
    _, agent, fallback, controller = movement_services
    service, capability = (fallback, Capability.COMMAND_SUBMIT) if binding == "fallback" else (controller, Capability.COMMAND_OBSERVE)
    stage = _install_stage(request, "sniffer", service, capability)
    messages = fallback.process(()) if binding == "fallback" else (_command(),)

    controller.process(messages)

    assert len(stage.observed_outputs) == 1
    assert stage.observed_outputs[0].capability == capability
    assert stage.observed_outputs[0].output == messages
    expected = messages[-1].payload
    agent.control.assert_called_once_with(target_speed=expected.target_speed, target_location=expected.target_location)


@pytest.mark.parametrize("name", ["spoofer", "dropper", "replayer", "sniffer"])
def test_controller_stage_deactivation_restores_current_commands(request, movement_services, name) -> None:
    _, agent, _, controller = movement_services
    params = None
    if name == "spoofer":
        params = {"rewrites": [{"path": "payload.target_speed", "operation": "set", "value": 99.0}]}
    elif name == "dropper":
        params = {"drop_rate": 1.0}
    stage = _install_stage(request, name, controller, Capability.COMMAND_OBSERVE, params)
    controller.process((_command(10.0),))
    controller.process((_command(20.0),))

    stage.deactivate()
    agent.control.reset_mock()
    controller.process((_command(30.0),))

    agent.control.assert_called_once_with(target_speed=30.0, target_location=Location(x=1.0))
