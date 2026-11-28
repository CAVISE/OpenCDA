"""Tests for localization delivery through SelfInformer."""

from types import SimpleNamespace
from unittest.mock import Mock

from opencda.core.application.behavior import BROADCAST_SERVICE_TYPE
from opencda.core.application.behavior.services.self_informer import SelfInformer
from opencda.core.application.behavior.types import Transform
from opencda.core.attack.adversary_framework import AttackStageRegistry, Status
from opencda.core.common.agent_manager import AgentManager
from opencda.core.sensing.localization import LocalizationSource, LocalizationState


class _Owner:
    def __init__(self, localization: LocalizationState) -> None:
        self.id = "node-1"
        self.agent = SimpleNamespace(localizer=Mock())
        self.agent.localizer.get_state.return_value = localization


def test_process_publishes_localizer_state() -> None:
    localization = LocalizationState(
        transform=Transform(),
        speed_kmh=12.5,
        source=LocalizationSource.GT,
    )
    owner = _Owner(localization)
    service = SelfInformer()
    service.on_attach(owner)

    (message,) = service.process(())

    assert message.src_owner_id == owner.id
    assert message.dst_owner_id == owner.id
    assert message.dst_service_type == BROADCAST_SERVICE_TYPE
    assert message.payload.localization is localization
    assert service.get_state().localization is localization
    owner.agent.localizer.get_state.assert_called_once_with()


def test_suppression_returns_an_empty_batch_and_deactivation_restores_publication() -> None:
    localization = LocalizationState(transform=Transform(), speed_kmh=12.5, source=LocalizationSource.GT)
    owner = _Owner(localization)
    service = SelfInformer()
    service.on_attach(owner)
    stage = AttackStageRegistry.create_stage("dropper", capabilities=("request.submit",), params={"drop_rate": 1.0})

    try:
        assert stage.execute((service,)).status == Status.SUCCESS
        assert service.process(()) == ()
    finally:
        stage.deactivate()

    (message,) = service.process(())
    assert message.payload.localization is localization
    assert message.dst_owner_id == owner.id


def test_suppressed_publication_does_not_break_agent_manager_delivery() -> None:
    localization = LocalizationState(transform=Transform(), speed_kmh=12.5, source=LocalizationSource.GT)
    agent = SimpleNamespace(use_carla_autopilot=False, localizer=Mock())
    agent.localizer.get_state.return_value = localization
    service = SelfInformer()
    owner = AgentManager(agent, "node-1", behavior_services=(service,))
    stage = AttackStageRegistry.create_stage("dropper", capabilities=("request.submit",), params={"drop_rate": 1.0})

    try:
        assert stage.execute((service,)).status == Status.SUCCESS
        messages, states = owner.update_behavior_services(())
    finally:
        stage.deactivate()

    assert messages == []
    assert states[service.service_type].localization is localization
