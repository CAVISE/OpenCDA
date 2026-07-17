"""Tests for the universal simulation agent."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opencda.core.application.behavior.types import Transform
from opencda.core.common.agent import Agent, VehicleComponents
from opencda.core.common.tick_profiler import TickProfiler
from opencda.core.sensing.localization.types import LocalizationSource, LocalizationState


def test_update_passes_shared_world_frame_to_localization_and_perception() -> None:
    world_frame = Mock()
    localization_state = LocalizationState(
        transform=Transform(),
        speed_kmh=0.0,
        source=LocalizationSource.GT,
    )
    agent = Agent.__new__(Agent)
    agent.localizer = Mock()
    agent.localizer.update.return_value = localization_state
    agent.perception_manager = Mock()
    agent._vehicle_components = None

    agent.update(world_frame)

    agent.localizer.update.assert_called_once_with(world_frame)
    agent.perception_manager.detect.assert_called_once_with(localization_state.transform.to_carla(), world_frame)


def test_update_without_world_frame_uses_legacy_component_signatures() -> None:
    localization_state = LocalizationState(
        transform=Transform(),
        speed_kmh=0.0,
        source=LocalizationSource.GT,
    )
    agent = Agent.__new__(Agent)
    agent.localizer = Mock()
    agent.localizer.update.return_value = localization_state
    agent.perception_manager = Mock()
    agent._vehicle_components = None

    agent.update()

    agent.localizer.update.assert_called_once_with()
    agent.perception_manager.detect.assert_called_once_with(localization_state.transform.to_carla())


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_update_skips_behavior_for_carla_autopilot(timing_enabled: bool) -> None:
    ego_pos = Mock()
    localization_state = SimpleNamespace(
        transform=SimpleNamespace(to_carla=Mock(return_value=ego_pos)),
        speed_kmh=25.0,
    )
    localizer = Mock()
    localizer.update.return_value = localization_state
    perception_manager = Mock()
    perception_manager.detect.return_value = {"vehicles": [], "traffic_lights": []}
    map_manager = Mock(static_bev=None)
    safety_manager = Mock()
    behavior_agent = Mock()
    controller = Mock()
    actor = Mock()
    carla_map = Mock()

    agent = Agent.__new__(Agent)
    agent.actor = actor
    agent.carla_map = carla_map
    agent.localizer = localizer
    agent.perception_manager = perception_manager
    agent._vehicle_components = VehicleComponents(
        map_manager=map_manager,
        safety_manager=safety_manager,
        behavior_agent=behavior_agent,
        controller=controller,
        use_carla_autopilot=True,
        carla_autopilot_port=8000,
    )
    profiler = TickProfiler(enabled=True) if timing_enabled else None

    agent.update(profiler=profiler)

    behavior_agent.update_information.assert_not_called()
    map_manager.update_information.assert_called_once_with(ego_pos)
    safety_manager.update_info.assert_called_once()
    controller.update_info.assert_called_once_with(ego_pos, 25.0)


def test_update_keeps_behavior_for_opencda_control() -> None:
    ego_pos = Mock()
    localization_state = SimpleNamespace(
        transform=SimpleNamespace(to_carla=Mock(return_value=ego_pos)),
        speed_kmh=25.0,
    )
    localizer = Mock()
    localizer.update.return_value = localization_state
    perception_manager = Mock()
    objects = {"vehicles": [], "traffic_lights": []}
    perception_manager.detect.return_value = objects
    behavior_agent = Mock()

    agent = Agent.__new__(Agent)
    agent.actor = Mock()
    agent.carla_map = Mock()
    agent.localizer = localizer
    agent.perception_manager = perception_manager
    agent._vehicle_components = VehicleComponents(
        map_manager=Mock(static_bev=None),
        safety_manager=Mock(),
        behavior_agent=behavior_agent,
        controller=Mock(),
        use_carla_autopilot=False,
        carla_autopilot_port=8000,
    )

    agent.update()

    behavior_agent.update_information.assert_called_once_with(ego_pos, 25.0, objects)
