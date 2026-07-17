"""Tests for the universal simulation agent."""

from unittest.mock import Mock

from opencda.core.application.behavior.types import Transform
from opencda.core.common.agent import Agent
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
