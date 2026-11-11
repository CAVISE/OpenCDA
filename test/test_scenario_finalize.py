"""Tests for scenario finalization and resource cleanup."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opencda.scenario_testing.scenario import Scenario


def _scenario() -> Scenario:
    scenario = Scenario.__new__(Scenario)
    scenario.single_cav_list = []
    scenario.rsu_list = []
    scenario.platoon_list = []
    scenario.bg_veh_list = []
    scenario.communication_manager = None
    scenario.coperception_model_manager = None
    scenario.scenario_metrics_collector = Mock()
    scenario.eval_manager = None
    scenario.scenario_manager = Mock()
    return scenario


def test_finalize_stops_collision_callbacks_before_evaluation_and_always_cleans_up() -> None:
    events: list[str] = []
    scenario = _scenario()
    manager = Mock(id="cav-1")
    manager.agent.stop_runtime_sensors.side_effect = lambda: events.append("stop")
    manager.destroy.side_effect = lambda: events.append("destroy")
    scenario.single_cav_list = [manager]
    scenario.eval_manager = Mock()

    def fail_evaluation(**_kwargs) -> None:
        events.append("evaluate")
        raise RuntimeError("report failed")

    scenario.eval_manager.evaluate.side_effect = fail_evaluation

    with pytest.raises(RuntimeError, match="report failed"):
        scenario.finalize(SimpleNamespace(record=False))

    assert events == ["stop", "evaluate", "destroy"]
    scenario.scenario_manager.close.assert_called_once_with()


def test_finalize_continues_cleanup_after_destroy_error() -> None:
    scenario = _scenario()
    first_manager = Mock(id="cav-1")
    first_manager.destroy.side_effect = RuntimeError("destroy failed")
    second_manager = Mock(id="cav-2")
    scenario.single_cav_list = [first_manager, second_manager]
    communication_manager = Mock()
    scenario.communication_manager = communication_manager

    with pytest.raises(RuntimeError, match="destroy failed"):
        scenario.finalize(SimpleNamespace(record=False))

    second_manager.destroy.assert_called_once_with()
    scenario.scenario_manager.close.assert_called_once_with()
    communication_manager.destroy.assert_called_once_with()
