"""Tests for scenario finalization and resource cleanup."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

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


@pytest.mark.parametrize(("config", "timeout"), [({}, 1.0), ({"sensor_sync": {"timeout_seconds": 3.5}}, 3.5)])
def test_cop_visibility_uses_configured_sensor_timeout(tmp_path, monkeypatch, config, timeout) -> None:
    """Use one configured frame deadline for both CoP sensors and GT selection."""
    scenario = _scenario()
    scenario.payload_handler = None
    cav = SimpleNamespace(id="cav-1", agent=SimpleNamespace(perception_manager=SimpleNamespace(semantic_sync_timeout_seconds=None)))
    rsu = SimpleNamespace(id="rsu-1", agent=SimpleNamespace(perception_manager=SimpleNamespace(semantic_sync_timeout_seconds=None)))
    scenario.single_cav_list = [cav]
    scenario.rsu_list = [rsu]
    model_module = ModuleType("opencda.core.common.coperception_model_manager")
    model_module.CoperceptionModelManager = Mock(return_value=Mock())
    communication_module = ModuleType("opencood.communication")
    communication_module.CommunicationDataInterface = Mock()
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    monkeypatch.setitem(sys.modules, communication_module.__name__, communication_module)

    scenario._init_coperception(SimpleNamespace(model_dir=str(tmp_path), with_advcp=False), "test", OmegaConf.create({"coperception": config}))

    assert scenario.coperception_data_processor.sensor_sync_timeout_seconds == timeout
    assert cav.agent.perception_manager.semantic_sync_timeout_seconds == timeout
    assert rsu.agent.perception_manager.semantic_sync_timeout_seconds == timeout
