"""Unit tests for opencda.core.common.vehicle_manager.VehicleManager.

Covers:
- ID parsing/autogeneration and duplicate handling (including autogen-on-duplicate)
- Invalid ID types (with and without autogen)
- Delegation and call chains in set_destination/update_info/run_step/destroy
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest


def _patch_vehicle_manager_deps(mocker):
    v2x = Mock()
    localizer = Mock()
    perception = Mock()
    map_manager = Mock()
    safety = Mock()
    agent = Mock()
    platoon_agent = Mock()
    controller = Mock()
    dumper = Mock()

    mocker.patch("opencda.core.common.vehicle_manager.V2XManager", return_value=v2x)
    mocker.patch("opencda.core.common.vehicle_manager.LocalizationManager", return_value=localizer)
    mocker.patch("opencda.core.common.vehicle_manager.PerceptionManager", return_value=perception)
    mocker.patch("opencda.core.common.vehicle_manager.MapManager", return_value=map_manager)
    mocker.patch("opencda.core.common.vehicle_manager.SafetyManager", return_value=safety)
    mocker.patch("opencda.core.common.vehicle_manager.BehaviorAgent", return_value=agent)
    mocker.patch("opencda.core.common.vehicle_manager.PlatooningBehaviorAgent", return_value=platoon_agent)
    mocker.patch("opencda.core.common.vehicle_manager.ControlManager", return_value=controller)
    mocker.patch("opencda.core.common.vehicle_manager.DataDumper", return_value=dumper)

    map_manager.static_bev = "static_bev"
    return {
        "v2x": v2x,
        "localizer": localizer,
        "perception": perception,
        "map_manager": map_manager,
        "safety": safety,
        "agent": agent,
        "platoon_agent": platoon_agent,
        "controller": controller,
        "dumper": dumper,
    }


def test_valid_id_from_config(mocker, minimal_vehicle_config, mock_cav_world):
    _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    cfg = {**minimal_vehicle_config, "id": 5}

    vm = VehicleManager(Mock(id=10), cfg, ["single"], Mock(), mock_cav_world, prefix="cav")
    assert vm.vid == "cav-5"
    mock_cav_world.update_vehicle_manager.assert_called_once_with(vm)


def test_duplicate_id_raises_when_autogen_disabled(mocker, minimal_vehicle_config, mock_cav_world):
    _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    cfg = {**minimal_vehicle_config, "id": 5}

    VehicleManager(Mock(id=10), cfg, ["single"], Mock(), mock_cav_world, prefix="cav")
    with pytest.raises(ValueError, match="Duplicate vehicle ID"):
        VehicleManager(Mock(id=11), cfg, ["single"], Mock(), mock_cav_world, prefix="cav", autogenerate_id_on_failure=False)


def test_duplicate_id_with_autogen_generates_new(mocker, minimal_vehicle_config, mock_cav_world):
    _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    cfg1 = {**minimal_vehicle_config, "id": 5}
    cfg2 = {**minimal_vehicle_config, "id": 5}

    vm1 = VehicleManager(Mock(id=10), cfg1, ["single"], Mock(), mock_cav_world, prefix="cav", autogenerate_id_on_failure=True)
    vm2 = VehicleManager(Mock(id=11), cfg2, ["single"], Mock(), mock_cav_world, prefix="cav", autogenerate_id_on_failure=True)

    assert vm1.vid == "cav-5"
    assert vm2.vid == "cav-1"
    assert vm1.vid != vm2.vid


def test_negative_id_with_autogen(mocker, minimal_vehicle_config, mock_cav_world):
    _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    cfg = {**minimal_vehicle_config, "id": -1}

    vm = VehicleManager(Mock(id=10), cfg, ["single"], Mock(), mock_cav_world, prefix="cav")
    assert vm.vid == "cav-1"


def test_negative_id_without_autogen(mocker, minimal_vehicle_config, mock_cav_world):
    _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    cfg = {**minimal_vehicle_config, "id": -1}

    with pytest.raises(ValueError):
        VehicleManager(Mock(id=10), cfg, ["single"], Mock(), mock_cav_world, prefix="cav", autogenerate_id_on_failure=False)


def test_invalid_id_type_with_autogen(mocker, minimal_vehicle_config, mock_cav_world):
    _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    cfg = {**minimal_vehicle_config, "id": "not_a_number"}

    vm = VehicleManager(Mock(id=10), cfg, ["single"], Mock(), mock_cav_world, prefix="cav", autogenerate_id_on_failure=True)
    assert vm.vid == "cav-1"


def test_invalid_id_type_without_autogen(mocker, minimal_vehicle_config, mock_cav_world):
    _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    cfg = {**minimal_vehicle_config, "id": "not_a_number"}

    with pytest.raises(ValueError):
        VehicleManager(Mock(id=10), cfg, ["single"], Mock(), mock_cav_world, prefix="cav", autogenerate_id_on_failure=False)


def test_missing_id_with_autogen(mocker, minimal_vehicle_config, mock_cav_world):
    _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vm = VehicleManager(Mock(id=10), minimal_vehicle_config, ["single"], Mock(), mock_cav_world, prefix="cav")
    assert vm.vid == "cav-1"


def test_missing_id_without_autogen(mocker, minimal_vehicle_config, mock_cav_world):
    _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    with pytest.raises(ValueError, match="No vehicle ID specified"):
        VehicleManager(Mock(id=10), minimal_vehicle_config, ["single"], Mock(), mock_cav_world, prefix="cav", autogenerate_id_on_failure=False)


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("cav", "cav-1"),
        ("platoon", "platoon-1"),
        ("unknown", "unknown-1"),
        ("garbage", "unknown-1"),
    ],
)
def test_prefix_autogen_behaviour(mocker, minimal_vehicle_config, mock_cav_world, prefix, expected):
    _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vm = VehicleManager(Mock(id=10), minimal_vehicle_config, ["single"], Mock(), mock_cav_world, prefix=prefix)
    assert vm.vid == expected


def test_set_destination_delegates(mocker, minimal_vehicle_config, mock_cav_world):
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vm = VehicleManager(Mock(id=10), minimal_vehicle_config, ["single"], Mock(), mock_cav_world, prefix="cav")
    vm.set_destination("start", "end", clean=True, end_reset=False)

    deps["agent"].set_destination.assert_called_once_with("start", "end", True, False)


def test_update_info_calls_chain(mocker, minimal_vehicle_config, mock_cav_world):
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vehicle = Mock(id=10)
    world = Mock()
    vehicle.get_world.return_value = world
    carla_map = Mock()

    vm = VehicleManager(vehicle, minimal_vehicle_config, ["single"], carla_map, mock_cav_world, prefix="cav")

    ego_pos = Mock()
    ego_spd = Mock()
    objects = [{"id": 1}]

    deps["localizer"].get_ego_pos.return_value = ego_pos
    deps["localizer"].get_ego_spd.return_value = ego_spd
    deps["perception"].detect.return_value = objects

    vm.update_info()

    deps["localizer"].localize.assert_called_once_with()
    deps["perception"].detect.assert_called_once_with(ego_pos)
    deps["map_manager"].update_information.assert_called_once_with(ego_pos)

    deps["safety"].update_info.assert_called_once()
    safety_input = deps["safety"].update_info.call_args.args[0]
    assert safety_input["ego_pos"] is ego_pos
    assert safety_input["ego_speed"] is ego_spd
    assert safety_input["objects"] == objects
    assert safety_input["carla_map"] is carla_map
    assert safety_input["world"] is world
    assert safety_input["static_bev"] == "static_bev"

    deps["v2x"].update_info.assert_called_once_with(ego_pos, ego_spd)
    deps["agent"].update_information.assert_called_once_with(ego_pos, ego_spd, objects)
    deps["controller"].update_info.assert_called_once_with(ego_pos, ego_spd)


def test_run_step_returns_control(mocker, minimal_vehicle_config, mock_cav_world):
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vm = VehicleManager(Mock(id=10), minimal_vehicle_config, ["single"], Mock(), mock_cav_world, prefix="cav")

    deps["agent"].run_step.return_value = (12.3, "target_pos")
    deps["controller"].run_step.return_value = "control"

    control = vm.run_step(target_speed=8.0)

    deps["map_manager"].run_step.assert_called_once_with()
    deps["agent"].run_step.assert_called_once_with(8.0)
    deps["controller"].run_step.assert_called_once_with(12.3, "target_pos")
    assert control == "control"


def test_run_step_with_data_dumper(mocker, minimal_vehicle_config, mock_cav_world):
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vm = VehicleManager(
        Mock(id=10),
        minimal_vehicle_config,
        ["single"],
        Mock(),
        mock_cav_world,
        prefix="cav",
        data_dumping=True,
        current_time="t0",
    )

    deps["agent"].run_step.return_value = (1.0, "p")
    deps["controller"].run_step.return_value = "ctrl"

    vm.run_step(target_speed=2.0)
    deps["dumper"].run_step.assert_called_once_with(deps["perception"], deps["localizer"], deps["agent"])


def test_destroy_calls_all(mocker, minimal_vehicle_config, mock_cav_world):
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vehicle = Mock(id=10)
    vm = VehicleManager(vehicle, minimal_vehicle_config, ["single"], Mock(), mock_cav_world, prefix="cav")

    vm.destroy()

    deps["perception"].destroy.assert_called_once_with()
    deps["localizer"].destroy.assert_called_once_with()
    vehicle.destroy.assert_called_once_with()
    deps["map_manager"].destroy.assert_called_once_with()
    deps["safety"].destroy.assert_called_once_with()


def test_platoon_application_uses_platooning_agent(mocker, minimal_vehicle_config, mock_cav_world):
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vm = VehicleManager(Mock(id=10), minimal_vehicle_config, ["platoon"], Mock(), mock_cav_world, prefix="platoon")
    assert vm.agent is deps["platoon_agent"]


def test_update_info_v2x_does_not_raise(mocker, minimal_vehicle_config, mock_cav_world):
    """update_info_v2x() is currently a no-op: should not raise and must not trigger side effects."""
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vm = VehicleManager(Mock(id=10), minimal_vehicle_config, ["single"], Mock(), mock_cav_world, prefix="cav")

    # Guard against future accidental behavior in update_info_v2x().
    for k in ("v2x", "localizer", "perception", "map_manager", "safety", "agent", "controller"):
        deps[k].reset_mock()

    assert vm.update_info_v2x() is None

    assert deps["v2x"].mock_calls == []
    assert deps["localizer"].mock_calls == []
    assert deps["perception"].mock_calls == []
    assert deps["map_manager"].mock_calls == []
    assert deps["safety"].mock_calls == []
    assert deps["agent"].mock_calls == []
    assert deps["controller"].mock_calls == []


def test_update_info_localizer_failure_propagates_and_stops_chain(mocker, minimal_vehicle_config, mock_cav_world):
    """If localizer.localize() fails, update_info() should propagate and not call downstream modules."""
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vehicle = Mock(id=10)
    vehicle.get_world.return_value = Mock()
    carla_map = Mock()
    vm = VehicleManager(vehicle, minimal_vehicle_config, ["single"], carla_map, mock_cav_world, prefix="cav")

    deps["localizer"].localize.side_effect = RuntimeError("localize failed")

    with pytest.raises(RuntimeError, match="localize failed"):
        vm.update_info()

    deps["perception"].detect.assert_not_called()
    deps["map_manager"].update_information.assert_not_called()
    deps["safety"].update_info.assert_not_called()
    deps["v2x"].update_info.assert_not_called()
    deps["agent"].update_information.assert_not_called()
    deps["controller"].update_info.assert_not_called()


def test_update_info_perception_failure_propagates_and_stops_chain(mocker, minimal_vehicle_config, mock_cav_world):
    """If perception.detect() fails, update_info() should propagate and not call later steps."""
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vehicle = Mock(id=10)
    vehicle.get_world.return_value = Mock()
    carla_map = Mock()
    vm = VehicleManager(vehicle, minimal_vehicle_config, ["single"], carla_map, mock_cav_world, prefix="cav")

    ego_pos = Mock()
    ego_spd = Mock()
    deps["localizer"].get_ego_pos.return_value = ego_pos
    deps["localizer"].get_ego_spd.return_value = ego_spd
    deps["perception"].detect.side_effect = RuntimeError("detect failed")

    with pytest.raises(RuntimeError, match="detect failed"):
        vm.update_info()

    deps["map_manager"].update_information.assert_not_called()
    deps["safety"].update_info.assert_not_called()
    deps["v2x"].update_info.assert_not_called()
    deps["agent"].update_information.assert_not_called()
    deps["controller"].update_info.assert_not_called()


def test_update_info_agent_failure_propagates_and_stops_before_controller(mocker, minimal_vehicle_config, mock_cav_world):
    """If agent.update_information() fails, update_info() should propagate and controller.update_info must not run."""
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vehicle = Mock(id=10)
    vehicle.get_world.return_value = Mock()
    carla_map = Mock()
    vm = VehicleManager(vehicle, minimal_vehicle_config, ["single"], carla_map, mock_cav_world, prefix="cav")

    ego_pos = Mock()
    ego_spd = Mock()
    objects = [{"id": 1}]

    deps["localizer"].get_ego_pos.return_value = ego_pos
    deps["localizer"].get_ego_spd.return_value = ego_spd
    deps["perception"].detect.return_value = objects
    deps["agent"].update_information.side_effect = RuntimeError("agent failed")

    with pytest.raises(RuntimeError, match="agent failed"):
        vm.update_info()

    deps["map_manager"].update_information.assert_called_once_with(ego_pos)
    deps["safety"].update_info.assert_called_once()
    deps["v2x"].update_info.assert_called_once_with(ego_pos, ego_spd)
    deps["controller"].update_info.assert_not_called()


def test_run_step_agent_failure_propagates_and_skips_controller(mocker, minimal_vehicle_config, mock_cav_world):
    """If agent.run_step() fails, run_step() should propagate and controller must not run."""
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vm = VehicleManager(Mock(id=10), minimal_vehicle_config, ["single"], Mock(), mock_cav_world, prefix="cav")
    deps["agent"].run_step.side_effect = RuntimeError("planner failed")

    with pytest.raises(RuntimeError, match="planner failed"):
        vm.run_step(target_speed=5.0)

    deps["map_manager"].run_step.assert_called_once_with()
    deps["controller"].run_step.assert_not_called()


def test_run_step_controller_failure_propagates_and_skips_data_dump(mocker, minimal_vehicle_config, mock_cav_world):
    """If controller.run_step() fails, run_step() should propagate and data dumper must not run."""
    deps = _patch_vehicle_manager_deps(mocker)
    from opencda.core.common.vehicle_manager import VehicleManager

    vm = VehicleManager(
        Mock(id=10),
        minimal_vehicle_config,
        ["single"],
        Mock(),
        mock_cav_world,
        prefix="cav",
        data_dumping=True,
        current_time="t0",
    )

    deps["agent"].run_step.return_value = (10.0, "target_pos")
    deps["controller"].run_step.side_effect = RuntimeError("control failed")

    deps["dumper"].reset_mock()
    with pytest.raises(RuntimeError, match="control failed"):
        vm.run_step(target_speed=5.0)

    deps["dumper"].run_step.assert_not_called()
