"""Focused tests for shared MapManager initialization and actor data."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module")
def map_manager_module():
    """Import the production module in place of the lightweight global test stub."""
    module_name = "opencda.core.map.map_manager"
    placeholder = sys.modules.get(module_name)
    sys.modules.pop(module_name, None)
    try:
        module = importlib.import_module(module_name)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if placeholder is not None:
            sys.modules[module_name] = placeholder


def _config(*, mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "visualize": False,
        "pixels_per_meter": 2,
        "raster_size": [224, 224],
        "lane_sample_resolution": 0.1,
    }


def test_inactive_map_manager_skips_static_map_preprocessing(map_manager_module) -> None:
    world = Mock()
    vehicle = Mock(id=17)
    vehicle.get_world.return_value = world
    carla_map = Mock()

    manager = map_manager_module.MapManager(vehicle, carla_map, _config(mode="disabled"))

    carla_map.get_topology.assert_not_called()
    world.get_actors.assert_not_called()
    assert manager.topology == ()
    assert manager.lane_info == {}
    assert manager.crosswalk_info == {}
    assert manager.traffic_light_info == {}
    assert manager.static_bev is None


def test_active_map_manager_still_builds_static_map_data(map_manager_module, mocker) -> None:
    world = Mock()
    vehicle = Mock(id=17)
    vehicle.get_world.return_value = world
    carla_map = Mock()
    shared_data = map_manager_module.SharedMapData.empty()
    build = mocker.patch.object(map_manager_module.SharedMapData, "build", return_value=shared_data)

    manager = map_manager_module.MapManager(vehicle, carla_map, _config(mode="full_bev"))

    assert manager.topology == ()
    build.assert_called_once_with(world, carla_map, 0.1)


def test_active_map_manager_uses_injected_shared_data(map_manager_module, mocker) -> None:
    vehicle = Mock(id=17)
    vehicle.get_world.return_value = Mock()
    carla_map = Mock()
    shared_data = map_manager_module.SharedMapData(
        topology=(Mock(),),
        lane_info={"lane-0": {}},
        crosswalk_info={},
        traffic_light_info={},
        bound_info={"lanes": {}, "crosswalks": {}},
    )
    build = mocker.patch.object(map_manager_module.SharedMapData, "build")

    manager = map_manager_module.MapManager(
        vehicle,
        carla_map,
        _config(mode="full_bev"),
        shared_map_data=shared_data,
    )

    assert manager.topology is shared_data.topology
    assert manager.lane_info is shared_data.lane_info
    build.assert_not_called()


def test_map_data_cache_builds_once_for_distinct_world_proxies_of_same_map(map_manager_module, mocker) -> None:
    from opencda.core.map.map_data import MapDataCache

    first_world_proxy = Mock()
    second_world_proxy = Mock()
    carla_map = Mock()
    shared_data = map_manager_module.SharedMapData.empty()
    build = mocker.patch.object(map_manager_module.SharedMapData, "build", return_value=shared_data)
    cache = MapDataCache()

    first = cache.get_or_build(first_world_proxy, carla_map, _config(mode="full_bev"))
    second = cache.get_or_build(second_world_proxy, carla_map, _config(mode="full_bev"))

    assert first is shared_data
    assert second is shared_data
    build.assert_called_once_with(first_world_proxy, carla_map, 0.1)


def test_map_data_cache_rebuilds_for_another_map(map_manager_module, mocker) -> None:
    from opencda.core.map.map_data import MapDataCache

    world = Mock()
    first_map = Mock()
    second_map = Mock()
    build = mocker.patch.object(
        map_manager_module.SharedMapData,
        "build",
        side_effect=[map_manager_module.SharedMapData.empty(), map_manager_module.SharedMapData.empty()],
    )
    cache = MapDataCache()

    cache.get_or_build(world, first_map, _config(mode="full_bev"))
    cache.get_or_build(world, second_map, _config(mode="full_bev"))

    assert build.call_count == 2


def test_load_agents_uses_world_frame_without_world_query(map_manager_module, mocker) -> None:
    actor = Mock()
    state = SimpleNamespace(actor_id=8, actor=actor, transform=Mock())
    world_frame = Mock()
    world_frame.nearby_vehicles.return_value = (state,)
    world_frame.shared_actor_value.side_effect = lambda _namespace, _actor_id, factory: factory()
    manager = map_manager_module.MapManager.__new__(map_manager_module.MapManager)
    manager.world = Mock()
    manager.center = SimpleNamespace(location=Mock())
    manager.raster_radius = 25.0
    manager._world_frame = world_frame
    actor_info = {"location": [0.0, 0.0, 0.0], "yaw": 0.0, "corners": []}
    build_actor_info = mocker.patch.object(manager, "_world_actor_info", return_value=actor_info)

    result = manager.load_agents_world()

    assert result == {8: actor_info}
    world_frame.nearby_vehicles.assert_called_once_with(manager.center.location, 25.0)
    world_frame.shared_actor_value.assert_called_once()
    build_actor_info.assert_called_once_with(state)
    manager.world.get_actors.assert_not_called()


def test_run_step_keeps_rebuilding_static_bev(map_manager_module) -> None:
    manager = map_manager_module.MapManager.__new__(map_manager_module.MapManager)
    manager.mode = map_manager_module.MapManagerMode.FULL_BEV
    manager.rasterize_static = Mock()
    manager.rasterize_dynamic = Mock()

    manager.run_step()
    manager.run_step()

    assert manager.rasterize_static.call_count == 2
    assert manager.rasterize_dynamic.call_count == 2


def test_offroad_only_mode_checks_driving_lane_without_preprocessing(map_manager_module, mocker) -> None:
    world = Mock()
    vehicle = Mock(id=17)
    vehicle.get_world.return_value = world
    carla_map = Mock()
    carla_map.get_waypoint.return_value = Mock()
    build = mocker.patch.object(map_manager_module.SharedMapData, "build")
    manager = map_manager_module.MapManager(vehicle, carla_map, _config(mode="offroad_only"))
    ego_pose = SimpleNamespace(location=Mock())

    manager.update_information(ego_pose)
    manager.rasterize_static = Mock()
    manager.rasterize_dynamic = Mock()
    manager.run_step()

    assert manager.on_road is True
    carla_map.get_waypoint.assert_called_once_with(
        ego_pose.location,
        project_to_road=False,
        lane_type=map_manager_module.carla.LaneType.Driving,
    )
    build.assert_not_called()
    manager.rasterize_static.assert_not_called()
    manager.rasterize_dynamic.assert_not_called()


def test_offroad_only_mode_reports_location_outside_driving_lane(map_manager_module) -> None:
    vehicle = Mock(id=17)
    vehicle.get_world.return_value = Mock()
    carla_map = Mock()
    carla_map.get_waypoint.return_value = None
    manager = map_manager_module.MapManager(vehicle, carla_map, _config(mode="offroad_only"))

    manager.update_information(SimpleNamespace(location=Mock()))

    assert manager.on_road is False


def test_disabled_mode_does_not_check_road_location(map_manager_module) -> None:
    vehicle = Mock(id=17)
    vehicle.get_world.return_value = Mock()
    carla_map = Mock()
    manager = map_manager_module.MapManager(vehicle, carla_map, _config(mode="disabled"))

    manager.update_information(SimpleNamespace(location=Mock()))

    assert manager.on_road is None
    carla_map.get_waypoint.assert_not_called()


def test_legacy_activate_flag_maps_to_previous_modes(map_manager_module) -> None:
    assert map_manager_module.resolve_map_manager_mode({"activate": True}) is map_manager_module.MapManagerMode.FULL_BEV
    assert map_manager_module.resolve_map_manager_mode({"activate": False}) is map_manager_module.MapManagerMode.DISABLED


def test_invalid_map_manager_mode_is_rejected(map_manager_module) -> None:
    with pytest.raises(ValueError, match="Unsupported MapManager mode"):
        map_manager_module.resolve_map_manager_mode({"mode": "unknown"})
