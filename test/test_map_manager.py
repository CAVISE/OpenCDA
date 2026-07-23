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


def _config(*, activate: bool) -> dict[str, object]:
    return {
        "activate": activate,
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

    manager = map_manager_module.MapManager(vehicle, carla_map, _config(activate=False))

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

    manager = map_manager_module.MapManager(vehicle, carla_map, _config(activate=True))

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
        _config(activate=True),
        shared_map_data=shared_data,
    )

    assert manager.topology is shared_data.topology
    assert manager.lane_info is shared_data.lane_info
    build.assert_not_called()


def test_map_data_cache_builds_once_per_world_map_and_resolution(map_manager_module, mocker) -> None:
    from opencda.core.map.map_data import MapDataCache

    world = Mock()
    carla_map = Mock()
    shared_data = map_manager_module.SharedMapData.empty()
    build = mocker.patch.object(map_manager_module.SharedMapData, "build", return_value=shared_data)
    cache = MapDataCache()

    first = cache.get_or_build(world, carla_map, _config(activate=True))
    second = cache.get_or_build(world, carla_map, _config(activate=True))

    assert first is shared_data
    assert second is shared_data
    build.assert_called_once_with(world, carla_map, 0.1)


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
    manager.activate = True
    manager.rasterize_static = Mock()
    manager.rasterize_dynamic = Mock()

    manager.run_step()
    manager.run_step()

    assert manager.rasterize_static.call_count == 2
    assert manager.rasterize_dynamic.call_count == 2
