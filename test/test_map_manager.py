"""Focused tests for MapManager initialization."""

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
    assert manager.topology == []
    assert manager.lane_info == {}
    assert manager.crosswalk_info == {}
    assert manager.traffic_light_info == {}
    assert manager.static_bev is None


def test_active_map_manager_still_builds_static_map_data(map_manager_module, mocker) -> None:
    world = Mock()
    vehicle = Mock(id=17)
    vehicle.get_world.return_value = world
    waypoint = SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(z=0.0)))
    carla_map = Mock()
    carla_map.get_topology.return_value = [(waypoint, Mock())]
    generate_tl_info = mocker.patch.object(map_manager_module.MapManager, "generate_tl_info")
    generate_lane_cross_info = mocker.patch.object(map_manager_module.MapManager, "generate_lane_cross_info")

    manager = map_manager_module.MapManager(vehicle, carla_map, _config(activate=True))

    assert manager.topology == [waypoint]
    generate_tl_info.assert_called_once_with(world)
    generate_lane_cross_info.assert_called_once_with()
