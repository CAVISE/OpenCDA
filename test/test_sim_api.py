"""Unit tests for opencda.scenario_testing.utils.sim_api.

Covers:
- car_blueprint_filter / multi_class_vehicle_blueprint_filter
- ScenarioManager.set_weather (assert WeatherParameters called correctly)
- ScenarioManager init exit path: sync_mode=False -> sys.exit
- Basic ScenarioManager methods that don't require real CARLA
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest


def _minimal_weather():
    return {
        "sun_altitude_angle": 10.0,
        "cloudiness": 20.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 5.0,
        "fog_density": 0.0,
        "fog_distance": 0.0,
        "fog_falloff": 0.0,
        "wetness": 0.0,
    }


def _minimal_scenario_params(sync_mode: bool = True):
    return {
        "current_time": "t0",
        "world": {
            "client_port": 2000,
            "sync_mode": sync_mode,
            "fixed_delta_seconds": 0.05,
            "weather": _minimal_weather(),
        },
    }


def _make_scenario_manager(mocker, scenario_params=None):
    from opencda.scenario_testing.utils.sim_api import ScenarioManager

    scenario_params = scenario_params or _minimal_scenario_params()

    settings1 = SimpleNamespace()
    settings2 = SimpleNamespace()

    world = Mock()
    world.get_settings.side_effect = [settings1, settings2]
    world.apply_settings = Mock()
    world.set_weather = Mock()
    world.get_map.return_value = Mock()
    world.tick = Mock()

    client = Mock()
    client.set_timeout = Mock()
    client.get_world.return_value = world

    mocker.patch("opencda.scenario_testing.utils.sim_api.carla.Client", return_value=client)

    sm = ScenarioManager(
        scenario_params,
        apply_ml=False,
        carla_version="0.9.15",
        town=None,
        xodr_path=None,
        cav_world=Mock(),
        carla_host="carla",
        carla_timeout=30.0,
    )
    return sm, world, client


@pytest.mark.parametrize("version", ["0.9.14", "0.9.15"])
def test_car_blueprint_filter_supported_versions(version):
    from opencda.scenario_testing.utils.sim_api import car_blueprint_filter

    blueprint_library = Mock()
    blueprint_library.find.side_effect = lambda name: f"bp:{name}"

    blueprints = car_blueprint_filter(blueprint_library, carla_version=version)

    assert isinstance(blueprints, list)
    assert len(blueprints) == 19
    assert blueprint_library.find.call_count == 19


def test_car_blueprint_filter_unsupported_exits():
    from opencda.scenario_testing.utils.sim_api import car_blueprint_filter

    with pytest.raises(SystemExit):
        car_blueprint_filter(Mock(), carla_version="0.9.13")


def test_multi_class_vehicle_blueprint_filter():
    from opencda.scenario_testing.utils.sim_api import multi_class_vehicle_blueprint_filter

    bp_meta = {
        "vehicle.a": {"class": "sedan"},
        "vehicle.b": {"class": "truck"},
        "vehicle.c": {"class": "sedan"},
    }

    blueprint_library = Mock()
    blueprint_library.find.side_effect = lambda name: f"bp:{name}"

    blueprints = multi_class_vehicle_blueprint_filter("sedan", blueprint_library, bp_meta)

    assert blueprints == ["bp:vehicle.a", "bp:vehicle.c"]
    assert blueprint_library.find.call_args_list == [call("vehicle.a"), call("vehicle.c")]


def test_set_weather_calls_weatherparameters(mocker):
    from opencda.scenario_testing.utils.sim_api import ScenarioManager

    wp = mocker.patch("opencda.scenario_testing.utils.sim_api.carla.WeatherParameters", return_value="WEATHER_OBJ")
    settings = _minimal_weather()

    out = ScenarioManager.set_weather(settings)
    assert out == "WEATHER_OBJ"

    wp.assert_called_once_with(
        sun_altitude_angle=10.0,
        cloudiness=20.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        fog_density=0.0,
        fog_distance=0.0,
        fog_falloff=0.0,
        wetness=0.0,
    )


def test_sync_mode_false_exits(mocker):
    from opencda.scenario_testing.utils.sim_api import ScenarioManager

    params = _minimal_scenario_params(sync_mode=False)

    settings1 = SimpleNamespace()
    settings2 = SimpleNamespace()
    world = Mock()
    world.get_settings.side_effect = [settings1, settings2]
    world.apply_settings = Mock()
    world.set_weather = Mock()
    world.get_map.return_value = Mock()

    client = Mock()
    client.set_timeout = Mock()
    client.get_world.return_value = world

    mocker.patch("opencda.scenario_testing.utils.sim_api.carla.Client", return_value=client)

    with pytest.raises(SystemExit, match="only supports sync simulation mode"):
        ScenarioManager(params, apply_ml=False, carla_version="0.9.15", town=None, xodr_path=None, cav_world=Mock(), carla_host="carla")


def test_create_vehicle_manager_empty(mocker):
    params = _minimal_scenario_params()
    params.pop("scenario", None)

    sm, _, _ = _make_scenario_manager(mocker, params)
    cav_list, cav_ids = sm.create_vehicle_manager(application=["single"], map_helper=None, data_dump=False)

    assert cav_list == []
    assert cav_ids == {}


def test_create_rsu_manager_empty(mocker):
    params = _minimal_scenario_params()
    params.pop("scenario", None)

    sm, _, _ = _make_scenario_manager(mocker, params)
    rsu_list, rsu_ids = sm.create_rsu_manager(data_dump=False)

    assert rsu_list == []
    assert rsu_ids == {}


def test_create_traffic_carla_none(mocker):
    sm, _, _ = _make_scenario_manager(mocker, _minimal_scenario_params())
    tm, bg_list = sm.create_traffic_carla()

    assert tm is None
    assert bg_list == []


def test_tick_calls_world_tick(mocker):
    sm, world, _ = _make_scenario_manager(mocker, _minimal_scenario_params())
    sm.tick()
    world.tick.assert_called_once_with()
