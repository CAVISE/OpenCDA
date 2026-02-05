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


def _make_mock_world():
    """Create a mocked CARLA world with stable settings mocks."""
    settings1 = SimpleNamespace()
    settings2 = SimpleNamespace()

    world = Mock()
    world.get_settings.side_effect = [settings1, settings2]
    world.apply_settings = Mock()
    world.set_weather = Mock()
    world.get_map.return_value = Mock()
    world.tick = Mock()

    return world


def _make_mock_client(world):
    """Create a mocked CARLA client bound to the provided world."""
    client = Mock()
    client.set_timeout = Mock()
    client.get_world.return_value = world
    return client


def _make_scenario_manager(mocker, scenario_params=None):
    from opencda.scenario_testing.utils.sim_api import ScenarioManager

    scenario_params = scenario_params or _minimal_scenario_params()

    world = _make_mock_world()
    client = _make_mock_client(world)

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

    world = _make_mock_world()
    client = _make_mock_client(world)

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

    # Future-proof: if ScenarioManager.__init__ ever ticks, the test should still only assert tick() from sm.tick().
    world.tick.reset_mock()

    sm.tick()
    world.tick.assert_called_once_with()


def _make_single_cav_scenario_params(minimal_vehicle_config, *, cav_id=7, spawn_position=None, destination=None):
    """Build minimal scenario_params for a single CAV spawn."""
    spawn_position = spawn_position or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    destination = destination or [1.0, 2.0, 3.0]

    params = _minimal_scenario_params()
    params["vehicle_base"] = minimal_vehicle_config
    params["scenario"] = {
        "single_cav_list": [
            {
                "id": cav_id,
                "spawn_position": spawn_position,
                "destination": destination,
            }
        ]
    }
    return params


def _setup_spawn_custom_actor(mocker):
    """Common setup for spawn_custom_actor tests."""
    sm, world, _ = _make_scenario_manager(mocker)

    bp_lib = Mock(spec_set=["find"])
    blueprint = Mock(spec_set=["id", "set_attribute"])
    blueprint.id = "vehicle.mock"
    blueprint.set_attribute = Mock()

    bp_lib.find.return_value = blueprint
    world.get_blueprint_library.return_value = bp_lib
    world.spawn_actor.return_value = "ACTOR"

    return sm, world, bp_lib, blueprint


def test_spawn_custom_actor_with_model(mocker):
    """spawn_custom_actor uses model from config."""
    sm, world, bp_lib, blueprint = _setup_spawn_custom_actor(mocker)

    spawn_transform = Mock()
    config = {"model": "vehicle.audi.a2"}

    out = sm.spawn_custom_actor(spawn_transform, config, fallback_model="vehicle.lincoln.mkz_2017")

    bp_lib.find.assert_called_once_with("vehicle.audi.a2")
    blueprint.set_attribute.assert_not_called()
    world.spawn_actor.assert_called_once_with(blueprint, spawn_transform)
    assert out == "ACTOR"


def test_spawn_custom_actor_with_color(mocker):
    """spawn_custom_actor sets color attribute if provided."""
    sm, world, bp_lib, blueprint = _setup_spawn_custom_actor(mocker)

    spawn_transform = Mock()
    config = {"model": "vehicle.audi.a2", "color": [255, 0, 0]}

    out = sm.spawn_custom_actor(spawn_transform, config, fallback_model="vehicle.lincoln.mkz_2017")

    assert out == "ACTOR"
    bp_lib.find.assert_called_once_with("vehicle.audi.a2")
    blueprint.set_attribute.assert_called_once_with("color", "255,0,0")
    world.spawn_actor.assert_called_once_with(blueprint, spawn_transform)


def test_spawn_custom_actor_fallback_model(mocker):
    """spawn_custom_actor uses fallback_model if no model in config."""
    sm, world, bp_lib, blueprint = _setup_spawn_custom_actor(mocker)

    spawn_transform = Mock()
    config = {}

    out = sm.spawn_custom_actor(spawn_transform, config, fallback_model="vehicle.lincoln.mkz_2017")

    assert out == "ACTOR"
    bp_lib.find.assert_called_once_with("vehicle.lincoln.mkz_2017")
    blueprint.set_attribute.assert_not_called()
    world.spawn_actor.assert_called_once_with(blueprint, spawn_transform)


def test_spawn_custom_actor_color_none_does_not_set_attribute(mocker):
    """spawn_custom_actor ignores color when config contains color=None."""
    sm, world, bp_lib, blueprint = _setup_spawn_custom_actor(mocker)

    spawn_transform = Mock()
    config = {"model": "vehicle.audi.a2", "color": None}

    out = sm.spawn_custom_actor(spawn_transform, config, fallback_model="vehicle.lincoln.mkz_2017")

    assert out == "ACTOR"
    bp_lib.find.assert_called_once_with("vehicle.audi.a2")
    blueprint.set_attribute.assert_not_called()
    world.spawn_actor.assert_called_once_with(blueprint, spawn_transform)


def test_spawn_custom_actor_color_invalid_type_raises(mocker):
    """spawn_custom_actor raises TypeError if color is not iterable (strict contract)."""
    sm, world, bp_lib, blueprint = _setup_spawn_custom_actor(mocker)

    spawn_transform = Mock()
    config = {"model": "vehicle.audi.a2", "color": 255}

    with pytest.raises(TypeError):
        sm.spawn_custom_actor(spawn_transform, config, fallback_model="vehicle.lincoln.mkz_2017")

    bp_lib.find.assert_called_once_with("vehicle.audi.a2")
    blueprint.set_attribute.assert_not_called()
    world.spawn_actor.assert_not_called()


def test_spawn_custom_actor_color_not_supported(mocker, caplog):
    """spawn_custom_actor logs warning if color attribute not supported and still spawns actor."""
    import logging

    sm, world, bp_lib, blueprint = _setup_spawn_custom_actor(mocker)

    blueprint.id = "vehicle.micro.microlino"
    blueprint.set_attribute.side_effect = IndexError("color not supported")
    bp_lib.find.return_value = blueprint

    spawn_transform = Mock()
    config = {"model": "vehicle.micro.microlino", "color": [255, 0, 0]}

    with caplog.at_level(logging.WARNING, logger="cavise.sim_api"):
        result = sm.spawn_custom_actor(spawn_transform, config, fallback_model="vehicle.lincoln.mkz_2017")

    assert result == "ACTOR"
    bp_lib.find.assert_called_once_with("vehicle.micro.microlino")
    blueprint.set_attribute.assert_called_once_with("color", "255,0,0")
    world.spawn_actor.assert_called_once_with(blueprint, spawn_transform)

    matching = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "cavise.sim_api" and blueprint.id in r.getMessage() and "color" in r.getMessage().lower()
    ]
    assert matching, f"Expected a WARNING from cavise.sim_api mentioning {blueprint.id!r} and color; got:\n{caplog.text}"


def test_close_restores_settings(mocker):
    """close() restores original world settings."""
    sm, world, _ = _make_scenario_manager(mocker)

    # ScenarioManager.__init__ calls apply_settings(new_settings) once.
    # We only want to assert what close() does.
    world.apply_settings.reset_mock()

    sm.close()

    world.apply_settings.assert_called_once_with(sm.origin_settings)


def test_create_vehicle_manager_single_cav(mocker, minimal_vehicle_config):
    """create_vehicle_manager creates one CAV when config has one entry."""
    from test import mocked_carla as carla

    params = _make_single_cav_scenario_params(minimal_vehicle_config, cav_id=7)

    sm, world, _ = _make_scenario_manager(mocker, params)

    world.tick.reset_mock()

    vehicle_actor = Mock(spec_set=["id", "get_location"])
    vehicle_actor.id = 123
    vehicle_actor.get_location.return_value = "start_loc"

    spawn_custom_actor = mocker.patch.object(sm, "spawn_custom_actor", return_value=vehicle_actor)

    vm_mock = Mock()
    vm_mock.vid = "cav-7"
    vm_mock.vehicle = vehicle_actor
    vm_mock.v2x_manager = Mock()
    vm_mock.v2x_manager.set_platoon = Mock()
    vm_mock.update_info = Mock()
    vm_mock.set_destination = Mock()

    vehicle_manager_ctor = mocker.patch("opencda.scenario_testing.utils.sim_api.VehicleManager", return_value=vm_mock)

    cav_list, cav_carla_list = sm.create_vehicle_manager(application=["single"], map_helper=None, data_dump=False)

    assert cav_list == [vm_mock]
    assert cav_carla_list == {123: "cav-7"}

    spawn_custom_actor.assert_called_once()
    spawn_args = spawn_custom_actor.call_args.args
    assert isinstance(spawn_args[0], carla.Transform)
    assert isinstance(spawn_args[0].location, carla.Location)
    assert isinstance(spawn_args[0].rotation, carla.Rotation)

    vehicle_manager_ctor.assert_called_once()
    _, ctor_cfg, ctor_app, ctor_map, ctor_world = vehicle_manager_ctor.call_args.args
    ctor_kwargs = vehicle_manager_ctor.call_args.kwargs

    assert ctor_app == ["single"]
    assert ctor_map is sm.carla_map
    assert ctor_world is sm.cav_world
    assert ctor_cfg["id"] == 7
    assert ctor_kwargs["prefix"] == "cav"
    assert ctor_kwargs["data_dumping"] is False
    assert ctor_kwargs["current_time"] == "t0"

    vm_mock.v2x_manager.set_platoon.assert_called_once_with(None)
    vm_mock.update_info.assert_called_once_with()
    vm_mock.set_destination.assert_called_once()

    args = vm_mock.set_destination.call_args.args
    kwargs = vm_mock.set_destination.call_args.kwargs

    assert args[0] == "start_loc"
    assert isinstance(args[1], carla.Location)
    assert args[1].x == pytest.approx(1.0)
    assert args[1].y == pytest.approx(2.0)
    assert args[1].z == pytest.approx(3.0)
    assert kwargs["clean"] is True

    assert world.tick.call_count == 1
