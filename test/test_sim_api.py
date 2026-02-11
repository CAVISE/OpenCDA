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

    world = Mock(
        spec_set=[
            "get_settings",
            "apply_settings",
            "set_weather",
            "get_map",
            "tick",
            "get_blueprint_library",
            "spawn_actor",
            "try_spawn_actor",
        ]
    )
    world.get_settings.side_effect = [settings1, settings2]
    world.apply_settings = Mock()
    world.set_weather = Mock()
    world.get_map.return_value = Mock(spec_set=[])
    world.tick = Mock()
    world.get_blueprint_library = Mock()
    world.spawn_actor = Mock()
    world.try_spawn_actor = Mock()

    return world


def _make_mock_client(world):
    """Create a mocked CARLA client bound to the provided world."""
    client = Mock(spec_set=["set_timeout", "get_world", "load_world", "get_trafficmanager"])
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

    blueprint_library = Mock(spec_set=["find"])
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

    spawn_transform = Mock(spec_set=[])
    config = {"model": "vehicle.audi.a2"}

    out = sm.spawn_custom_actor(spawn_transform, config, fallback_model="vehicle.lincoln.mkz_2017")

    bp_lib.find.assert_called_once_with("vehicle.audi.a2")
    blueprint.set_attribute.assert_not_called()
    world.spawn_actor.assert_called_once_with(blueprint, spawn_transform)
    assert out == "ACTOR"


def test_spawn_custom_actor_with_color(mocker):
    """spawn_custom_actor sets color attribute if provided."""
    sm, world, bp_lib, blueprint = _setup_spawn_custom_actor(mocker)

    spawn_transform = Mock(spec_set=[])
    config = {"model": "vehicle.audi.a2", "color": [255, 0, 0]}

    out = sm.spawn_custom_actor(spawn_transform, config, fallback_model="vehicle.lincoln.mkz_2017")

    assert out == "ACTOR"
    bp_lib.find.assert_called_once_with("vehicle.audi.a2")
    blueprint.set_attribute.assert_called_once_with("color", "255,0,0")
    world.spawn_actor.assert_called_once_with(blueprint, spawn_transform)


def test_spawn_custom_actor_fallback_model(mocker):
    """spawn_custom_actor uses fallback_model if no model in config."""
    sm, world, bp_lib, blueprint = _setup_spawn_custom_actor(mocker)

    spawn_transform = Mock(spec_set=[])
    config = {}

    out = sm.spawn_custom_actor(spawn_transform, config, fallback_model="vehicle.lincoln.mkz_2017")

    assert out == "ACTOR"
    bp_lib.find.assert_called_once_with("vehicle.lincoln.mkz_2017")
    blueprint.set_attribute.assert_not_called()
    world.spawn_actor.assert_called_once_with(blueprint, spawn_transform)


def test_spawn_custom_actor_color_none_does_not_set_attribute(mocker):
    """spawn_custom_actor ignores color when config contains color=None."""
    sm, world, bp_lib, blueprint = _setup_spawn_custom_actor(mocker)

    spawn_transform = Mock(spec_set=[])
    config = {"model": "vehicle.audi.a2", "color": None}

    out = sm.spawn_custom_actor(spawn_transform, config, fallback_model="vehicle.lincoln.mkz_2017")

    assert out == "ACTOR"
    bp_lib.find.assert_called_once_with("vehicle.audi.a2")
    blueprint.set_attribute.assert_not_called()
    world.spawn_actor.assert_called_once_with(blueprint, spawn_transform)


def test_spawn_custom_actor_color_invalid_type_raises(mocker):
    """spawn_custom_actor raises TypeError if color is not iterable (strict contract)."""
    sm, world, bp_lib, blueprint = _setup_spawn_custom_actor(mocker)

    spawn_transform = Mock(spec_set=[])
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

    spawn_transform = Mock(spec_set=[])
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
    vm_mock.v2x_manager = Mock(spec_set=["set_platoon"])
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


def test_create_platoon_manager_creates_one_platoon_two_members(mocker, minimal_vehicle_config):
    """create_platoon_manager creates a platoon and assigns lead/member + returns carla-id mapping."""
    from test import mocked_carla as carla

    params = _minimal_scenario_params()
    params["vehicle_base"] = minimal_vehicle_config
    params["platoon_base"] = {}
    params["scenario"] = {
        "platoon_list": [
            {
                "destination": [10.0, 20.0, 0.0],
                "members": [
                    {"id": 1, "spawn_position": [0.0, 0.0, 0.0, 0.0, 90.0, 0.0]},
                    {"id": 2, "spawn_position": [5.0, 0.0, 0.0, 0.0, 90.0, 0.0]},
                ],
            }
        ]
    }

    sm, world, _ = _make_scenario_manager(mocker, params)
    world.tick.reset_mock()

    cav_world_instance = Mock()
    cav_world_ctor = mocker.patch("opencda.scenario_testing.utils.sim_api.CavWorld", return_value=cav_world_instance)

    platoon_manager = Mock(spec_set=["set_lead", "add_member", "set_destination", "update_member_order"])
    platoon_manager.set_lead = Mock()
    platoon_manager.add_member = Mock()
    platoon_manager.set_destination = Mock()
    platoon_manager.update_member_order = Mock()
    platoon_manager_ctor = mocker.patch("opencda.scenario_testing.utils.sim_api.PlatooningManager", return_value=platoon_manager)

    actor1 = Mock(spec_set=["id"])
    actor1.id = 101
    actor2 = Mock(spec_set=["id"])
    actor2.id = 102
    spawn_custom_actor = mocker.patch.object(sm, "spawn_custom_actor", side_effect=[actor1, actor2])

    vm1 = Mock()
    vm1.vid = "platoon-1"
    vm1.vehicle = actor1
    vm2 = Mock()
    vm2.vid = "platoon-2"
    vm2.vehicle = actor2
    vehicle_manager_ctor = mocker.patch("opencda.scenario_testing.utils.sim_api.VehicleManager", side_effect=[vm1, vm2])

    platoons, mapping = sm.create_platoon_manager(map_helper=None, data_dump=False)

    cav_world_ctor.assert_called_once_with(False)
    assert sm.cav_world is cav_world_instance

    assert platoons == [platoon_manager]
    assert mapping == {101: "platoon-1", 102: "platoon-2"}

    spawn_custom_actor.assert_called()
    assert spawn_custom_actor.call_count == 2

    platoon_manager_ctor.assert_called_once()
    assert platoon_manager_ctor.call_args.args[1] is cav_world_instance

    assert vehicle_manager_ctor.call_count == 2
    for call_ in vehicle_manager_ctor.call_args_list:
        assert call_.args[2] == ["platoon"]
        assert call_.kwargs["prefix"] == "platoon"
        assert call_.kwargs["data_dumping"] is False
        assert call_.kwargs["current_time"] == "t0"

    platoon_manager.set_lead.assert_called_once_with(vm1)
    platoon_manager.add_member.assert_called_once_with(vm2, leader=False)

    platoon_manager.set_destination.assert_called_once()
    destination = platoon_manager.set_destination.call_args.args[0]
    assert isinstance(destination, carla.Location)
    assert destination.x == pytest.approx(10.0)
    assert destination.y == pytest.approx(20.0)
    assert destination.z == pytest.approx(0.0)

    platoon_manager.update_member_order.assert_called_once_with()
    world.tick.assert_called_once_with()


def _setup_traffic_spawning_world(sm, world):
    """Common strict setup for background traffic spawning tests."""
    from test import mocked_carla as carla

    blueprint_library = Mock(spec_set=["find"])
    blueprint = Mock(spec_set=["set_attribute", "has_attribute", "get_attribute"])
    blueprint.set_attribute = Mock()
    blueprint.has_attribute.return_value = True
    attr = Mock(spec_set=["recommended_values"])
    attr.recommended_values = ["0,0,0"]
    blueprint.get_attribute.return_value = attr
    blueprint_library.find.return_value = blueprint

    world.get_blueprint_library.return_value = blueprint_library

    sm.use_multi_class_bp = False
    return carla, blueprint_library, blueprint


def test_spawn_vehicles_by_list_non_random_spawns_and_configures_tm(mocker):
    """spawn_vehicles_by_list spawns vehicles and configures TrafficManager when random=False."""

    sm, world, _ = _make_scenario_manager(mocker, _minimal_scenario_params())
    carla, blueprint_library, blueprint = _setup_traffic_spawning_world(sm, world)

    mocker.patch("opencda.scenario_testing.utils.sim_api.car_blueprint_filter", return_value=[Mock()])

    actor1 = Mock(spec_set=["set_autopilot"])
    actor2 = Mock(spec_set=["set_autopilot"])
    world.spawn_actor.side_effect = [actor1, actor2]

    tm = Mock(
        spec_set=[
            "vehicle_percentage_speed_difference",
            "auto_lane_change",
        ]
    )
    tm.vehicle_percentage_speed_difference = Mock()
    tm.auto_lane_change = Mock()

    traffic_config = {
        "random": False,
        "auto_lane_change": True,
        "vehicle_list": [
            {"spawn_position": [1.0, 2.0, 0.3, 0.0, 90.0, 0.0], "vehicle_speed_perc": 10},
            {"spawn_position": [3.0, 4.0, 0.3, 0.0, 90.0, 0.0]},
        ],
    }

    out = sm.spawn_vehicles_by_list(tm, traffic_config, bg_list=[])

    assert out == [actor1, actor2]
    assert world.spawn_actor.call_count == 2

    expected_t1 = carla.Transform(carla.Location(1.0, 2.0, 0.3), carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))
    expected_t2 = carla.Transform(carla.Location(3.0, 4.0, 0.3), carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))

    assert world.spawn_actor.call_args_list[0].args[0] is blueprint
    assert world.spawn_actor.call_args_list[0].args[1] == expected_t1
    assert world.spawn_actor.call_args_list[1].args[0] is blueprint
    assert world.spawn_actor.call_args_list[1].args[1] == expected_t2

    blueprint.set_attribute.assert_any_call("color", "0, 255, 0")
    assert blueprint.set_attribute.call_count == 2

    actor1.set_autopilot.assert_called_once_with(True, 8000)
    actor2.set_autopilot.assert_called_once_with(True, 8000)

    tm.vehicle_percentage_speed_difference.assert_called_once_with(actor1, 10)
    tm.auto_lane_change.assert_has_calls([call(actor1, True), call(actor2, True)], any_order=False)

    blueprint_library.find.assert_called()


def test_spawn_vehicle_by_range_non_random_spawns_one_and_configures_tm(mocker):
    """spawn_vehicle_by_range spawns vehicles based on range and configures TrafficManager."""

    sm, world, _ = _make_scenario_manager(mocker, _minimal_scenario_params())
    carla, _, blueprint = _setup_traffic_spawning_world(sm, world)

    mocker.patch("opencda.scenario_testing.utils.sim_api.car_blueprint_filter", return_value=[Mock()])
    mocker.patch("opencda.scenario_testing.utils.sim_api.shuffle", side_effect=lambda x: None)
    mocker.patch("opencda.scenario_testing.utils.sim_api.random.randint", return_value=0)

    waypoint_transform = carla.Transform(carla.Location(0.0, 0.0, 0.0), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
    sm.carla_map = Mock()
    sm.carla_map.get_waypoint.return_value = Mock(transform=waypoint_transform)

    actor = Mock(spec_set=["set_autopilot"])
    world.try_spawn_actor = Mock(return_value=actor)

    tm = Mock(
        spec_set=[
            "auto_lane_change",
            "ignore_lights_percentage",
            "ignore_signs_percentage",
            "ignore_vehicles_percentage",
            "ignore_walkers_percentage",
            "random_left_lanechange_percentage",
            "random_right_lanechange_percentage",
            "vehicle_percentage_speed_difference",
        ]
    )
    tm.auto_lane_change = Mock()
    tm.ignore_lights_percentage = Mock()
    tm.ignore_signs_percentage = Mock()
    tm.ignore_vehicles_percentage = Mock()
    tm.ignore_walkers_percentage = Mock()
    tm.random_left_lanechange_percentage = Mock()
    tm.random_right_lanechange_percentage = Mock()
    tm.vehicle_percentage_speed_difference = Mock()

    traffic_config = {
        "random": False,
        "auto_lane_change": False,
        "range": [[0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1]],
        "ignore_lights_percentage": 0,
        "ignore_signs_percentage": 0,
        "ignore_vehicles_percentage": 0,
        "ignore_walkers_percentage": 0,
        "random_left_lanechange_percentage": 0,
        "random_right_lanechange_percentage": 0,
        "global_speed_perc": 15,
    }

    out = sm.spawn_vehicle_by_range(tm, traffic_config, bg_list=[])

    assert out == [actor]
    world.try_spawn_actor.assert_called_once()
    blueprint.set_attribute.assert_called_once_with("color", "0, 255, 0")

    spawned_transform = world.try_spawn_actor.call_args.args[1]
    assert isinstance(spawned_transform, carla.Transform)
    assert spawned_transform.location.z == pytest.approx(0.3)

    actor.set_autopilot.assert_called_once_with(True, 8000)
    tm.auto_lane_change.assert_called_once_with(actor, False)

    tm.ignore_lights_percentage.assert_called_once_with(actor, 0)
    tm.ignore_signs_percentage.assert_called_once_with(actor, 0)
    tm.ignore_vehicles_percentage.assert_called_once_with(actor, 0)
    tm.ignore_walkers_percentage.assert_called_once_with(actor, 0)


def _make_color_blueprint(*, recommended_colors):
    """Create a strict mock blueprint that supports the color attribute."""
    blueprint = Mock(spec_set=["has_attribute", "get_attribute", "set_attribute"])
    blueprint.has_attribute.return_value = True
    attr = Mock(spec_set=["recommended_values"])
    attr.recommended_values = list(recommended_colors)
    blueprint.get_attribute.return_value = attr
    blueprint.set_attribute = Mock()
    return blueprint


def test_spawn_vehicles_by_list_random_selects_blueprint_and_sets_color(mocker):
    """spawn_vehicles_by_list with random=True selects a blueprint from filter list and sets a random color."""
    sm, world, _ = _make_scenario_manager(mocker, _minimal_scenario_params())
    _setup_traffic_spawning_world(sm, world)

    bp_random = _make_color_blueprint(recommended_colors=["1,2,3"])
    mocker.patch("opencda.scenario_testing.utils.sim_api.car_blueprint_filter", return_value=[bp_random])
    mocker.patch("opencda.scenario_testing.utils.sim_api.random.choice", side_effect=[bp_random, "1,2,3"])

    actor = Mock(spec_set=["set_autopilot"])
    world.spawn_actor.return_value = actor

    tm = Mock(spec_set=["vehicle_percentage_speed_difference", "auto_lane_change"])
    tm.vehicle_percentage_speed_difference = Mock()
    tm.auto_lane_change = Mock()

    traffic_config = {
        "random": True,
        "auto_lane_change": False,
        "vehicle_list": [{"spawn_position": [1.0, 2.0, 0.3, 0.0, 90.0, 0.0], "vehicle_speed_perc": 5}],
    }

    out = sm.spawn_vehicles_by_list(tm, traffic_config, bg_list=[])

    assert out == [actor]
    world.spawn_actor.assert_called_once()
    assert world.spawn_actor.call_args.args[0] is bp_random
    bp_random.has_attribute.assert_called_once_with("color")
    bp_random.set_attribute.assert_called_once_with("color", "1,2,3")
    actor.set_autopilot.assert_called_once_with(True, 8000)
    tm.vehicle_percentage_speed_difference.assert_called_once_with(actor, 5)
    tm.auto_lane_change.assert_called_once_with(actor, False)


def test_spawn_vehicles_by_list_random_multi_class_uses_label_sampling_and_filter(mocker):
    """spawn_vehicles_by_list with random=True and use_multi_class_bp=True uses np.random.choice  multi_class_vehicle_blueprint_filter."""
    sm, world, _ = _make_scenario_manager(mocker, _minimal_scenario_params())

    blueprint_library = Mock(spec_set=["find"])
    blueprint_library.find.return_value = Mock()
    world.get_blueprint_library.return_value = blueprint_library

    sm.use_multi_class_bp = True
    sm.bp_meta = {"vehicle.a": {"class": "sedan"}}
    sm.bp_class_sample_prob = {"sedan": 1.0}

    bp_random = _make_color_blueprint(recommended_colors=["7,7,7"])
    np_choice = mocker.patch("opencda.scenario_testing.utils.sim_api.np.random.choice", return_value="sedan")
    mclass = mocker.patch("opencda.scenario_testing.utils.sim_api.multi_class_vehicle_blueprint_filter", return_value=[bp_random])
    mocker.patch("opencda.scenario_testing.utils.sim_api.random.choice", side_effect=[bp_random, "7,7,7"])

    actor = Mock(spec_set=["set_autopilot"])
    world.spawn_actor.return_value = actor

    tm = Mock(spec_set=["vehicle_percentage_speed_difference", "auto_lane_change"])
    tm.vehicle_percentage_speed_difference = Mock()
    tm.auto_lane_change = Mock()

    traffic_config = {
        "random": True,
        "auto_lane_change": True,
        "vehicle_list": [{"spawn_position": [1.0, 2.0, 0.3, 0.0, 90.0, 0.0]}],
    }

    out = sm.spawn_vehicles_by_list(tm, traffic_config, bg_list=[])

    assert out == [actor]
    np_choice.assert_called_once()
    assert np_choice.call_args.kwargs["p"] == [1.0]
    mclass.assert_called_once_with("sedan", blueprint_library, sm.bp_meta)
    world.spawn_actor.assert_called_once()
    assert world.spawn_actor.call_args.args[0] is bp_random
    bp_random.set_attribute.assert_called_once_with("color", "7,7,7")
    tm.auto_lane_change.assert_called_once_with(actor, True)


def test_spawn_vehicle_by_range_random_selects_blueprint_and_sets_color(mocker):
    """spawn_vehicle_by_range with random=True selects a blueprint and sets a random color."""
    from test import mocked_carla as carla

    sm, world, _ = _make_scenario_manager(mocker, _minimal_scenario_params())
    _setup_traffic_spawning_world(sm, world)

    mocker.patch("opencda.scenario_testing.utils.sim_api.shuffle", side_effect=lambda x: None)
    mocker.patch("opencda.scenario_testing.utils.sim_api.random.randint", return_value=0)

    bp_random = _make_color_blueprint(recommended_colors=["9,9,9"])
    mocker.patch("opencda.scenario_testing.utils.sim_api.car_blueprint_filter", return_value=[bp_random])
    mocker.patch("opencda.scenario_testing.utils.sim_api.random.choice", side_effect=[bp_random, "9,9,9"])

    waypoint_transform = carla.Transform(carla.Location(0.0, 0.0, 0.0), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
    sm.carla_map = Mock()
    sm.carla_map.get_waypoint.return_value = Mock(transform=waypoint_transform)

    actor = Mock(spec_set=["set_autopilot"])
    world.try_spawn_actor = Mock(return_value=actor)

    tm = Mock(
        spec_set=[
            "auto_lane_change",
            "ignore_lights_percentage",
            "ignore_signs_percentage",
            "ignore_vehicles_percentage",
            "ignore_walkers_percentage",
            "random_left_lanechange_percentage",
            "random_right_lanechange_percentage",
            "vehicle_percentage_speed_difference",
        ]
    )
    tm.auto_lane_change = Mock()
    tm.ignore_lights_percentage = Mock()
    tm.ignore_signs_percentage = Mock()
    tm.ignore_vehicles_percentage = Mock()
    tm.ignore_walkers_percentage = Mock()
    tm.random_left_lanechange_percentage = Mock()
    tm.random_right_lanechange_percentage = Mock()
    tm.vehicle_percentage_speed_difference = Mock()

    traffic_config = {
        "random": True,
        "auto_lane_change": False,
        "range": [[0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1]],
        "ignore_lights_percentage": 0,
        "ignore_signs_percentage": 0,
        "ignore_vehicles_percentage": 0,
        "ignore_walkers_percentage": 0,
        "random_left_lanechange_percentage": 0,
        "random_right_lanechange_percentage": 0,
        "global_speed_perc": 15,
    }

    out = sm.spawn_vehicle_by_range(tm, traffic_config, bg_list=[])

    assert out == [actor]
    world.try_spawn_actor.assert_called_once()
    assert world.try_spawn_actor.call_args.args[0] is bp_random
    bp_random.set_attribute.assert_called_once_with("color", "9,9,9")
    tm.vehicle_percentage_speed_difference.assert_called_once_with(actor, 15)


def test_spawn_vehicle_by_range_random_multi_class_fallback_to_library_filter(mocker, caplog):
    """If multi_class_vehicle_blueprint_filter returns empty list, spawn_vehicle_by_range falls back to blueprint_library.filter()."""
    import logging
    from test import mocked_carla as carla

    sm, world, _ = _make_scenario_manager(mocker, _minimal_scenario_params())

    blueprint_library = Mock(spec_set=["find", "filter"])
    blueprint_library.find.return_value = Mock()
    world.get_blueprint_library.return_value = blueprint_library

    sm.use_multi_class_bp = True
    sm.bp_meta = {"vehicle.a": {"class": "sedan"}}
    sm.bp_class_sample_prob = {"sedan": 1.0}

    mocker.patch("opencda.scenario_testing.utils.sim_api.shuffle", side_effect=lambda x: None)
    mocker.patch("opencda.scenario_testing.utils.sim_api.random.randint", return_value=0)
    mocker.patch("opencda.scenario_testing.utils.sim_api.np.random.choice", return_value="sedan")
    mocker.patch("opencda.scenario_testing.utils.sim_api.multi_class_vehicle_blueprint_filter", return_value=[])

    bp_fallback = _make_color_blueprint(recommended_colors=["3,3,3"])
    blueprint_library.filter.return_value = [bp_fallback]
    mocker.patch("opencda.scenario_testing.utils.sim_api.random.choice", side_effect=[bp_fallback, "3,3,3"])

    waypoint_transform = carla.Transform(carla.Location(0.0, 0.0, 0.0), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
    sm.carla_map = Mock()
    sm.carla_map.get_waypoint.return_value = Mock(transform=waypoint_transform)

    actor = Mock(spec_set=["set_autopilot"])
    world.try_spawn_actor = Mock(return_value=actor)

    tm = Mock(
        spec_set=[
            "auto_lane_change",
            "ignore_lights_percentage",
            "ignore_signs_percentage",
            "ignore_vehicles_percentage",
            "ignore_walkers_percentage",
            "random_left_lanechange_percentage",
            "random_right_lanechange_percentage",
            "vehicle_percentage_speed_difference",
        ]
    )
    tm.auto_lane_change = Mock()
    tm.ignore_lights_percentage = Mock()
    tm.ignore_signs_percentage = Mock()
    tm.ignore_vehicles_percentage = Mock()
    tm.ignore_walkers_percentage = Mock()
    tm.random_left_lanechange_percentage = Mock()
    tm.random_right_lanechange_percentage = Mock()
    tm.vehicle_percentage_speed_difference = Mock()

    traffic_config = {
        "random": True,
        "auto_lane_change": False,
        "range": [[0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1]],
        "ignore_lights_percentage": 0,
        "ignore_signs_percentage": 0,
        "ignore_vehicles_percentage": 0,
        "ignore_walkers_percentage": 0,
        "random_left_lanechange_percentage": 0,
        "random_right_lanechange_percentage": 0,
        "global_speed_perc": 15,
    }

    with caplog.at_level(logging.WARNING, logger="cavise.sim_api"):
        out = sm.spawn_vehicle_by_range(tm, traffic_config, bg_list=[])

    assert out == [actor]
    blueprint_library.filter.assert_called_once_with("vehicle.*")
    assert world.try_spawn_actor.call_args.args[0] is bp_fallback
    assert "No blueprints found for label" in caplog.text


def test_create_platoon_manager_uses_map_helper_when_spawn_special_present(mocker, minimal_vehicle_config):
    """create_platoon_manager uses map_helper() when member has spawn_special."""
    from test import mocked_carla as carla

    params = _minimal_scenario_params()
    params["vehicle_base"] = minimal_vehicle_config
    params["platoon_base"] = {}
    params["scenario"] = {
        "platoon_list": [
            {
                "destination": [1.0, 1.0, 0.0],
                "members": [{"id": 1, "spawn_special": [42, 43, 44]}],
            }
        ]
    }

    sm, world, _ = _make_scenario_manager(mocker, params)
    world.tick.reset_mock()

    cav_world_instance = Mock()
    mocker.patch("opencda.scenario_testing.utils.sim_api.CavWorld", return_value=cav_world_instance)

    platoon_manager = Mock()
    platoon_manager.set_lead = Mock()
    platoon_manager.set_destination = Mock()
    platoon_manager.update_member_order = Mock()
    mocker.patch("opencda.scenario_testing.utils.sim_api.PlatooningManager", return_value=platoon_manager)

    expected_transform = carla.Transform(carla.Location(10.0, 20.0, 0.3), carla.Rotation(pitch=1.0, yaw=2.0, roll=3.0))
    map_helper = Mock(return_value=expected_transform)

    actor = Mock(spec_set=["id"])
    actor.id = 501
    spawn_custom_actor = mocker.patch.object(sm, "spawn_custom_actor", return_value=actor)

    vm = Mock()
    vm.vid = "platoon-1"
    vm.vehicle = actor
    mocker.patch("opencda.scenario_testing.utils.sim_api.VehicleManager", return_value=vm)

    platoons, mapping = sm.create_platoon_manager(map_helper=map_helper, data_dump=False)

    assert platoons == [platoon_manager]
    assert mapping == {501: "platoon-1"}
    map_helper.assert_called_once_with("0.9.15", 42, 43, 44)

    spawn_custom_actor.assert_called_once()
    assert spawn_custom_actor.call_args.args[0] == expected_transform
    platoon_manager.set_lead.assert_called_once_with(vm)
    world.tick.assert_called_once_with()


def test_create_platoon_manager_multiple_platoons_combines_mapping_and_ticks_each_platoon(mocker, minimal_vehicle_config):
    """create_platoon_manager creates 2 platoons and returns a combined carla-id mapping (ticks once per platoon)."""
    from test import mocked_carla as carla

    params = _minimal_scenario_params()
    params["vehicle_base"] = minimal_vehicle_config
    params["platoon_base"] = {}
    params["scenario"] = {
        "platoon_list": [
            {
                "destination": [10.0, 0.0, 0.0],
                "members": [
                    {"id": 1, "spawn_position": [0.0, 0.0, 0.0, 0.0, 90.0, 0.0]},
                    {"id": 2, "spawn_position": [5.0, 0.0, 0.0, 0.0, 90.0, 0.0]},
                ],
            },
            {
                "destination": [20.0, 0.0, 0.0],
                "members": [
                    {"id": 3, "spawn_position": [100.0, 0.0, 0.0, 0.0, 90.0, 0.0]},
                    {"id": 4, "spawn_position": [105.0, 0.0, 0.0, 0.0, 90.0, 0.0]},
                ],
            },
        ]
    }

    sm, world, _ = _make_scenario_manager(mocker, params)
    world.tick.reset_mock()

    cav_world_instance = Mock()
    cav_world_ctor = mocker.patch("opencda.scenario_testing.utils.sim_api.CavWorld", return_value=cav_world_instance)

    def _make_platoon_manager_mock():
        pm = Mock(spec_set=["set_lead", "add_member", "set_destination", "update_member_order"])
        pm.set_lead = Mock()
        pm.add_member = Mock()
        pm.set_destination = Mock()
        pm.update_member_order = Mock()
        return pm

    pm1 = _make_platoon_manager_mock()
    pm2 = _make_platoon_manager_mock()
    platoon_manager_ctor = mocker.patch("opencda.scenario_testing.utils.sim_api.PlatooningManager", side_effect=[pm1, pm2])

    actor1 = Mock(spec_set=["id"])
    actor1.id = 101
    actor2 = Mock(spec_set=["id"])
    actor2.id = 102
    actor3 = Mock(spec_set=["id"])
    actor3.id = 201
    actor4 = Mock(spec_set=["id"])
    actor4.id = 202
    spawn_custom_actor = mocker.patch.object(sm, "spawn_custom_actor", side_effect=[actor1, actor2, actor3, actor4])

    vm1 = Mock()
    vm1.vid = "platoon-1"
    vm1.vehicle = actor1
    vm2 = Mock()
    vm2.vid = "platoon-2"
    vm2.vehicle = actor2
    vm3 = Mock()
    vm3.vid = "platoon-3"
    vm3.vehicle = actor3
    vm4 = Mock()
    vm4.vid = "platoon-4"
    vm4.vehicle = actor4
    vehicle_manager_ctor = mocker.patch("opencda.scenario_testing.utils.sim_api.VehicleManager", side_effect=[vm1, vm2, vm3, vm4])

    platoons, mapping = sm.create_platoon_manager(map_helper=None, data_dump=False)

    cav_world_ctor.assert_called_once_with(False)
    assert sm.cav_world is cav_world_instance

    assert platoons == [pm1, pm2]
    assert mapping == {
        101: "platoon-1",
        102: "platoon-2",
        201: "platoon-3",
        202: "platoon-4",
    }

    assert spawn_custom_actor.call_count == 4
    assert vehicle_manager_ctor.call_count == 4
    for call_ in vehicle_manager_ctor.call_args_list:
        assert call_.args[2] == ["platoon"]
        assert call_.kwargs["prefix"] == "platoon"
        assert call_.kwargs["data_dumping"] is False
        assert call_.kwargs["current_time"] == "t0"

    assert platoon_manager_ctor.call_count == 2
    assert platoon_manager_ctor.call_args_list[0].args[1] is cav_world_instance
    assert platoon_manager_ctor.call_args_list[1].args[1] is cav_world_instance

    pm1.set_lead.assert_called_once_with(vm1)
    pm1.add_member.assert_called_once_with(vm2, leader=False)
    pm2.set_lead.assert_called_once_with(vm3)
    pm2.add_member.assert_called_once_with(vm4, leader=False)

    pm1.set_destination.assert_called_once()
    dest1 = pm1.set_destination.call_args.args[0]
    assert isinstance(dest1, carla.Location)
    assert dest1.x == pytest.approx(10.0)
    assert dest1.y == pytest.approx(0.0)
    assert dest1.z == pytest.approx(0.0)

    pm2.set_destination.assert_called_once()
    dest2 = pm2.set_destination.call_args.args[0]
    assert isinstance(dest2, carla.Location)
    assert dest2.x == pytest.approx(20.0)
    assert dest2.y == pytest.approx(0.0)
    assert dest2.z == pytest.approx(0.0)

    pm1.update_member_order.assert_called_once_with()
    pm2.update_member_order.assert_called_once_with()
    assert world.tick.call_count == 2
    world.tick.assert_has_calls([call(), call()], any_order=False)


def test_spawn_vehicles_by_list_random_no_color_attribute_does_not_set_color(mocker):
    """spawn_vehicles_by_list with random=True should not set color if blueprint has no color attribute."""
    sm, world, _ = _make_scenario_manager(mocker, _minimal_scenario_params())
    _setup_traffic_spawning_world(sm, world)

    bp_random = Mock(spec_set=["has_attribute", "set_attribute"])
    bp_random.has_attribute.return_value = False
    bp_random.set_attribute = Mock()

    mocker.patch("opencda.scenario_testing.utils.sim_api.car_blueprint_filter", return_value=[bp_random])
    mocker.patch("opencda.scenario_testing.utils.sim_api.random.choice", return_value=bp_random)

    actor = Mock(spec_set=["set_autopilot"])
    world.spawn_actor.return_value = actor

    tm = Mock(spec_set=["vehicle_percentage_speed_difference", "auto_lane_change"])
    tm.vehicle_percentage_speed_difference = Mock()
    tm.auto_lane_change = Mock()

    traffic_config = {
        "random": True,
        "auto_lane_change": True,
        "vehicle_list": [{"spawn_position": [1.0, 2.0, 0.3, 0.0, 90.0, 0.0]}],
    }

    out = sm.spawn_vehicles_by_list(tm, traffic_config, bg_list=[])

    assert out == [actor]
    world.spawn_actor.assert_called_once()
    assert world.spawn_actor.call_args.args[0] is bp_random

    bp_random.has_attribute.assert_called_once_with("color")
    bp_random.set_attribute.assert_not_called()
    actor.set_autopilot.assert_called_once_with(True, 8000)
    tm.auto_lane_change.assert_called_once_with(actor, True)


def test_spawn_vehicle_by_range_try_spawn_actor_none_skips_and_returns_empty(mocker):
    """spawn_vehicle_by_range should skip positions where try_spawn_actor returns None."""
    from test import mocked_carla as carla

    sm, world, _ = _make_scenario_manager(mocker, _minimal_scenario_params())
    _setup_traffic_spawning_world(sm, world)

    mocker.patch("opencda.scenario_testing.utils.sim_api.car_blueprint_filter", return_value=[Mock()])
    mocker.patch("opencda.scenario_testing.utils.sim_api.shuffle", side_effect=lambda x: None)
    mocker.patch("opencda.scenario_testing.utils.sim_api.random.randint", return_value=0)

    waypoint_transform = carla.Transform(carla.Location(0.0, 0.0, 0.0), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
    sm.carla_map = Mock()
    sm.carla_map.get_waypoint.return_value = Mock(transform=waypoint_transform)

    world.try_spawn_actor = Mock(return_value=None)

    tm = Mock(
        spec_set=[
            "auto_lane_change",
            "ignore_lights_percentage",
            "ignore_signs_percentage",
            "ignore_vehicles_percentage",
            "ignore_walkers_percentage",
            "random_left_lanechange_percentage",
            "random_right_lanechange_percentage",
            "vehicle_percentage_speed_difference",
        ]
    )
    tm.auto_lane_change = Mock()
    tm.ignore_lights_percentage = Mock()
    tm.ignore_signs_percentage = Mock()
    tm.ignore_vehicles_percentage = Mock()
    tm.ignore_walkers_percentage = Mock()
    tm.random_left_lanechange_percentage = Mock()
    tm.random_right_lanechange_percentage = Mock()
    tm.vehicle_percentage_speed_difference = Mock()

    traffic_config = {
        "random": False,
        "auto_lane_change": False,
        "range": [[0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1]],
        "ignore_lights_percentage": 0,
        "ignore_signs_percentage": 0,
        "ignore_vehicles_percentage": 0,
        "ignore_walkers_percentage": 0,
        "random_left_lanechange_percentage": 0,
        "random_right_lanechange_percentage": 0,
        "global_speed_perc": 15,
    }

    out = sm.spawn_vehicle_by_range(tm, traffic_config, bg_list=[])

    assert out == []
    world.try_spawn_actor.assert_called_once()
    tm.auto_lane_change.assert_not_called()
    tm.ignore_lights_percentage.assert_not_called()
    tm.ignore_signs_percentage.assert_not_called()
    tm.ignore_vehicles_percentage.assert_not_called()
    tm.ignore_walkers_percentage.assert_not_called()
    tm.random_left_lanechange_percentage.assert_not_called()
    tm.random_right_lanechange_percentage.assert_not_called()
    tm.vehicle_percentage_speed_difference.assert_not_called()


def test_spawn_vehicle_by_range_calls_lanechange_percentages_when_nonzero(mocker):
    """spawn_vehicle_by_range should call random lanechange percentage methods when configured."""
    from test import mocked_carla as carla

    sm, world, _ = _make_scenario_manager(mocker, _minimal_scenario_params())
    _setup_traffic_spawning_world(sm, world)

    mocker.patch("opencda.scenario_testing.utils.sim_api.car_blueprint_filter", return_value=[Mock()])
    mocker.patch("opencda.scenario_testing.utils.sim_api.shuffle", side_effect=lambda x: None)
    mocker.patch("opencda.scenario_testing.utils.sim_api.random.randint", return_value=0)

    waypoint_transform = carla.Transform(carla.Location(0.0, 0.0, 0.0), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
    sm.carla_map = Mock()
    sm.carla_map.get_waypoint.return_value = Mock(transform=waypoint_transform)

    actor = Mock(spec_set=["set_autopilot"])
    world.try_spawn_actor = Mock(return_value=actor)

    tm = Mock(
        spec_set=[
            "auto_lane_change",
            "ignore_lights_percentage",
            "ignore_signs_percentage",
            "ignore_vehicles_percentage",
            "ignore_walkers_percentage",
            "random_left_lanechange_percentage",
            "random_right_lanechange_percentage",
            "vehicle_percentage_speed_difference",
        ]
    )
    tm.auto_lane_change = Mock()
    tm.ignore_lights_percentage = Mock()
    tm.ignore_signs_percentage = Mock()
    tm.ignore_vehicles_percentage = Mock()
    tm.ignore_walkers_percentage = Mock()
    tm.random_left_lanechange_percentage = Mock()
    tm.random_right_lanechange_percentage = Mock()
    tm.vehicle_percentage_speed_difference = Mock()

    traffic_config = {
        "random": False,
        "auto_lane_change": True,
        "range": [[0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1]],
        "ignore_lights_percentage": 0,
        "ignore_signs_percentage": 0,
        "ignore_vehicles_percentage": 0,
        "ignore_walkers_percentage": 0,
        "random_left_lanechange_percentage": 5,
        "random_right_lanechange_percentage": 7,
        "global_speed_perc": 15,
    }

    out = sm.spawn_vehicle_by_range(tm, traffic_config, bg_list=[])

    assert out == [actor]
    tm.random_left_lanechange_percentage.assert_called_once_with(actor, 5)
    tm.random_right_lanechange_percentage.assert_called_once_with(actor, 7)
    tm.auto_lane_change.assert_called_once_with(actor, True)
    tm.vehicle_percentage_speed_difference.assert_called_once_with(actor, 15)


def test_create_platoon_manager_spawn_position_is_converted_to_transform(mocker, minimal_vehicle_config):
    """create_platoon_manager converts spawn_position to carla.Transform with correct roll/yaw/pitch mapping."""
    from test import mocked_carla as carla

    params = _minimal_scenario_params()
    params["vehicle_base"] = minimal_vehicle_config
    params["platoon_base"] = {}
    params["scenario"] = {
        "platoon_list": [
            {
                "destination": [1.0, 1.0, 0.0],
                "members": [
                    {"id": 1, "spawn_position": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
                    {"id": 2, "spawn_position": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]},
                ],
            }
        ]
    }

    sm, _, _ = _make_scenario_manager(mocker, params)

    mocker.patch("opencda.scenario_testing.utils.sim_api.CavWorld", return_value=Mock())
    pm = Mock(spec_set=["set_lead", "add_member", "set_destination", "update_member_order"])
    pm.set_lead = Mock()
    pm.add_member = Mock()
    pm.set_destination = Mock()
    pm.update_member_order = Mock()
    mocker.patch("opencda.scenario_testing.utils.sim_api.PlatooningManager", return_value=pm)

    actor1 = Mock(spec_set=["id"])
    actor1.id = 1001
    actor2 = Mock(spec_set=["id"])
    actor2.id = 1002
    spawn_custom_actor = mocker.patch.object(sm, "spawn_custom_actor", side_effect=[actor1, actor2])

    vm1 = Mock(vid="platoon-1", vehicle=actor1)
    vm2 = Mock(vid="platoon-2", vehicle=actor2)
    mocker.patch("opencda.scenario_testing.utils.sim_api.VehicleManager", side_effect=[vm1, vm2])

    sm.create_platoon_manager(map_helper=None, data_dump=False)

    expected_t1 = carla.Transform(carla.Location(1.0, 2.0, 3.0), carla.Rotation(pitch=6.0, yaw=5.0, roll=4.0))
    expected_t2 = carla.Transform(carla.Location(7.0, 8.0, 9.0), carla.Rotation(pitch=12.0, yaw=11.0, roll=10.0))

    assert spawn_custom_actor.call_args_list[0].args[0] == expected_t1
    assert spawn_custom_actor.call_args_list[1].args[0] == expected_t2


def test_create_rsu_manager_single_rsu(mocker, minimal_rsu_config):
    """create_rsu_manager creates one RSU when scenario has one entry and returns correct carla-id mapping."""
    from test import mocked_carla as carla

    params = _minimal_scenario_params()
    params["rsu_base"] = minimal_rsu_config
    params["scenario"] = {"rsu_list": [{"id": 3, "spawn_position": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}]}

    sm, world, _ = _make_scenario_manager(mocker, params)

    bp_lib = Mock(spec_set=["find"])
    static_bp = Mock(spec_set=[])
    bp_lib.find.return_value = static_bp
    world.get_blueprint_library.return_value = bp_lib

    actor = Mock(spec_set=["id"])
    actor.id = 999
    world.spawn_actor.return_value = actor

    rsu_mgr = Mock()
    rsu_mgr.rid = "rsu-3"
    rsu_ctor = mocker.patch("opencda.scenario_testing.utils.sim_api.RSUManager", return_value=rsu_mgr)

    rsu_list, rsu_ids = sm.create_rsu_manager(data_dump=False)

    assert rsu_list == [rsu_mgr]
    assert rsu_ids == {999: "rsu-3"}

    bp_lib.find.assert_called_once_with("static.prop.gnome")

    expected_transform = carla.Transform(
        carla.Location(1.0, 2.0, 3.0),
        carla.Rotation(pitch=6.0, yaw=5.0, roll=4.0),
    )
    world.spawn_actor.assert_called_once_with(static_bp, expected_transform)

    rsu_ctor.assert_called_once()
    assert rsu_ctor.call_args.args[0] is world
    assert rsu_ctor.call_args.args[2] is sm.carla_map
    assert rsu_ctor.call_args.args[3] is sm.cav_world
    assert rsu_ctor.call_args.args[4] == "t0"
    assert rsu_ctor.call_args.args[5] is False


def test_create_traffic_carla_with_vehicle_list_uses_spawn_vehicles_by_list(mocker):
    """create_traffic_carla uses spawn_vehicles_by_list when vehicle_list is a list."""
    params = _minimal_scenario_params()
    params["carla_traffic_manager"] = {
        "global_distance": 2.0,
        "sync_mode": True,
        "set_osm_mode": False,
        "global_speed_perc": 10,
        "auto_lane_change": True,
        "random": False,
        "vehicle_list": [{"spawn_position": [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]}],
    }

    sm, world, client = _make_scenario_manager(mocker, params)

    tm = Mock(
        spec_set=[
            "set_global_distance_to_leading_vehicle",
            "set_synchronous_mode",
            "set_osm_mode",
            "global_percentage_speed_difference",
        ]
    )
    tm.set_global_distance_to_leading_vehicle = Mock()
    tm.set_synchronous_mode = Mock()
    tm.set_osm_mode = Mock()
    tm.global_percentage_speed_difference = Mock()
    client.get_trafficmanager.return_value = tm

    sm.spawn_vehicles_by_list = Mock(return_value=["V1"])
    sm.spawn_vehicle_by_range = Mock(return_value=["SHOULD_NOT_HAPPEN"])

    out_tm, bg_list = sm.create_traffic_carla()

    assert out_tm is tm
    assert bg_list == ["V1"]
    sm.spawn_vehicles_by_list.assert_called_once_with(tm, params["carla_traffic_manager"], [])
    sm.spawn_vehicle_by_range.assert_not_called()


def test_create_traffic_carla_with_range_uses_spawn_vehicle_by_range(mocker):
    """create_traffic_carla uses spawn_vehicle_by_range when vehicle_list is not a list (range-based spawning)."""
    params = _minimal_scenario_params()
    params["carla_traffic_manager"] = {
        "global_distance": 2.0,
        "sync_mode": True,
        "set_osm_mode": False,
        "global_speed_perc": 10,
        "auto_lane_change": True,
        "random": False,
        "vehicle_list": "RANGE_MODE",
        "range": [[0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1]],
        "ignore_lights_percentage": 0,
        "ignore_signs_percentage": 0,
        "ignore_vehicles_percentage": 0,
        "ignore_walkers_percentage": 0,
        "random_left_lanechange_percentage": 0,
        "random_right_lanechange_percentage": 0,
    }

    sm, world, client = _make_scenario_manager(mocker, params)

    tm = Mock(
        spec_set=[
            "set_global_distance_to_leading_vehicle",
            "set_synchronous_mode",
            "set_osm_mode",
            "global_percentage_speed_difference",
        ]
    )
    tm.set_global_distance_to_leading_vehicle = Mock()
    tm.set_synchronous_mode = Mock()
    tm.set_osm_mode = Mock()
    tm.global_percentage_speed_difference = Mock()
    client.get_trafficmanager.return_value = tm

    sm.spawn_vehicles_by_list = Mock(return_value=["SHOULD_NOT_HAPPEN"])
    sm.spawn_vehicle_by_range = Mock(return_value=["V2"])

    out_tm, bg_list = sm.create_traffic_carla()

    assert out_tm is tm
    assert bg_list == ["V2"]
    sm.spawn_vehicle_by_range.assert_called_once_with(tm, params["carla_traffic_manager"], [])
    sm.spawn_vehicles_by_list.assert_not_called()
