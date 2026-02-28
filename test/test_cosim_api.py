"""Unit tests for opencda.scenario_testing.utils.cosim_api.CoScenarioManager.

We avoid running CoScenarioManager.__init__ (it requires FS and SUMO).
Instead we instantiate via __new__ and set only required fields for each method.

Covers:
- traffic_light_ids
- get_traffic_light_state
- spawn_actor success/failure
- synchronize_vehicle
- destroy_actor
- close() cleanup logic

Note:
We use test.mocked_carla classes to build transforms/locations.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test import mocked_carla as carla


def _make_cosim_manager_without_init():
    from opencda.scenario_testing.utils.cosim_api import CoScenarioManager

    mgr = CoScenarioManager.__new__(CoScenarioManager)
    mgr._tls = {}
    mgr.client = Mock()
    mgr.world = Mock()
    mgr.origin_settings = Mock()
    mgr.sumo = Mock()
    mgr.sumo2carla_ids = {}
    mgr.carla2sumo_ids = {}
    return mgr


def test_traffic_light_ids():
    mgr = _make_cosim_manager_without_init()
    mgr._tls = {"a": Mock(), "b": Mock()}

    assert mgr.traffic_light_ids == {"a", "b"}


def test_get_traffic_light_state_exists():
    mgr = _make_cosim_manager_without_init()
    mgr._tls = {"id1": SimpleNamespace(state="GREEN")}

    assert mgr.get_traffic_light_state("id1") == "GREEN"


def test_get_traffic_light_state_missing():
    mgr = _make_cosim_manager_without_init()
    mgr._tls = {"id1": SimpleNamespace(state="GREEN")}

    assert mgr.get_traffic_light_state("missing") is None


def test_spawn_actor_success():
    from opencda.co_simulation.sumo_integration.constants import SPAWN_OFFSET_Z

    mgr = _make_cosim_manager_without_init()

    blueprint = Mock(spec_set=[])
    in_transform = carla.Transform(carla.Location(1.0, 2.0, 3.0), carla.Rotation(0.0, 0.0, 0.0))

    def _apply_batch_sync(batch, do_tick):
        assert do_tick is False
        assert len(batch) == 1
        assert batch[0].transform.location.z == pytest.approx(3.0 + SPAWN_OFFSET_Z)
        return [SimpleNamespace(error=None, actor_id=123)]

    mgr.client.apply_batch_sync.side_effect = _apply_batch_sync

    actor_id = mgr.spawn_actor(blueprint, in_transform)
    assert actor_id == 123


def test_spawn_actor_failure_returns_invalid_id():
    from opencda.co_simulation.sumo_integration.constants import INVALID_ACTOR_ID

    mgr = _make_cosim_manager_without_init()

    blueprint = Mock(spec_set=[])
    in_transform = carla.Transform(carla.Location(0.0, 0.0, 0.0), carla.Rotation(0.0, 0.0, 0.0))

    mgr.client.apply_batch_sync.return_value = [SimpleNamespace(error="boom", actor_id=999)]

    actor_id = mgr.spawn_actor(blueprint, in_transform)
    assert actor_id == INVALID_ACTOR_ID


def test_synchronize_vehicle_exists():
    mgr = _make_cosim_manager_without_init()

    vehicle = Mock()
    mgr.world.get_actor.return_value = vehicle

    tr = carla.Transform(carla.Location(1.0, 1.0, 1.0), carla.Rotation(0.0, 0.0, 0.0))
    assert mgr.synchronize_vehicle(vehicle_id=10, transform=tr) is True
    vehicle.set_transform.assert_called_once_with(tr)


def test_synchronize_vehicle_missing_returns_false():
    mgr = _make_cosim_manager_without_init()
    mgr.world.get_actor.return_value = None

    tr = carla.Transform(carla.Location(1.0, 1.0, 1.0), carla.Rotation(0.0, 0.0, 0.0))
    assert mgr.synchronize_vehicle(vehicle_id=10, transform=tr) is False


def test_destroy_actor_exists():
    mgr = _make_cosim_manager_without_init()

    actor = Mock()
    actor.destroy.return_value = True
    mgr.world.get_actor.return_value = actor

    assert mgr.destroy_actor(actor_id=10) is True
    actor.destroy.assert_called_once_with()


def test_destroy_actor_missing_returns_false():
    mgr = _make_cosim_manager_without_init()
    mgr.world.get_actor.return_value = None

    assert mgr.destroy_actor(actor_id=10) is False


def test_close_cleans_up():
    from opencda.scenario_testing.utils.cosim_api import CoScenarioManager

    mgr = _make_cosim_manager_without_init()

    mgr.world.apply_settings = Mock()
    mgr.sumo2carla_ids = {"sumo-1": 10, "sumo-2": 20}
    mgr.carla2sumo_ids = {10: "sumo-1", 30: "sumo-30"}

    tl = Mock()
    tl.type_id = "traffic.traffic_light"
    other = Mock()
    other.type_id = "vehicle.other"
    mgr.world.get_actors.return_value = [tl, other]

    mgr.destroy_actor = Mock(return_value=True)

    CoScenarioManager.close(mgr)

    mgr.world.apply_settings.assert_called_once_with(mgr.origin_settings)
    mgr.destroy_actor.assert_any_call(10)
    mgr.destroy_actor.assert_any_call(20)

    mgr.sumo.destroy_actor.assert_any_call("sumo-1")
    mgr.sumo.destroy_actor.assert_any_call("sumo-30")

    tl.freeze.assert_called_once_with(False)
    mgr.sumo.close.assert_called_once_with()


def _make_scenario_params_for_cosim(*, sync_mode: bool = True) -> dict:
    """Create minimal scenario params required to initialize CoScenarioManager."""
    return {
        "current_time": "t0",
        "world": {
            "client_port": 2000,
            "sync_mode": sync_mode,
            "fixed_delta_seconds": 0.05,
            "weather": {
                "sun_altitude_angle": 10.0,
                "cloudiness": 20.0,
                "precipitation": 0.0,
                "precipitation_deposits": 0.0,
                "wind_intensity": 5.0,
                "fog_density": 0.0,
                "fog_distance": 0.0,
                "fog_falloff": 0.0,
                "wetness": 0.0,
            },
        },
        "sumo": {
            "port": 8813,
            "host": "localhost",
            "gui": False,
            "client_order": 1,
            "step_length": 0.05,
        },
    }


def _make_world_and_client_for_cosim_init():
    """Build strict CARLA world/client mocks suitable for CoScenarioManager.__init__."""
    origin_settings = SimpleNamespace()
    new_settings = SimpleNamespace()

    carla_map = Mock(spec_set=["get_all_landmarks_of_type"])
    landmark_ok = SimpleNamespace(id="L1")
    landmark_empty = SimpleNamespace(id="")
    carla_map.get_all_landmarks_of_type.return_value = [landmark_ok, landmark_empty]

    blueprint_library = Mock(spec_set=[], name="blueprint_library")

    world = Mock(spec_set=["get_settings", "apply_settings", "set_weather", "get_map", "get_blueprint_library", "get_traffic_light"])
    world.get_settings.side_effect = [origin_settings, new_settings]
    world.get_map.return_value = carla_map
    world.get_blueprint_library.return_value = blueprint_library

    traffic_light = Mock(name="traffic_light")
    world.get_traffic_light.return_value = traffic_light

    client = Mock(spec_set=["set_timeout", "get_world"])
    client.get_world.return_value = world

    return world, client, carla_map, blueprint_library, traffic_light, origin_settings, new_settings


def test_coscenario_manager_init_builds_sumo_and_tls_and_bridge_helper(mocker, tmp_path):
    """__init__ smoke: builds SUMO simulation, extracts TLS mapping, and initializes BridgeHelper fields."""
    from opencda.scenario_testing.utils.cosim_api import CoScenarioManager

    # Create fake SUMO config file: <dir>/<basename>.sumocfg
    sumo_dir = tmp_path / "scenario_a"
    sumo_dir.mkdir()
    sumo_cfg = sumo_dir / f"{sumo_dir.name}.sumocfg"
    sumo_cfg.write_text("<configuration/>", encoding="utf-8")

    scenario_params = _make_scenario_params_for_cosim(sync_mode=True)

    world, client, _, blueprint_library, traffic_light, origin_settings, new_settings = _make_world_and_client_for_cosim_init()
    mocker.patch("opencda.scenario_testing.utils.sim_api.carla.Client", return_value=client)

    # Mock SUMO sim
    sumo_instance = Mock(spec_set=["switch_off_traffic_lights", "get_net_offset"])
    sumo_instance.get_net_offset.return_value = (123.0, 456.0)
    sumo_ctor = mocker.patch("opencda.scenario_testing.utils.cosim_api.SumoSimulation", return_value=sumo_instance)

    # Patch BridgeHelper to an isolated object so we can assert assignments without leaking state.
    bridge_helper = SimpleNamespace(blueprint_library=None, offset=None)
    mocker.patch("opencda.scenario_testing.utils.cosim_api.BridgeHelper", bridge_helper)

    node_ids = {"platoon": {}, "cav": {}, "rsu": {}}

    mgr = CoScenarioManager(
        scenario_params=scenario_params,
        apply_ml=False,
        carla_version="0.9.15",
        node_ids=node_ids,
        sumo_file_parent_path=str(sumo_dir),
        town=None,
        xodr_path=None,
        cav_world=Mock(),
        carla_host="carla",
    )

    # ScenarioManager part
    assert mgr.world is world
    assert mgr.origin_settings is origin_settings
    world.apply_settings.assert_called_once_with(new_settings)
    assert getattr(new_settings, "synchronous_mode") is True
    assert getattr(new_settings, "fixed_delta_seconds") == pytest.approx(0.05)

    # TLS mapping part
    assert mgr._tls == {"L1": traffic_light}
    assert mgr.traffic_light_ids == {"L1"}

    # SUMO construction part
    sumo_ctor.assert_called_once_with(
        str(sumo_cfg),
        scenario_params["sumo"]["step_length"],
        scenario_params["sumo"]["host"],
        scenario_params["sumo"]["port"],
        scenario_params["sumo"]["gui"],
        scenario_params["sumo"]["client_order"],
    )
    sumo_instance.switch_off_traffic_lights.assert_called_once_with()

    # BridgeHelper assignments
    assert bridge_helper.blueprint_library is blueprint_library
    assert bridge_helper.offset == (123.0, 456.0)

    # Basic mappings initialized
    assert mgr.sumo2carla_ids == {}
    assert mgr.carla2sumo_ids == {}
    assert mgr.node_ids == node_ids


def test_coscenario_manager_init_missing_sumocfg_raises_assertion(mocker, tmp_path):
    """__init__ should assert if <dir>/<basename>.sumocfg does not exist."""
    from opencda.scenario_testing.utils.cosim_api import CoScenarioManager

    sumo_dir = tmp_path / "scenario_missing"
    sumo_dir.mkdir()

    scenario_params = _make_scenario_params_for_cosim(sync_mode=True)
    world, client, *_ = _make_world_and_client_for_cosim_init()
    mocker.patch("opencda.scenario_testing.utils.sim_api.carla.Client", return_value=client)

    node_ids = {"platoon": {}, "cav": {}, "rsu": {}}

    with pytest.raises(AssertionError, match=r"does not exist"):
        CoScenarioManager(
            scenario_params=scenario_params,
            apply_ml=False,
            carla_version="0.9.15",
            node_ids=node_ids,
            sumo_file_parent_path=str(sumo_dir),
            town=None,
            xodr_path=None,
            cav_world=Mock(),
            carla_host="carla",
        )


def _setup_tick_base(mgr):
    """Configure minimal required attributes for CoScenarioManager.tick()."""
    mgr._active_actors = set()
    mgr.spawned_actors = set()
    mgr.destroyed_actors = set()

    mgr.node_ids = {"platoon": {}, "cav": {}, "rsu": {}}

    mgr.cav_world = Mock(spec_set=["update_sumo_vehicles"])
    mgr.cav_world.update_sumo_vehicles = Mock()

    mgr.sumo.tick = Mock()
    mgr.sumo.subscribe = Mock()
    mgr.sumo.unsubscribe = Mock()
    mgr.sumo.get_actor = Mock()
    mgr.sumo.spawn_actor = Mock()
    mgr.sumo.destroy_actor = Mock()
    mgr.sumo.synchronize_vehicle = Mock()
    mgr.sumo.synchronize_traffic_light = Mock()

    mgr.sumo.spawned_actors = set()
    mgr.sumo.destroyed_actors = set()
    mgr.sumo.traffic_light_ids = set()

    mgr.world.tick = Mock()
    mgr.world.get_actors.return_value = []
    mgr.world.get_actor = Mock(return_value=None)

    mgr._tls = {}


def _make_vehicle_or_static_actor(actor_id: int, type_id: str, *, color: str | None, transform=None):
    """Create a strict CARLA actor mock suitable for CARLA->SUMO branch."""
    actor = Mock(spec_set=["id", "type_id", "attributes", "bounding_box", "get_transform"])
    actor.id = actor_id
    actor.type_id = type_id
    actor.attributes = {} if color is None else {"color": color}
    actor.bounding_box = SimpleNamespace(extent=carla.Vector3D(1.0, 1.0, 1.0))
    actor.get_transform.return_value = transform or carla.Transform(carla.Location(0.0, 0.0, 0.0), carla.Rotation(0.0, 0.0, 0.0))
    return actor


def _make_simple_actor(actor_id: int, type_id: str):
    """Create a strict CARLA actor mock with only id/type_id (e.g., walkers)."""
    actor = Mock(spec_set=["id", "type_id"])
    actor.id = actor_id
    actor.type_id = type_id
    return actor


def _install_world_actors(mgr, actors):
    """Configure world.get_actors and world.get_actor to return a stable set of actors."""
    actors = list(actors)
    mgr.world.get_actors.return_value = actors

    by_id = {}
    for a in actors:
        by_id[getattr(a, "id")] = a

    mgr.world.get_actor.side_effect = lambda actor_id: by_id.get(actor_id)


def _make_sumo_actor(*, transform_name="sumo_transform", extent_name="sumo_extent"):
    """Create a strict SUMO actor mock used by SUMO->CARLA branch (only transform+extent are expected)."""
    actor = Mock(spec_set=["transform", "extent"])
    actor.transform = Mock(name=transform_name)
    actor.extent = Mock(name=extent_name)
    return actor


def _assert_update_sumo_vehicles_last_call_matches_mapping(mgr):
    """Assert cav_world.update_sumo_vehicles was called and last call matches current mapping.

    This is future-proof against refactors that may call update_sumo_vehicles multiple times per tick.
    """
    assert mgr.cav_world.update_sumo_vehicles.call_count >= 1
    last_mapping = mgr.cav_world.update_sumo_vehicles.call_args_list[-1].args[0]
    assert last_mapping == mgr.sumo2carla_ids


def test_tick_spawns_sumo_actor_in_carla_updates_mapping_and_cav_world(mocker):
    """tick(): when SUMO spawns an actor, it is spawned in CARLA and mapping is updated."""
    from opencda.scenario_testing.utils import cosim_api

    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    mgr.sumo.spawned_actors = {"sumo-1"}
    mgr.carla2sumo_ids = {}

    mgr.sumo.get_actor.return_value = _make_sumo_actor()

    blueprint = Mock(spec_set=[], name="carla_blueprint")
    carla_transform = carla.Transform(carla.Location(1.0, 2.0, 3.0), carla.Rotation(0.0, 0.0, 0.0))

    mocker.patch.object(cosim_api.BridgeHelper, "get_carla_blueprint", return_value=blueprint, create=True)
    mocker.patch.object(cosim_api.BridgeHelper, "get_carla_transform", return_value=carla_transform, create=True)

    mgr.spawn_actor = Mock(return_value=123)
    mgr.synchronize_vehicle = Mock(return_value=True)
    mgr.destroy_actor = Mock(return_value=True)

    mgr.tick()

    mgr.sumo.tick.assert_called_once_with()
    mgr.sumo.subscribe.assert_called_once_with("sumo-1")
    mgr.spawn_actor.assert_called_once_with(blueprint, carla_transform)

    # Mapping created; also synchronized (update loop runs after spawn)
    assert mgr.sumo2carla_ids == {"sumo-1": 123}
    mgr.synchronize_vehicle.assert_called_once_with(123, carla_transform)

    mgr.world.tick.assert_called_once_with()
    _assert_update_sumo_vehicles_last_call_matches_mapping(mgr)


def test_tick_destroys_sumo_actor_from_carla_removes_mapping(mocker):
    """tick(): when SUMO destroys an actor, CARLA actor is destroyed and mapping removed."""
    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    mgr.sumo2carla_ids = {"sumo-9": 321}
    mgr.sumo.destroyed_actors = {"sumo-9"}
    mgr.destroy_actor = Mock(return_value=True)

    mgr.tick()

    mgr.destroy_actor.assert_called_once_with(321)
    assert mgr.sumo2carla_ids == {}
    _assert_update_sumo_vehicles_last_call_matches_mapping(mgr)


def test_tick_updates_sumo_actor_transform_in_carla(mocker):
    """tick(): existing SUMO-controlled actor is synchronized in CARLA via synchronize_vehicle()."""
    from opencda.scenario_testing.utils import cosim_api

    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    mgr.sumo2carla_ids = {"sumo-1": 111}

    mgr.sumo.get_actor.return_value = _make_sumo_actor()

    carla_transform = carla.Transform(carla.Location(9.0, 8.0, 7.0), carla.Rotation(0.0, 0.0, 0.0))
    mocker.patch.object(cosim_api.BridgeHelper, "get_carla_transform", return_value=carla_transform, create=True)

    mgr.synchronize_vehicle = Mock(return_value=True)
    mgr.spawn_actor = Mock()
    mgr.destroy_actor = Mock()

    mgr.tick()

    mgr.synchronize_vehicle.assert_called_once_with(111, carla_transform)
    mgr.spawn_actor.assert_not_called()
    mgr.destroy_actor.assert_not_called()
    _assert_update_sumo_vehicles_last_call_matches_mapping(mgr)


def test_tick_sumo_spawn_with_no_blueprint_unsubscribes(mocker):
    """tick(): if get_carla_blueprint returns None, SUMO actor is unsubscribed and not spawned."""
    from opencda.scenario_testing.utils import cosim_api

    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    mgr.sumo.spawned_actors = {"sumo-1"}
    mgr.carla2sumo_ids = {}

    mgr.sumo.get_actor.return_value = _make_sumo_actor()

    mocker.patch.object(cosim_api.BridgeHelper, "get_carla_blueprint", return_value=None, create=True)

    mgr.spawn_actor = Mock()
    mgr.synchronize_vehicle = Mock(return_value=True)
    mgr.destroy_actor = Mock(return_value=True)

    mgr.tick()

    mgr.sumo.subscribe.assert_called_once_with("sumo-1")
    mgr.sumo.unsubscribe.assert_called_once_with("sumo-1")
    mgr.spawn_actor.assert_not_called()
    # No accidental downstream effects when blueprint is missing.
    mgr.synchronize_vehicle.assert_not_called()
    mgr.destroy_actor.assert_not_called()
    assert mgr.sumo2carla_ids == {}
    _assert_update_sumo_vehicles_last_call_matches_mapping(mgr)


def test_tick_spawns_carla_actor_in_sumo_and_subscribes_and_syncs(mocker):
    """tick(): when a new CARLA actor appears, it is spawned in SUMO, subscribed, and synchronized."""
    from opencda.scenario_testing.utils import cosim_api

    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    mgr.node_ids = {"platoon": {}, "cav": {10: "cav-10"}, "rsu": {}}

    carla_actor = _make_vehicle_or_static_actor(10, "vehicle.mock", color="1,2,3")
    _install_world_actors(mgr, [carla_actor])

    mocker.patch.object(cosim_api.BridgeHelper, "get_sumo_vtype", return_value="passenger", create=True)
    mocker.patch.object(cosim_api.BridgeHelper, "get_sumo_transform", return_value="SUMO_TR", create=True)

    mgr.sumo.spawn_actor.return_value = "sumo-10"
    mgr.sumo.get_actor.return_value = Mock(spec_set=[])

    mgr.tick()

    mgr.sumo.spawn_actor.assert_called_once_with("passenger", "cav-10", "1,2,3")
    mgr.sumo.subscribe.assert_called_once_with("sumo-10")
    assert mgr.carla2sumo_ids == {10: "sumo-10"}
    mgr.sumo.synchronize_vehicle.assert_called_once_with("sumo-10", "SUMO_TR", None)


def test_tick_destroys_carla_actor_in_sumo_removes_mapping(mocker):
    """tick(): if CARLA actor disappears, SUMO actor is destroyed and mapping removed."""
    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    mgr._active_actors = {10}
    mgr.world.get_actors.return_value = []

    mgr.carla2sumo_ids = {10: "sumo-10"}

    mgr.tick()

    mgr.sumo.destroy_actor.assert_called_once_with("sumo-10")
    assert mgr.carla2sumo_ids == {}


def test_tick_syncs_traffic_lights_common_landmarks(mocker):
    """tick(): synchronizes traffic lights for landmark IDs present in both SUMO and CARLA."""
    from opencda.scenario_testing.utils import cosim_api

    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    mgr._tls = {"tl-1": SimpleNamespace(state="GREEN")}
    mgr.sumo.traffic_light_ids = {"tl-1"}
    mgr.sumo.synchronize_traffic_light = Mock()

    mocker.patch.object(cosim_api.BridgeHelper, "get_sumo_traffic_light_state", return_value="G", create=True)

    mgr.tick()

    mgr.sumo.synchronize_traffic_light.assert_called_once_with("tl-1", "G")
    _assert_update_sumo_vehicles_last_call_matches_mapping(mgr)


def test_tick_ignores_destroyed_sumo_actor_not_in_mapping(mocker):
    """tick(): ignores SUMO destroyed actor IDs that are not present in sumo2carla_ids."""
    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    mgr.sumo.destroyed_actors = {"sumo-missing"}
    mgr.destroy_actor = Mock()

    mgr.tick()

    mgr.destroy_actor.assert_not_called()
    assert mgr.sumo2carla_ids == {}
    mgr.cav_world.update_sumo_vehicles.assert_called_once_with({})


@pytest.mark.parametrize("type_id", ["vehicle.mock", "static.prop.gnome"])
def test_tick_spawns_unknown_carla_actor_in_sumo_when_not_in_node_ids_vehicle_and_static(mocker, type_id):
    """tick(): unknown node_id fallback works for both vehicle.* and static.* actors."""
    from opencda.scenario_testing.utils import cosim_api

    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    mgr.sumo.spawned_actors = set()
    mgr.sumo.destroyed_actors = set()
    mgr.sumo2carla_ids = {}

    mgr.node_ids = {"platoon": {}, "cav": {}, "rsu": {}}

    unknown_id = 55
    unknown_color = "4,5,6"

    carla_actor = _make_vehicle_or_static_actor(unknown_id, type_id, color=unknown_color)
    _install_world_actors(mgr, [carla_actor])

    mocker.patch.object(cosim_api.BridgeHelper, "get_sumo_vtype", return_value="passenger", create=True)
    mocker.patch.object(cosim_api.BridgeHelper, "get_sumo_transform", return_value="SUMO_TR", create=True)

    mgr.sumo.spawn_actor = Mock(return_value=f"sumo-{unknown_id}")
    mgr.sumo.get_actor = Mock(return_value=Mock())
    mgr.sumo.subscribe = Mock()
    mgr.sumo.synchronize_vehicle = Mock()

    mgr.tick()

    mgr.sumo.spawn_actor.assert_called_once_with("passenger", f"unknown-{unknown_id}", unknown_color)
    mgr.sumo.subscribe.assert_called_once_with(f"sumo-{unknown_id}")
    assert mgr.carla2sumo_ids == {unknown_id: f"sumo-{unknown_id}"}
    mgr.sumo.synchronize_vehicle.assert_called_once_with(f"sumo-{unknown_id}", "SUMO_TR", None)


def test_tick_ignores_non_vehicle_and_non_static_actors_in_carla_to_sumo_branch(mocker):
    """tick(): actors that are not vehicle.* or static.* must be ignored (no SUMO spawn)."""
    from opencda.scenario_testing.utils import cosim_api

    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    # Disable SUMO->CARLA side effects
    mgr.sumo.spawned_actors = set()
    mgr.sumo.destroyed_actors = set()
    mgr.sumo2carla_ids = {}

    walker = _make_simple_actor(77, "walker.pedestrian.0001")
    _install_world_actors(mgr, [walker])

    get_sumo_vtype = mocker.patch.object(cosim_api.BridgeHelper, "get_sumo_vtype", return_value="passenger", create=True)
    mgr.sumo.spawn_actor = Mock()
    mgr.sumo.subscribe = Mock()

    mgr.tick()

    mgr.sumo.spawn_actor.assert_not_called()
    mgr.sumo.subscribe.assert_not_called()
    # This branch must not even ask for SUMO vtype (actor type is filtered out before).
    get_sumo_vtype.assert_not_called()
    mgr.sumo.synchronize_vehicle.assert_not_called()
    mgr.sumo.destroy_actor.assert_not_called()
    assert mgr.carla2sumo_ids == {}


def test_tick_does_not_spawn_in_sumo_when_get_sumo_vtype_is_none(mocker):
    """tick(): if BridgeHelper.get_sumo_vtype returns None, CARLA actor must not be spawned in SUMO."""
    from opencda.scenario_testing.utils import cosim_api

    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    # Disable SUMO->CARLA side effects
    mgr.sumo.spawned_actors = set()
    mgr.sumo.destroyed_actors = set()
    mgr.sumo2carla_ids = {}

    carla_actor = _make_vehicle_or_static_actor(88, "vehicle.mock", color=None)
    _install_world_actors(mgr, [carla_actor])

    mocker.patch.object(cosim_api.BridgeHelper, "get_sumo_vtype", return_value=None, create=True)
    mgr.sumo.spawn_actor = Mock()
    mgr.sumo.subscribe = Mock()

    mgr.tick()

    mgr.sumo.spawn_actor.assert_not_called()
    mgr.sumo.subscribe.assert_not_called()
    # If we never spawned an actor in SUMO, there is nothing to synchronize/destroy.
    mgr.sumo.synchronize_vehicle.assert_not_called()
    mgr.sumo.destroy_actor.assert_not_called()
    assert mgr.carla2sumo_ids == {}


def test_tick_syncs_traffic_lights_when_carla_state_is_none(mocker):
    """tick(): if CARLA traffic light state is None, it should still be passed through BridgeHelper and synchronized in SUMO."""
    from opencda.scenario_testing.utils import cosim_api

    mgr = _make_cosim_manager_without_init()
    _setup_tick_base(mgr)

    # Common landmark exists but has state None
    mgr._tls = {"tl-1": SimpleNamespace(state=None)}
    mgr.sumo.traffic_light_ids = {"tl-1"}
    mgr.sumo.synchronize_traffic_light = Mock()

    get_sumo_state = mocker.patch.object(cosim_api.BridgeHelper, "get_sumo_traffic_light_state", return_value="UNKNOWN", create=True)

    mgr.tick()

    get_sumo_state.assert_called_once_with(None)
    mgr.sumo.synchronize_traffic_light.assert_called_once_with("tl-1", "UNKNOWN")
