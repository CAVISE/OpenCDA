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

    blueprint = Mock()
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

    blueprint = Mock()
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
