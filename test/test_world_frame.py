"""Tests for the shared per-tick CARLA world frame."""

from types import SimpleNamespace
from unittest.mock import Mock

import carla
import pytest

from opencda.core.common.world_frame import WorldFrame


def _actor(actor_id: int, type_id: str) -> Mock:
    actor = Mock()
    actor.id = actor_id
    actor.type_id = type_id
    return actor


def _actor_snapshot(actor_id: int, x: float, y: float, z: float = 0.0) -> Mock:
    snapshot = Mock()
    snapshot.id = actor_id
    snapshot.get_transform.return_value = carla.Transform(carla.Location(x=x, y=y, z=z))
    snapshot.get_velocity.return_value = carla.Vector3D(x=float(actor_id), y=0.0, z=0.0)
    return snapshot


def test_capture_reads_world_once_and_uses_actor_snapshots() -> None:
    vehicle = _actor(1, "vehicle.test")
    walker = _actor(2, "walker.pedestrian.test")
    rsu = _actor(3, "static.prop.gnome")
    sensor = _actor(4, "sensor.camera.rgb")
    traffic_light = _actor(5, "traffic.traffic_light")

    snapshots = {actor_id: _actor_snapshot(actor_id, x=float(actor_id), y=0.0) for actor_id in range(1, 6)}
    world_snapshot = Mock()
    world_snapshot.frame = 17
    world_snapshot.timestamp = SimpleNamespace(elapsed_seconds=2.5)
    world_snapshot.find.side_effect = snapshots.get
    world = Mock()
    world.get_snapshot.return_value = world_snapshot
    world.get_actors.return_value = [vehicle, walker, rsu, sensor, traffic_light]

    world_frame = WorldFrame.capture(world, frame=17)

    assert world_frame.frame == 17
    assert world_frame.timestamp == pytest.approx(2.5)
    assert world_frame.actor_state(1).transform is snapshots[1].get_transform.return_value
    assert world_frame.actor_state(3).actor is rsu
    assert world_frame.traffic_lights == (traffic_light,)
    with pytest.raises(KeyError, match="Actor 4"):
        world_frame.actor_state(4)
    world.get_snapshot.assert_called_once_with()
    world.get_actors.assert_called_once_with()
    vehicle.get_transform.assert_not_called()
    vehicle.get_velocity.assert_not_called()


def test_spatial_queries_filter_radius_type_and_actor_id() -> None:
    actors = {
        1: _actor(1, "vehicle.ego"),
        2: _actor(2, "vehicle.near"),
        3: _actor(3, "walker.pedestrian.near"),
        4: _actor(4, "vehicle.outside"),
    }
    states = {
        1: _actor_snapshot(1, 0.0, 0.0),
        2: _actor_snapshot(2, 3.0, 4.0),
        3: _actor_snapshot(3, -4.0, 0.0),
        4: _actor_snapshot(4, 50.0, 0.0),
    }
    world_snapshot = Mock(frame=9, timestamp=SimpleNamespace(elapsed_seconds=1.0))
    world_snapshot.find.side_effect = states.get
    world = Mock()
    world.get_snapshot.return_value = world_snapshot
    world.get_actors.return_value = list(actors.values())
    world_frame = WorldFrame.capture(world, cell_size=10.0)

    dynamic = world_frame.nearby_dynamic(carla.Location(), radius=6.0, exclude_actor_id=1)
    vehicles = world_frame.nearby_vehicles(carla.Location(), radius=6.0, exclude_actor_id=1)

    assert [state.actor_id for state in dynamic] == [2, 3]
    assert [state.actor_id for state in vehicles] == [2]


def test_capture_rejects_mismatched_tick_frame() -> None:
    world_snapshot = Mock(frame=11, timestamp=SimpleNamespace(elapsed_seconds=1.0))
    world = Mock()
    world.get_snapshot.return_value = world_snapshot

    with pytest.raises(RuntimeError, match="does not match tick frame 10"):
        WorldFrame.capture(world, frame=10)
