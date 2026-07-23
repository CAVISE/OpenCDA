"""Tests for the shared per-tick CARLA world frame."""

from types import SimpleNamespace
from unittest.mock import Mock

import carla
import pytest

from opencda.core.common.world_frame import WorldFrame, WorldFrameBuilder


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


def test_builder_caches_traffic_light_geometry_and_refreshes_state() -> None:
    traffic_light = _actor(5, "traffic.traffic_light")
    traffic_light.get_transform.return_value = carla.Transform(carla.Location(x=10.0, y=20.0, z=0.0))
    traffic_light.trigger_volume = SimpleNamespace(
        location=carla.Location(),
        extent=carla.Vector3D(x=1.0, y=1.0, z=1.0),
    )
    traffic_light.get_state.side_effect = ["RED", "GREEN"]

    waypoint = Mock()
    waypoint.road_id = 7
    waypoint.transform = carla.Transform(carla.Location(x=11.0, y=20.0, z=0.0))
    waypoint.is_intersection = True
    carla_map = Mock()
    carla_map.get_waypoint.return_value = waypoint

    snapshots = []
    for frame in (17, 18):
        world_snapshot = Mock(frame=frame, timestamp=SimpleNamespace(elapsed_seconds=float(frame)))
        world_snapshot.find.return_value = _actor_snapshot(5, x=10.0, y=20.0)
        snapshots.append(world_snapshot)
    world = Mock()
    world.get_snapshot.side_effect = snapshots
    world.get_actors.return_value = [traffic_light]

    builder = WorldFrameBuilder(world, carla_map)
    first_frame = builder.capture(frame=17)
    second_frame = builder.capture(frame=18)

    assert first_frame.traffic_light_states[0].state == "RED"
    assert second_frame.traffic_light_states[0].state == "GREEN"
    assert second_frame.traffic_light_states[0].road_id == 7
    assert second_frame.traffic_light_states[0].intersection_location is waypoint.transform.location
    carla_map.get_waypoint.assert_called_once()
    traffic_light.get_transform.assert_called_once()
    assert traffic_light.get_state.call_count == 2


def test_shared_actor_value_is_created_once_per_frame_and_namespace() -> None:
    world_snapshot = Mock(frame=21, timestamp=SimpleNamespace(elapsed_seconds=2.0))
    world_snapshot.find.return_value = _actor_snapshot(1, x=0.0, y=0.0)
    world = Mock()
    world.get_snapshot.return_value = world_snapshot
    world.get_actors.return_value = [_actor(1, "vehicle.test")]
    world_frame = WorldFrame.capture(world)
    factory = Mock(side_effect=[object(), object()])

    first = world_frame.shared_actor_value("obstacle", 1, factory)
    second = world_frame.shared_actor_value("obstacle", 1, factory)
    other_namespace = world_frame.shared_actor_value("other", 1, factory)

    assert first is second
    assert other_namespace is not first
    assert factory.call_count == 2
