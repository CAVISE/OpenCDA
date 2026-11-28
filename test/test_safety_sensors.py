"""Tests for safety sensor adapters."""

from types import SimpleNamespace
from unittest.mock import Mock

import carla
import numpy as np

from opencda.core.safety.sensors import CollisionSensor, OffRoadDetector, TrafficLightDector


def test_collision_sensor_binds_precreated_actor_and_callback() -> None:
    sensor_actor = Mock()
    sensor_actor.is_alive = True

    collision_sensor = CollisionSensor.from_sensor_actor(
        sensor_actor,
        {"history_size": 5, "col_thresh": 1.0},
    )

    assert collision_sensor.sensor is sensor_actor
    sensor_actor.listen.assert_called_once()

    collision_sensor.destroy()

    sensor_actor.stop.assert_called_once_with()
    sensor_actor.destroy.assert_called_once_with()


def test_collision_sensor_stop_is_idempotent_and_preserves_history() -> None:
    sensor_actor = Mock()
    sensor_actor.is_alive = True
    collision_sensor = CollisionSensor.from_sensor_actor(
        sensor_actor,
        {"history_size": 5, "col_thresh": 1.0},
    )
    collision_sensor._history.append((10, 2.5))

    collision_sensor.stop()
    collision_sensor.stop()

    sensor_actor.stop.assert_called_once_with()
    assert list(collision_sensor._history) == [(10, 2.5)]

    collision_sensor.destroy()

    sensor_actor.stop.assert_called_once_with()
    sensor_actor.destroy.assert_called_once_with()
    assert not collision_sensor._history


def test_offroad_detector_uses_direct_road_state_without_bev() -> None:
    detector = OffRoadDetector({})

    detector.tick({"on_road": True, "static_bev": None})
    assert detector.return_status() == {"offroad": False}

    detector.tick({"on_road": False, "static_bev": None})
    assert detector.return_status() == {"offroad": True}


def test_offroad_detector_keeps_full_bev_fallback() -> None:
    detector = OffRoadDetector({})
    static_bev = np.zeros((3, 3, 3), dtype=np.uint8)
    static_bev[1, 1] = 255

    detector.tick({"on_road": None, "static_bev": static_bev})

    assert detector.return_status() == {"offroad": True}


def test_traffic_light_trigger_waypoints_stop_before_junction_with_current_api() -> None:
    junction = SimpleNamespace(is_junction=True)
    approach = SimpleNamespace(road_id=7, lane_id=-1, is_junction=False, next=lambda _distance: [junction])
    trigger = SimpleNamespace(road_id=7, lane_id=-1, is_junction=False, next=lambda _distance: [approach])
    assert all(not hasattr(waypoint, "is_intersection") for waypoint in (trigger, approach, junction))

    traffic_light = SimpleNamespace(
        get_transform=lambda: carla.Transform(carla.Location(x=10.0)),
        trigger_volume=SimpleNamespace(location=carla.Location(), extent=carla.Vector3D(x=0.5, y=1.0, z=1.0)),
    )
    detector = TrafficLightDector.__new__(TrafficLightDector)
    detector._map = Mock()
    detector._map.get_waypoint.return_value = trigger

    assert detector._get_traffic_light_trigger_waypoints(traffic_light) == [approach]
    detector._map.get_waypoint.assert_called_once()
