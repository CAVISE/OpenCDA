"""Tests for safety sensor adapters."""

from unittest.mock import Mock

from opencda.core.safety.sensors import CollisionSensor


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
