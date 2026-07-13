"""Tests for the sensor localization provider."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from opencda.core.sensing.localization import LocalizationSource, Localizer, SensorLocalizer


def _carla_map() -> Mock:
    carla_map = Mock()
    carla_map.transform_to_geolocation.return_value = SimpleNamespace(latitude=10.0, longitude=20.0)
    return carla_map


def _gnss() -> SimpleNamespace:
    return SimpleNamespace(lat=1.0, lon=2.0, alt=3.0, timestamp=1.0, sensor=Mock())


def test_rsu_update_returns_gnss_position_and_zero_speed() -> None:
    gnss = _gnss()
    localizer = SensorLocalizer(carla_map=_carla_map(), gnss=gnss)

    with patch(
        "opencda.core.sensing.localization.sensor_localizer.geo_to_transform",
        return_value=(4.0, 5.0, 6.0),
    ):
        state = localizer.update()

    assert isinstance(localizer, Localizer)
    assert state.transform.location.x == 4.0
    assert state.transform.location.y == 5.0
    assert state.transform.location.z == 6.0
    assert state.speed_kmh == 0.0
    assert state.source is LocalizationSource.SENSOR
    assert state.timestamp == 1.0


def test_for_actor_uses_same_gnss_flow_with_optional_imu() -> None:
    sensor_localizer_module = __import__(
        "opencda.core.sensing.localization.sensor_localizer",
        fromlist=["GnssSensor"],
    )
    gnss = _gnss()
    imu = SimpleNamespace(sensor=Mock())
    estimator = Mock()
    actor = Mock()

    with (
        patch.object(sensor_localizer_module, "GnssSensor", return_value=gnss) as gnss_factory,
        patch.object(sensor_localizer_module, "ImuSensor", return_value=imu) as imu_factory,
    ):
        vehicle_localizer = SensorLocalizer.for_actor(
            actor=actor,
            config={"gnss": {}, "dt": 0.05},
            carla_map=_carla_map(),
            use_imu=True,
            estimator=estimator,
        )
        rsu_localizer = SensorLocalizer.for_actor(
            actor=actor,
            config={"gnss": {}},
            carla_map=_carla_map(),
            use_imu=False,
        )

    assert isinstance(vehicle_localizer, SensorLocalizer)
    assert isinstance(rsu_localizer, SensorLocalizer)
    assert gnss_factory.call_count == 2
    imu_factory.assert_called_once_with(actor)


def test_vehicle_update_fuses_gnss_and_imu() -> None:
    gnss = _gnss()
    imu = SimpleNamespace(compass=0.0, gyroscope=(0.0, 0.0, 0.25), sensor=Mock())
    estimator = Mock()
    estimator.run_step.return_value = (7.0, 8.0, 0.5, 5.0)
    localizer = SensorLocalizer(carla_map=_carla_map(), gnss=gnss, imu=imu, estimator=estimator)

    with patch(
        "opencda.core.sensing.localization.sensor_localizer.geo_to_transform",
        side_effect=[(0.0, 0.0, 1.0), (3.0, 4.0, 1.0)],
    ):
        initial_state = localizer.update()
        gnss.timestamp = 2.0
        state = localizer.update()

    assert initial_state.speed_kmh == 0.0
    estimator.run_step_init.assert_called_once_with(0.0, 0.0, pytest.approx(-1.5707963267948966), 0.0)
    estimator.run_step.assert_called_once_with(3.0, 4.0, pytest.approx(-1.5707963267948966), 5.0, 0.25)
    assert state.transform.location.x == 7.0
    assert state.transform.location.y == 8.0
    assert state.transform.rotation.yaw == pytest.approx(28.6478897565)
    assert state.speed_kmh == pytest.approx(18.0)


def test_imu_requires_estimator() -> None:
    with pytest.raises(ValueError, match="estimator is required"):
        SensorLocalizer(carla_map=_carla_map(), gnss=_gnss(), imu=Mock())


def test_get_state_requires_update() -> None:
    localizer = SensorLocalizer(carla_map=_carla_map(), gnss=_gnss())

    with pytest.raises(RuntimeError, match=r"Call update\(\) first"):
        localizer.get_state()


def test_destroy_releases_sensors_once() -> None:
    gnss = _gnss()
    imu = SimpleNamespace(sensor=Mock())
    localizer = SensorLocalizer(carla_map=_carla_map(), gnss=gnss, imu=imu, estimator=Mock())

    localizer.destroy()
    localizer.destroy()

    gnss.sensor.destroy.assert_called_once_with()
    imu.sensor.destroy.assert_called_once_with()
