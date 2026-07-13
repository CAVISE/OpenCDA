"""Sensor-based localization provider."""

from __future__ import annotations

import math
from typing import Any, Mapping, Protocol

import carla

from opencda.core.application.behavior.types import Location, Rotation, Transform
from opencda.core.sensing.localization.coordinate_transform import geo_to_transform
from opencda.core.sensing.localization.kalman_filter import KalmanFilter
from opencda.core.sensing.localization.sensors import GnssSensor, ImuSensor
from opencda.core.sensing.localization.types import LocalizationSource, LocalizationState


class _Estimator(Protocol):
    def run_step_init(self, x: float, y: float, heading: float, velocity: float) -> None: ...

    def run_step(
        self,
        x: float,
        y: float,
        heading: float,
        velocity: float,
        yaw_rate_imu: float,
    ) -> tuple[float, float, float, float]: ...


class SensorLocalizer:
    """Produce localization states from GNSS and optional IMU measurements."""

    def __init__(
        self,
        carla_map: carla.Map,
        gnss: Any,
        imu: Any | None = None,
        estimator: _Estimator | None = None,
    ) -> None:
        if imu is not None and estimator is None:
            raise ValueError("An estimator is required when IMU localization is enabled.")

        self._gnss = gnss
        self._imu = imu
        self._estimator = estimator
        self._state: LocalizationState | None = None
        self._previous_measurement: tuple[float, float, float, float] | None = None
        self._estimator_initialized = False
        self._destroyed = False

        geo_ref = carla_map.transform_to_geolocation(carla.Location(x=0, y=0, z=0))
        self._reference_latitude = geo_ref.latitude
        self._reference_longitude = geo_ref.longitude

    @classmethod
    def for_actor(
        cls,
        actor: carla.Actor,
        config: Mapping[str, Any],
        carla_map: carla.Map,
        use_imu: bool,
        estimator: _Estimator | None = None,
    ) -> "SensorLocalizer":
        """Create a sensor localizer attached to a CARLA actor."""
        return cls(
            carla_map=carla_map,
            gnss=GnssSensor(actor, config["gnss"]),
            imu=ImuSensor(actor) if use_imu else None,
            estimator=(estimator or KalmanFilter(float(config["dt"]))) if use_imu else None,
        )

    def update(self) -> LocalizationState:
        x, y, z = geo_to_transform(
            self._gnss.lat,
            self._gnss.lon,
            self._gnss.alt,
            self._reference_latitude,
            self._reference_longitude,
            0.0,
        )
        timestamp = float(self._gnss.timestamp)

        if self._imu is None:
            transform = Transform(location=Location(x=x, y=y, z=z))
            speed_kmh = 0.0
        else:
            transform, speed_kmh = self._update_fused_state(x, y, z, timestamp)

        self._previous_measurement = (x, y, z, timestamp)
        self._state = LocalizationState(
            transform=transform,
            speed_kmh=speed_kmh,
            source=LocalizationSource.SENSOR,
            timestamp=timestamp,
        )
        return self._state

    def _update_fused_state(self, x: float, y: float, z: float, timestamp: float) -> tuple[Transform, float]:
        estimator = self._estimator
        if estimator is None:
            raise RuntimeError("SensorLocalizer estimator is not configured.")

        speed_mps = self._calculate_speed_mps(x, y, z, timestamp)
        heading_degrees = self._calculate_heading_degrees(x, y)
        heading_radians = math.radians(heading_degrees)

        if not self._estimator_initialized:
            estimator.run_step_init(x, y, heading_radians, speed_mps)
            estimated_x = x
            estimated_y = y
            estimated_heading_degrees = heading_degrees
            estimated_speed_mps = speed_mps
            self._estimator_initialized = True
        else:
            estimated_x, estimated_y, estimated_heading, estimated_speed_mps = estimator.run_step(
                x,
                y,
                heading_radians,
                speed_mps,
                self._get_yaw_rate(),
            )
            estimated_heading_degrees = math.degrees(estimated_heading)

        return (
            Transform(
                location=Location(x=estimated_x, y=estimated_y, z=z),
                rotation=Rotation(yaw=estimated_heading_degrees),
            ),
            estimated_speed_mps * 3.6,
        )

    def _calculate_speed_mps(self, x: float, y: float, z: float, timestamp: float) -> float:
        previous = self._previous_measurement
        if previous is None:
            return 0.0

        previous_x, previous_y, previous_z, previous_timestamp = previous
        elapsed = timestamp - previous_timestamp
        if elapsed <= 0:
            return 0.0

        distance = math.sqrt((x - previous_x) ** 2 + (y - previous_y) ** 2 + (z - previous_z) ** 2)
        return distance / elapsed

    def _calculate_heading_degrees(self, x: float, y: float) -> float:
        compass = getattr(self._imu, "compass", None)
        if compass is not None:
            return self._normalize_yaw(math.degrees(float(compass)) - 90.0)

        previous = self._previous_measurement
        if previous is not None:
            delta_x = x - previous[0]
            delta_y = y - previous[1]
            if delta_x != 0 or delta_y != 0:
                return math.degrees(math.atan2(delta_y, delta_x))

        if self._state is not None:
            return self._state.transform.rotation.yaw
        return 0.0

    def _get_yaw_rate(self) -> float:
        gyroscope = getattr(self._imu, "gyroscope", None)
        return float(gyroscope[2]) if gyroscope is not None else 0.0

    @staticmethod
    def _normalize_yaw(yaw: float) -> float:
        return (yaw + 180.0) % 360.0 - 180.0

    def get_state(self) -> LocalizationState:
        if self._state is None:
            raise RuntimeError("SensorLocalizer has no state. Call update() first.")
        return self._state

    def destroy(self) -> None:
        if self._destroyed:
            return

        self._gnss.sensor.destroy()
        if self._imu is not None:
            self._imu.sensor.destroy()
        self._destroyed = True
