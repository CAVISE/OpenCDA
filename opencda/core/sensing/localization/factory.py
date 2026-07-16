"""Localization provider factory."""

from __future__ import annotations

from typing import Any, Mapping

import carla

from opencda.core.sensing.localization.gt_localizer import GTLocalizer
from opencda.core.sensing.localization.kalman_filter import KalmanFilter
from opencda.core.sensing.localization.protocol import Localizer
from opencda.core.sensing.localization.sensor_localizer import SensorLocalizer
from opencda.core.sensing.sensor_types import SensorActorBundle


def create_localizer(
    actor: carla.Actor,
    config: Mapping[str, Any],
    carla_map: carla.Map,
    *,
    use_imu: bool,
    sensor_actors: SensorActorBundle | None = None,
) -> Localizer:
    """Create a localization provider for a CARLA actor."""
    provider = resolve_localization_provider(config)
    if provider == "gt":
        return GTLocalizer(actor)

    estimator = _create_estimator(config) if use_imu else None
    if sensor_actors is not None:
        if sensor_actors.gnss is None:
            raise ValueError("Sensor localization requires a GNSS actor.")
        if use_imu and sensor_actors.imu is None:
            raise ValueError("CAV sensor localization requires an IMU actor.")
        return SensorLocalizer.for_sensor_actors(
            carla_map=carla_map,
            gnss_actor=sensor_actors.gnss,
            imu_actor=sensor_actors.imu if use_imu else None,
            estimator=estimator,
        )

    return SensorLocalizer.for_actor(
        actor=actor,
        config=config,
        carla_map=carla_map,
        use_imu=use_imu,
        estimator=estimator,
    )


def resolve_localization_provider(config: Mapping[str, Any]) -> str:
    provider = config.get("provider")
    if provider is None:
        activate = config.get("activate")
        if not isinstance(activate, bool):
            raise ValueError("Localization config must define 'provider' or boolean 'activate'.")
        provider = "sensor" if activate else "gt"

    if provider not in {"gt", "sensor"}:
        raise ValueError("Localization provider must be either 'gt' or 'sensor'.")
    return provider


def _create_estimator(config: Mapping[str, Any]) -> Any:
    estimator_name = config.get("estimator", "kf")
    dt = float(config["dt"])

    if estimator_name == "kf":
        return KalmanFilter(dt)
    if estimator_name == "ekf":
        from opencda.customize.core.sensing.localization.extented_kalman_filter import ExtentedKalmanFilter

        return ExtentedKalmanFilter(dt)

    raise ValueError("Localization estimator must be either 'kf' or 'ekf'.")
