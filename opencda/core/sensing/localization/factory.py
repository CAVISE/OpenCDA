"""Localization provider factory."""

from __future__ import annotations

from typing import Any, Mapping

import carla

from opencda.core.sensing.localization.gt_localizer import GTLocalizer
from opencda.core.sensing.localization.kalman_filter import KalmanFilter
from opencda.core.sensing.localization.protocol import Localizer
from opencda.core.sensing.localization.sensor_localizer import SensorLocalizer


def create_localizer(
    actor: carla.Actor,
    config: Mapping[str, Any],
    carla_map: carla.Map,
    *,
    use_imu: bool,
) -> Localizer:
    """Create a localization provider for a CARLA actor."""
    provider = _resolve_provider(config)
    if provider == "gt":
        return GTLocalizer(actor)

    estimator = _create_estimator(config) if use_imu else None
    return SensorLocalizer.for_actor(
        actor=actor,
        config=config,
        carla_map=carla_map,
        use_imu=use_imu,
        estimator=estimator,
    )


def _resolve_provider(config: Mapping[str, Any]) -> str:
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
