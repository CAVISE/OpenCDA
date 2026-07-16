"""Tests for localization provider selection."""

from unittest.mock import Mock, patch

import pytest

from opencda.core.sensing.localization.factory import create_localizer
from opencda.core.sensing.sensor_types import SensorActorBundle


@pytest.mark.parametrize(
    ("config", "expected_provider"),
    [
        ({"provider": "gt"}, "gt"),
        ({"activate": False}, "gt"),
        ({"provider": "sensor", "dt": 0.05}, "sensor"),
        ({"activate": True, "dt": 0.05}, "sensor"),
    ],
)
def test_create_localizer_selects_provider(config: dict, expected_provider: str) -> None:
    actor = Mock()
    carla_map = Mock()

    with (
        patch("opencda.core.sensing.localization.factory.GTLocalizer") as gt_localizer,
        patch("opencda.core.sensing.localization.factory.SensorLocalizer.for_actor") as sensor_localizer,
    ):
        result = create_localizer(actor, config, carla_map, use_imu=True)

    if expected_provider == "gt":
        gt_localizer.assert_called_once_with(actor)
        sensor_localizer.assert_not_called()
        assert result is gt_localizer.return_value
    else:
        gt_localizer.assert_not_called()
        sensor_localizer.assert_called_once()
        assert sensor_localizer.call_args.kwargs["actor"] is actor
        assert sensor_localizer.call_args.kwargs["carla_map"] is carla_map
        assert sensor_localizer.call_args.kwargs["use_imu"] is True
        assert result is sensor_localizer.return_value


def test_rsu_sensor_provider_does_not_create_estimator() -> None:
    with (
        patch("opencda.core.sensing.localization.factory._create_estimator") as create_estimator,
        patch("opencda.core.sensing.localization.factory.SensorLocalizer.for_actor") as sensor_localizer,
    ):
        create_localizer(Mock(), {"provider": "sensor"}, Mock(), use_imu=False)

    create_estimator.assert_not_called()
    assert sensor_localizer.call_args.kwargs["estimator"] is None


def test_sensor_provider_binds_precreated_sensor_actors() -> None:
    gnss_actor = Mock()
    imu_actor = Mock()
    sensor_actors = SensorActorBundle(gnss=gnss_actor, imu=imu_actor)

    with (
        patch("opencda.core.sensing.localization.factory.SensorLocalizer.for_actor") as legacy_factory,
        patch("opencda.core.sensing.localization.factory.SensorLocalizer.for_sensor_actors") as batch_factory,
    ):
        result = create_localizer(
            Mock(),
            {"provider": "sensor", "dt": 0.05},
            Mock(),
            use_imu=True,
            sensor_actors=sensor_actors,
        )

    legacy_factory.assert_not_called()
    assert batch_factory.call_args.kwargs["gnss_actor"] is gnss_actor
    assert batch_factory.call_args.kwargs["imu_actor"] is imu_actor
    assert result is batch_factory.return_value


@pytest.mark.parametrize("provider", ["carla", "", None])
def test_invalid_provider_is_rejected(provider: str | None) -> None:
    config = {"provider": provider}

    with pytest.raises(ValueError, match="provider"):
        create_localizer(Mock(), config, Mock(), use_imu=False)


def test_invalid_estimator_is_rejected() -> None:
    config = {"provider": "sensor", "estimator": "unknown", "dt": 0.05}

    with pytest.raises(ValueError, match="estimator"):
        create_localizer(Mock(), config, Mock(), use_imu=True)
