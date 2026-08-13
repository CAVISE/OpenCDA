"""Tests for the ground-truth localization provider."""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import carla
import pytest

from opencda.core.sensing.localization import GTLocalizer, LocalizationSource, Localizer


def test_update_reads_actor_state() -> None:
    actor = Mock()
    actor.get_transform.return_value = carla.Transform(
        carla.Location(x=1.0, y=2.0, z=3.0),
        carla.Rotation(pitch=4.0, yaw=5.0, roll=6.0),
    )
    actor.get_velocity.return_value = carla.Vector3D(x=3.0, y=4.0, z=0.0)
    localizer = GTLocalizer(actor)

    state = localizer.update()

    assert isinstance(localizer, Localizer)
    assert state.transform.location.x == 1.0
    assert state.transform.location.y == 2.0
    assert state.transform.location.z == 3.0
    assert state.transform.rotation.yaw == 5.0
    assert state.speed_kmh == pytest.approx(18.0)
    assert state.source is LocalizationSource.GT
    assert localizer.get_state() is state


def test_get_state_requires_update() -> None:
    localizer = GTLocalizer(Mock())

    with pytest.raises(RuntimeError, match=r"Call update\(\) first"):
        localizer.get_state()


def test_update_uses_shared_world_frame_without_actor_state_calls() -> None:
    actor = Mock(id=7)
    actor_state = Mock(
        transform=carla.Transform(carla.Location(x=10.0, y=20.0, z=1.0)),
        velocity=carla.Vector3D(x=0.0, y=5.0, z=0.0),
    )
    world_frame = Mock(frame=42, timestamp=3.5)
    world_frame.actor_state.return_value = actor_state

    state = GTLocalizer(actor).update(world_frame)

    world_frame.actor_state.assert_called_once_with(7)
    actor.get_transform.assert_not_called()
    actor.get_velocity.assert_not_called()
    assert state.transform.location.x == 10.0
    assert state.speed_kmh == pytest.approx(18.0)
    assert state.frame == 42
    assert state.timestamp == pytest.approx(3.5)


def test_state_is_immutable() -> None:
    actor = Mock()
    actor.get_transform.return_value = carla.Transform()
    actor.get_velocity.return_value = carla.Vector3D()
    state = GTLocalizer(actor).update()

    with pytest.raises(FrozenInstanceError):
        state.speed_kmh = 1.0


def test_destroy_does_not_destroy_actor() -> None:
    actor = Mock()

    GTLocalizer(actor).destroy()

    actor.destroy.assert_not_called()
