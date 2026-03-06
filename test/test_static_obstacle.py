"""
Unit tests for opencda.core.sensing.perception.static_obstacle.

These tests validate the data/geometry contracts for:
- BoundingBox
- StaticObstacle
- TrafficLight (including trigger location helper)

CARLA is not available in CI; a lightweight stub is provided via test/conftest.py.
"""

from __future__ import annotations

import math

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def _make_corners(min_xyz: tuple[float, float, float], max_xyz: tuple[float, float, float]) -> np.ndarray:
    """Build 8 corners of an axis-aligned box from min/max bounds."""
    x0, y0, z0 = min_xyz
    x1, y1, z1 = max_xyz
    return np.array(
        [
            [x0, y0, z0],
            [x0, y0, z1],
            [x0, y1, z0],
            [x0, y1, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
        ],
        dtype=float,
    )


def test_bounding_box_computes_center_and_extent_from_corners():
    """BoundingBox computes center location and half-axes extents from corners."""
    from opencda.core.sensing.perception.static_obstacle import BoundingBox

    corners = _make_corners(min_xyz=(0.0, -1.0, 5.0), max_xyz=(2.0, 3.0, 7.0))
    bbox = BoundingBox(corners)

    assert bbox.location.x == pytest.approx(1.0)
    assert bbox.location.y == pytest.approx(1.0)
    assert bbox.location.z == pytest.approx(6.0)

    assert bbox.extent.x == pytest.approx(1.0)
    assert bbox.extent.y == pytest.approx(2.0)
    assert bbox.extent.z == pytest.approx(1.0)

    assert isinstance(bbox.location.x, (int, float))
    assert isinstance(bbox.extent.z, (int, float))


def test_bounding_box_rejects_wrong_corner_shape_current_behavior():
    """
    Negative scenario: wrong corners shape.

    Current behavior: corners with shape (8, 2) causes IndexError because code accesses corners[:, 2].
    """
    from opencda.core.sensing.perception.static_obstacle import BoundingBox

    bad_corners = np.zeros((8, 2), dtype=float)
    with pytest.raises(IndexError):
        BoundingBox(bad_corners)


def test_bounding_box_rejects_list_input_current_behavior():
    """
    Negative scenario: list input instead of numpy array.

    Current behavior: list does not support 2D slicing (corners[:, 0]) and raises TypeError.
    """
    from opencda.core.sensing.perception.static_obstacle import BoundingBox

    corners_list = _make_corners(min_xyz=(0.0, 0.0, 0.0), max_xyz=(1.0, 1.0, 1.0)).tolist()
    with pytest.raises(TypeError):
        BoundingBox(corners_list)


def test_static_obstacle_stores_bounding_box_and_o3d_bbx_reference():
    """StaticObstacle stores bounding_box and keeps o3d_bbx reference unchanged."""
    from opencda.core.sensing.perception.static_obstacle import StaticObstacle

    corners = _make_corners(min_xyz=(-1.0, -2.0, 0.0), max_xyz=(3.0, 2.0, 4.0))
    o3d_bbx = object()

    obstacle = StaticObstacle(corners, o3d_bbx)

    assert obstacle.o3d_bbx is o3d_bbx
    assert obstacle.bounding_box.location.x == pytest.approx(1.0)
    assert obstacle.bounding_box.location.y == pytest.approx(0.0)
    assert obstacle.bounding_box.location.z == pytest.approx(2.0)
    assert obstacle.bounding_box.extent.x == pytest.approx(2.0)
    assert obstacle.bounding_box.extent.y == pytest.approx(2.0)
    assert obstacle.bounding_box.extent.z == pytest.approx(2.0)


def test_traffic_light_getters_return_constructor_values():
    """TrafficLight getters return the values provided to the constructor."""
    from opencda.core.sensing.perception.static_obstacle import TrafficLight

    import carla

    actor = object()
    trigger_location = carla.Location(x=10.0, y=20.0, z=30.0)
    state = object()

    tl = TrafficLight(actor, trigger_location, state)

    assert tl.actor is actor
    assert tl.get_location() is trigger_location
    assert tl.get_state() is state


def test_get_trafficlight_trigger_location_calls_get_transform_and_transform_and_returns_new_location():
    """
    Contract test for get_trafficlight_trigger_location:
    - must call actor.get_transform() and base_transform.transform(trigger_volume.location)
    - must return a new Location instance with the same coordinates
    """
    from opencda.core.sensing.perception.static_obstacle import TrafficLight

    import carla

    area_loc = carla.Location(10.0, 20.0, 30.0)
    base_transform = Mock(spec_set=["rotation", "transform"])
    base_transform.rotation = carla.Rotation(yaw=0.0)
    base_transform.transform.return_value = area_loc

    traffic_light = Mock(spec_set=["get_transform", "trigger_volume"])
    traffic_light.get_transform.return_value = base_transform
    traffic_light.trigger_volume = SimpleNamespace(
        location=carla.Location(1.0, 2.0, 3.0),
        extent=carla.Vector3D(0.0, 0.0, 4.0),
    )

    out = TrafficLight.get_trafficlight_trigger_location(traffic_light)

    assert isinstance(out, carla.Location)
    assert out == area_loc
    assert out is not area_loc
    traffic_light.get_transform.assert_called_once_with()
    base_transform.transform.assert_called_once_with(traffic_light.trigger_volume.location)


@pytest.mark.parametrize("extent_z", [0.0, 1.0, 10.0, 123.4])
def test_get_trafficlight_trigger_location_is_independent_of_extent_z_current_behavior(extent_z):
    """
    Current behavior: extent.z does not affect the returned trigger location coordinates.
    """
    from opencda.core.sensing.perception.static_obstacle import TrafficLight

    import carla

    area_loc = carla.Location(7.0, 8.0, 9.0)
    base_transform = Mock(spec_set=["rotation", "transform"])
    base_transform.rotation = carla.Rotation(yaw=30.0)
    base_transform.transform.return_value = area_loc

    traffic_light = Mock(spec_set=["get_transform", "trigger_volume"])
    traffic_light.get_transform.return_value = base_transform
    traffic_light.trigger_volume = SimpleNamespace(
        location=carla.Location(0.5, 0.5, 0.5),
        extent=carla.Vector3D(0.0, 0.0, float(extent_z)),
    )

    out = TrafficLight.get_trafficlight_trigger_location(traffic_light)

    assert out == area_loc
    assert out is not area_loc
    traffic_light.get_transform.assert_called_once_with()
    base_transform.transform.assert_called_once_with(traffic_light.trigger_volume.location)


class _FakeTriggerVolume:
    def __init__(self, location, extent):
        self.location = location
        self.extent = extent


class _FakeTransform:
    """
    Minimal transform stub for this module.

    Implements:
    - .rotation.yaw
    - transform(Location) applying yaw rotation around Z and then translation.
    """

    def __init__(self, location, yaw_degrees: float):
        import carla

        self.location = location
        self.rotation = carla.Rotation(yaw=yaw_degrees)

    def transform(self, point_location):
        import carla

        yaw = math.radians(float(self.rotation.yaw))
        x_rot = math.cos(yaw) * float(point_location.x) - math.sin(yaw) * float(point_location.y)
        y_rot = math.sin(yaw) * float(point_location.x) + math.cos(yaw) * float(point_location.y)
        z = float(point_location.z)

        return carla.Location(
            x=float(self.location.x) + x_rot,
            y=float(self.location.y) + y_rot,
            z=float(self.location.z) + z,
        )


class _FakeTrafficLightActor:
    def __init__(self, transform, trigger_volume):
        self._transform = transform
        self.trigger_volume = trigger_volume

    def get_transform(self):
        return self._transform


def _expected_transform_translation_and_yaw(base_location, yaw_degrees: float, local_location):
    yaw = math.radians(float(yaw_degrees))
    x_rot = math.cos(yaw) * float(local_location.x) - math.sin(yaw) * float(local_location.y)
    y_rot = math.sin(yaw) * float(local_location.x) + math.cos(yaw) * float(local_location.y)
    return (
        float(base_location.x) + x_rot,
        float(base_location.y) + y_rot,
        float(base_location.z) + float(local_location.z),
    )


def test_get_trafficlight_trigger_location_yaw_0_transforms_trigger_location():
    """yaw=0: output equals transformed trigger volume location."""
    from opencda.core.sensing.perception.static_obstacle import TrafficLight

    import carla

    base_loc = carla.Location(x=10.0, y=20.0, z=30.0)
    base_transform = _FakeTransform(location=base_loc, yaw_degrees=0.0)

    trigger_loc = carla.Location(x=1.0, y=2.0, z=3.0)
    trigger_volume = _FakeTriggerVolume(location=trigger_loc, extent=carla.Vector3D(x=4.0, y=5.0, z=6.0))

    actor = _FakeTrafficLightActor(transform=base_transform, trigger_volume=trigger_volume)

    result = TrafficLight.get_trafficlight_trigger_location(actor)

    exp_x, exp_y, exp_z = _expected_transform_translation_and_yaw(base_loc, 0.0, trigger_loc)
    assert result.x == pytest.approx(exp_x)
    assert result.y == pytest.approx(exp_y)
    assert result.z == pytest.approx(exp_z)


def test_get_trafficlight_trigger_location_yaw_90_applies_rotation_in_transform():
    """yaw=90: output reflects rotation applied inside transform()."""
    from opencda.core.sensing.perception.static_obstacle import TrafficLight

    import carla

    base_loc = carla.Location(x=10.0, y=20.0, z=30.0)
    base_transform = _FakeTransform(location=base_loc, yaw_degrees=90.0)

    trigger_loc = carla.Location(x=1.0, y=2.0, z=0.0)
    trigger_volume = _FakeTriggerVolume(location=trigger_loc, extent=carla.Vector3D(x=1.0, y=1.0, z=1.0))

    actor = _FakeTrafficLightActor(transform=base_transform, trigger_volume=trigger_volume)

    result = TrafficLight.get_trafficlight_trigger_location(actor)

    exp_x, exp_y, exp_z = _expected_transform_translation_and_yaw(base_loc, 90.0, trigger_loc)
    assert result.x == pytest.approx(exp_x)
    assert result.y == pytest.approx(exp_y)
    assert result.z == pytest.approx(exp_z)


@pytest.mark.parametrize("yaw_degrees", [-45.0, 15.0, 180.0])
def test_get_trafficlight_trigger_location_various_yaws_applies_rotation_in_transform(yaw_degrees):
    """Various yaw values: output reflects yaw applied inside transform()."""
    from opencda.core.sensing.perception.static_obstacle import TrafficLight

    import carla

    base_loc = carla.Location(x=10.0, y=20.0, z=30.0)
    base_transform = _FakeTransform(location=base_loc, yaw_degrees=float(yaw_degrees))

    trigger_loc = carla.Location(x=1.0, y=2.0, z=3.0)
    trigger_volume = _FakeTriggerVolume(location=trigger_loc, extent=carla.Vector3D(x=0.0, y=0.0, z=0.0))

    actor = _FakeTrafficLightActor(transform=base_transform, trigger_volume=trigger_volume)

    result = TrafficLight.get_trafficlight_trigger_location(actor)
    exp_x, exp_y, exp_z = _expected_transform_translation_and_yaw(base_loc, float(yaw_degrees), trigger_loc)
    assert result.x == pytest.approx(exp_x)
    assert result.y == pytest.approx(exp_y)
    assert result.z == pytest.approx(exp_z)
