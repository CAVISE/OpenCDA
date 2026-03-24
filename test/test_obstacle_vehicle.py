"""
Contract unit tests for opencda.core.sensing.perception.obstacle_vehicle.

These tests are deterministic and avoid CARLA/Open3D runtime dependencies by:
- Using the CARLA stubs installed via test/conftest.py (sys.modules["carla"]).
- Patching Open3D API surface used by obstacle_vehicle.set_vehicle().
- Patching sensor_transformation.world_to_sensor() for stable geometry checks.
"""

from __future__ import annotations

import types

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


class _FakeAxisAlignedBoundingBox:
    """Minimal stand-in for open3d.geometry.AxisAlignedBoundingBox used for unit tests."""

    def __init__(self, *, min_bound, max_bound):
        self.min_bound = np.asarray(min_bound, dtype=float)
        self.max_bound = np.asarray(max_bound, dtype=float)
        self.color = None


@pytest.fixture
def fake_open3d_aabb(monkeypatch):
    """
    Ensure open3d.geometry.AxisAlignedBoundingBox exists and is deterministic.

    test/conftest.py installs a lightweight open3d stub module that may not include geometry.
    """
    import sys

    o3d = sys.modules["open3d"]
    if not hasattr(o3d, "geometry"):
        o3d.geometry = types.SimpleNamespace()

    monkeypatch.setattr(o3d.geometry, "AxisAlignedBoundingBox", _FakeAxisAlignedBoundingBox, raising=False)
    return o3d


class _FakeLidarSensor:
    def __init__(self, transform):
        self._transform = transform

    def get_transform(self):
        return self._transform


class _FakeVehicleActor:
    """Minimal vehicle actor API used by ObstacleVehicle.set_vehicle()."""

    def __init__(self, *, actor_id, location, transform, bounding_box, velocity, type_id, attributes):
        self.id = actor_id
        self._location = location
        self._transform = transform
        self.bounding_box = bounding_box
        self._velocity = velocity
        self.type_id = type_id
        if attributes is not None:
            self.attributes = attributes

    def get_location(self):
        return self._location

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity


def test_is_vehicle_cococlass_true_for_vehicle_labels_and_false_otherwise():
    from opencda.core.sensing.perception.obstacle_vehicle import is_vehicle_cococlass

    for label in (1, 2, 3, 5, 7):
        assert is_vehicle_cococlass(label) is True

    for label in (0, 4, 6, 8, 10, 13):
        assert is_vehicle_cococlass(label) is False

    assert is_vehicle_cococlass(np.int64(1)) is True
    assert is_vehicle_cococlass(1.0) is True
    assert is_vehicle_cococlass(np.int64(9)) is False


def test_bounding_box_computes_center_and_extent_from_corners():
    from opencda.core.sensing.perception.obstacle_vehicle import BoundingBox

    corners = _make_corners(min_xyz=(0.0, -1.0, 5.0), max_xyz=(2.0, 3.0, 7.0))
    bbox = BoundingBox(corners)

    assert bbox.location.x == pytest.approx(1.0)
    assert bbox.location.y == pytest.approx(1.0)
    assert bbox.location.z == pytest.approx(6.0)

    assert bbox.extent.x == pytest.approx(1.0)
    assert bbox.extent.y == pytest.approx(2.0)
    assert bbox.extent.z == pytest.approx(1.0)


def test_bounding_box_rejects_wrong_corner_shape_current_behavior():
    """
    Negative scenario: wrong corners shape.

    Current behavior: corners with shape (8, 2) causes IndexError because code accesses corners[:, 2].
    """
    from opencda.core.sensing.perception.obstacle_vehicle import BoundingBox

    bad_corners = np.zeros((8, 2), dtype=float)
    with pytest.raises(IndexError):
        BoundingBox(bad_corners)


def test_obstacle_vehicle_init_without_vehicle_sets_defaults():
    import carla

    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    corners = _make_corners(min_xyz=(-1.0, -2.0, 0.0), max_xyz=(3.0, 2.0, 4.0))
    o3d_bbx = object()

    ov = ObstacleVehicle(corners, o3d_bbx)

    assert ov.carla_id == -1
    assert ov.transform is None
    assert ov.o3d_bbx is o3d_bbx
    assert isinstance(ov.velocity, carla.Vector3D)
    assert ov.velocity.x == pytest.approx(0.0)
    assert ov.velocity.y == pytest.approx(0.0)
    assert ov.velocity.z == pytest.approx(0.0)

    assert ov.location == ov.bounding_box.location
    assert ov.location.x == pytest.approx(1.0)
    assert ov.location.y == pytest.approx(0.0)
    assert ov.location.z == pytest.approx(2.0)


def test_obstacle_vehicle_getters_and_setters():
    import carla

    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    corners = _make_corners(min_xyz=(0.0, 0.0, 0.0), max_xyz=(2.0, 2.0, 2.0))
    ov = ObstacleVehicle(corners, o3d_bbx=object())

    assert ov.get_location() == ov.location
    assert ov.get_transform() is None
    assert ov.get_velocity() == ov.velocity

    ov.set_carla_id(999)
    assert ov.carla_id == 999

    v = carla.Vector3D(x=1.1, y=2.2, z=3.3)
    ov.set_velocity(v)
    assert ov.get_velocity() == v


def test_set_vehicle_without_lidar_assigns_actor_fields_and_does_not_create_o3d_bbx():
    import carla

    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    vehicle_location = carla.Location(x=5.0, y=6.0, z=7.0)
    vehicle_transform = carla.Transform(vehicle_location, carla.Rotation(yaw=10.0))
    vehicle_bbox = carla.BoundingBox(carla.Location(x=0.0, y=0.0, z=0.5), carla.Vector3D(x=1.0, y=2.0, z=0.5))
    vehicle_velocity = carla.Vector3D(x=0.1, y=0.2, z=0.3)

    actor = _FakeVehicleActor(
        actor_id=101,
        location=vehicle_location,
        transform=vehicle_transform,
        bounding_box=vehicle_bbox,
        velocity=vehicle_velocity,
        type_id="vehicle.tesla.model3",
        attributes={"color": "255,0,0"},
    )

    ov = ObstacleVehicle(None, None, vehicle=actor, lidar=None, sumo2carla_ids={})

    assert ov.carla_id == 101
    assert ov.location == vehicle_location
    assert ov.transform == vehicle_transform
    assert ov.bounding_box == vehicle_bbox
    assert ov.velocity == vehicle_velocity
    assert ov.type_id == "vehicle.tesla.model3"
    assert ov.color == "255,0,0"

    assert not hasattr(ov, "o3d_bbx")


def test_set_vehicle_color_is_none_when_attributes_missing_or_color_missing():
    import carla

    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    vehicle_location = carla.Location(x=1.0, y=2.0, z=3.0)
    vehicle_transform = carla.Transform(vehicle_location, carla.Rotation(yaw=0.0))
    vehicle_bbox = carla.BoundingBox(carla.Location(x=0.0, y=0.0, z=0.0), carla.Vector3D(x=1.0, y=1.0, z=1.0))
    vehicle_velocity = carla.Vector3D(x=0.0, y=0.0, z=0.0)

    actor_no_attributes = _FakeVehicleActor(
        actor_id=11,
        location=vehicle_location,
        transform=vehicle_transform,
        bounding_box=vehicle_bbox,
        velocity=vehicle_velocity,
        type_id="vehicle.test.no_attributes",
        attributes=None,
    )
    ov1 = ObstacleVehicle(None, None, vehicle=actor_no_attributes, lidar=None, sumo2carla_ids={})
    assert ov1.color is None

    actor_no_color = _FakeVehicleActor(
        actor_id=12,
        location=vehicle_location,
        transform=vehicle_transform,
        bounding_box=vehicle_bbox,
        velocity=vehicle_velocity,
        type_id="vehicle.test.no_color",
        attributes={},
    )
    ov2 = ObstacleVehicle(None, None, vehicle=actor_no_color, lidar=None, sumo2carla_ids={})
    assert ov2.color is None


def test_set_vehicle_with_sumo_speed_overrides_velocity_x_only(monkeypatch):
    import carla

    from opencda.core.sensing.perception import obstacle_vehicle as ov_mod

    vehicle_location = carla.Location(x=5.0, y=6.0, z=7.0)
    vehicle_transform = carla.Transform(vehicle_location, carla.Rotation(yaw=0.0))
    vehicle_bbox = carla.BoundingBox(carla.Location(x=0.0, y=0.0, z=0.5), carla.Vector3D(x=1.0, y=2.0, z=0.5))
    vehicle_velocity = carla.Vector3D(x=0.1, y=0.2, z=0.3)

    actor = _FakeVehicleActor(
        actor_id=202,
        location=vehicle_location,
        transform=vehicle_transform,
        bounding_box=vehicle_bbox,
        velocity=vehicle_velocity,
        type_id="vehicle.audi.tt",
        attributes={},
    )

    monkeypatch.setattr(ov_mod, "get_speed_sumo", lambda _mapping, _carla_id: 12.3)

    ov = ov_mod.ObstacleVehicle(None, None, vehicle=actor, lidar=None, sumo2carla_ids={"sumo_1": 202})

    assert ov.carla_id == 202
    assert ov.velocity.x == pytest.approx(12.3)
    assert ov.velocity.y == pytest.approx(0.0)
    assert ov.velocity.z == pytest.approx(0.0)


def test_init_with_vehicle_and_sumo2carla_ids_none_does_not_call_get_speed_sumo(monkeypatch):
    """
    Contract: ObstacleVehicle.__init__ should treat sumo2carla_ids=None as empty mapping.
    In that case, get_speed_sumo must not be called and CARLA velocity remains unchanged.
    """
    import carla

    from opencda.core.sensing.perception import obstacle_vehicle as ov_mod

    vehicle_location = carla.Location(x=1.0, y=2.0, z=3.0)
    vehicle_transform = carla.Transform(vehicle_location, carla.Rotation(yaw=0.0))
    vehicle_bbox = carla.BoundingBox(carla.Location(x=0.0, y=0.0, z=0.5), carla.Vector3D(x=1.0, y=2.0, z=0.5))
    vehicle_velocity = carla.Vector3D(x=0.1, y=0.2, z=0.3)

    actor = _FakeVehicleActor(
        actor_id=999,
        location=vehicle_location,
        transform=vehicle_transform,
        bounding_box=vehicle_bbox,
        velocity=vehicle_velocity,
        type_id="vehicle.test.sumo_none",
        attributes={},
    )

    def _must_not_be_called(_mapping, _carla_id):  # noqa: ARG001
        raise AssertionError("get_speed_sumo must not be called when sumo2carla_ids is None/empty")

    monkeypatch.setattr(ov_mod, "get_speed_sumo", _must_not_be_called)

    ov = ov_mod.ObstacleVehicle(None, None, vehicle=actor, lidar=None, sumo2carla_ids=None)
    assert ov.carla_id == 999
    assert ov.velocity == vehicle_velocity


def test_set_vehicle_with_sumo_speed_non_positive_does_not_override(monkeypatch):
    import carla

    from opencda.core.sensing.perception import obstacle_vehicle as ov_mod

    vehicle_location = carla.Location(x=5.0, y=6.0, z=7.0)
    vehicle_transform = carla.Transform(vehicle_location, carla.Rotation(yaw=0.0))
    vehicle_bbox = carla.BoundingBox(carla.Location(x=0.0, y=0.0, z=0.5), carla.Vector3D(x=1.0, y=2.0, z=0.5))
    vehicle_velocity = carla.Vector3D(x=0.1, y=0.2, z=0.3)

    actor = _FakeVehicleActor(
        actor_id=203,
        location=vehicle_location,
        transform=vehicle_transform,
        bounding_box=vehicle_bbox,
        velocity=vehicle_velocity,
        type_id="vehicle.test.sumo_non_positive",
        attributes={},
    )

    monkeypatch.setattr(ov_mod, "get_speed_sumo", lambda _mapping, _carla_id: 0.0)
    ov = ov_mod.ObstacleVehicle(None, None, vehicle=actor, lidar=None, sumo2carla_ids={"sumo_1": 203})

    assert ov.velocity == vehicle_velocity


def test_set_vehicle_with_lidar_creates_aabb_with_expected_bounds_and_color(fake_open3d_aabb, monkeypatch):
    import carla

    from opencda.core.sensing.perception import obstacle_vehicle as ov_mod

    captured = {"calls": 0}

    def _world_to_sensor_assert_input(stack_boundary, _lidar_tf):
        assert isinstance(stack_boundary, np.ndarray)
        assert stack_boundary.shape == (4, 2)
        assert np.allclose(stack_boundary[3, :], 1.0)
        captured["calls"] += 1
        return stack_boundary

    monkeypatch.setattr(ov_mod.st, "world_to_sensor", _world_to_sensor_assert_input)

    vehicle_location = carla.Location(x=5.0, y=6.0, z=7.0)
    vehicle_transform = carla.Transform(vehicle_location, carla.Rotation(yaw=0.0))

    vehicle_bbox_location = carla.Location(x=0.0, y=0.0, z=0.5)
    vehicle_bbox_extent = carla.Vector3D(x=1.0, y=2.0, z=0.5)
    vehicle_bbox = carla.BoundingBox(vehicle_bbox_location, vehicle_bbox_extent)

    vehicle_velocity = carla.Vector3D(x=0.0, y=0.0, z=0.0)

    actor = _FakeVehicleActor(
        actor_id=303,
        location=vehicle_location,
        transform=vehicle_transform,
        bounding_box=vehicle_bbox,
        velocity=vehicle_velocity,
        type_id="vehicle.mercedes.coupe",
        attributes={},
    )

    lidar_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0), carla.Rotation(yaw=0.0))
    lidar = _FakeLidarSensor(transform=lidar_tf)

    ov = ov_mod.ObstacleVehicle(None, None, vehicle=actor, lidar=lidar, sumo2carla_ids={})

    assert captured["calls"] == 1

    assert hasattr(ov, "o3d_bbx")
    assert isinstance(ov.o3d_bbx, _FakeAxisAlignedBoundingBox)
    assert ov.o3d_bbx.color == (1, 0, 0)

    min_world = np.array(
        [
            vehicle_location.x - vehicle_bbox_extent.x,
            vehicle_location.y - vehicle_bbox_extent.y,
            vehicle_location.z + vehicle_bbox_location.z - vehicle_bbox_extent.z,
        ],
        dtype=float,
    )
    max_world = np.array(
        [
            vehicle_location.x + vehicle_bbox_extent.x,
            vehicle_location.y + vehicle_bbox_extent.y,
            vehicle_location.z + vehicle_bbox_location.z + vehicle_bbox_extent.z,
        ],
        dtype=float,
    )

    expected_min_sensor = np.array([-max_world[0], min_world[1], min_world[2]], dtype=float)
    expected_max_sensor = np.array([-min_world[0], max_world[1], max_world[2]], dtype=float)

    assert ov.o3d_bbx.min_bound == pytest.approx(expected_min_sensor)
    assert ov.o3d_bbx.max_bound == pytest.approx(expected_max_sensor)


def test_to_dict_returns_location_dict_and_carla_id_when_location_supports_to_dict():
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    corners = _make_corners(min_xyz=(0.0, 0.0, 0.0), max_xyz=(2.0, 4.0, 6.0))
    ov = ObstacleVehicle(corners, o3d_bbx=object())

    data = ov.to_dict()
    assert data["carla_id"] == -1
    assert data["location"] == {"x": 1.0, "y": 2.0, "z": 3.0}


def test_repr_and_str_include_location_and_id():
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    corners = _make_corners(min_xyz=(0.0, 0.0, 0.0), max_xyz=(2.0, 2.0, 2.0))
    ov = ObstacleVehicle(corners, o3d_bbx=object())

    rep = repr(ov)
    s = str(ov)

    assert "Location(" in rep
    assert ", -1" in rep

    assert "Location(" in s
    assert ", -1" in s
