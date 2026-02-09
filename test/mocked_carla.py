"""Lightweight CARLA API mocks used by unit tests.

This module is reused across the test-suite. It must remain backward compatible.

Compatibility requirements:
- Keep Camera and Lidar classes (used by existing tests).
- Transform supports BOTH signatures:
  1) Transform(location, rotation)  # new style
  2) Transform(x, y, z, pitch=0, yaw=0, roll=0)  # old style
- BoundingBox supports BOTH signatures:
  1) BoundingBox(corners)  # corners is array-like (8,3)
  2) BoundingBox(location, extent)  # new style
- Vehicle uses deterministic default corners (no flaky tests).
"""

from __future__ import annotations

import math

import numpy as np


class _FloatAttrMixin:
    """Mixin providing __eq__ with math.isclose for float attributes."""

    _cmp_attrs: tuple = ()
    _rel_tol: float = 1e-9
    _abs_tol: float = 1e-9

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return all(
            math.isclose(
                float(getattr(self, attr)),
                float(getattr(other, attr)),
                rel_tol=self._rel_tol,
                abs_tol=self._abs_tol,
            )
            for attr in self._cmp_attrs
        )


class Location(_FloatAttrMixin):
    """A mock class for carla.Location."""

    _cmp_attrs = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        if not isinstance(other, Location):
            return NotImplemented
        return Location(self.x + other.x, self.y + other.y, self.z + other.z)

    def __repr__(self):
        return f"Location(x={self.x}, y={self.y}, z={self.z})"


class Rotation(_FloatAttrMixin):
    """A mock class for carla.Rotation."""

    _cmp_attrs = ("pitch", "yaw", "roll")

    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    def __repr__(self):
        return f"Rotation(pitch={self.pitch}, yaw={self.yaw}, roll={self.roll})"


class Transform:
    """A mock class for carla.Transform.

    Supports both:
    - Transform(location, rotation)
    - Transform(x, y, z, pitch=0, yaw=0, roll=0)
    """

    def __init__(self, *args, **kwargs):
        # New style: Transform(location, rotation)
        if len(args) == 2 and isinstance(args[0], Location) and isinstance(args[1], Rotation):
            # Strict contract: new-style signature does not accept any kwargs.
            if kwargs:
                raise TypeError(f"Unexpected keyword arguments for Transform(location, rotation): {list(kwargs.keys())}")
            self.location = args[0]
            self.rotation = args[1]
            return

        # Old style: Transform(x, y, z, pitch=0, yaw=0, roll=0)
        if len(args) in {0, 3}:
            if len(args) == 0:
                x = kwargs.pop("x", 0.0)
                y = kwargs.pop("y", 0.0)
                z = kwargs.pop("z", 0.0)
            else:
                x, y, z = args

            pitch = kwargs.pop("pitch", 0.0)
            yaw = kwargs.pop("yaw", 0.0)
            roll = kwargs.pop("roll", 0.0)

            # Validate no unexpected kwargs
            if kwargs:
                raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

            self.location = Location(x, y, z)
            self.rotation = Rotation(pitch=pitch, yaw=yaw, roll=roll)
            return

        raise TypeError("Unsupported Transform signature. Use Transform(location, rotation) or Transform(x, y, z, pitch=0, yaw=0, roll=0).")

    def __repr__(self):
        return f"Transform(location={self.location!r}, rotation={self.rotation!r})"

    def __eq__(self, other):
        if not isinstance(other, Transform):
            return False
        return self.location == other.location and self.rotation == other.rotation


class Vector3D(_FloatAttrMixin):
    """A mock class for carla.Vector3D."""

    _cmp_attrs = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"


# Default corners for deterministic BoundingBox (unit cube centered at origin)
_DEFAULT_CORNERS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)


class BoundingBox:
    """A mock class for carla.BoundingBox.

    Supports:
    - BoundingBox(corners) where corners is array-like of shape (8, 3)
    - BoundingBox(location, extent)
    - BoundingBox() - default empty bbox
    """

    def __init__(self, *args):
        self.corners = None

        # Default empty bbox
        if len(args) == 0:
            self.location = Location()
            self.extent = Vector3D()
            return

        if len(args) == 1:
            corners = args[0]
            if not isinstance(corners, np.ndarray):
                corners = np.asarray(corners, dtype=np.float64)
            if corners.shape != (8, 3):
                raise ValueError(f"corners must have shape (8, 3), got {corners.shape}")
            self.corners = corners
            center = corners.mean(axis=0)
            mins = corners.min(axis=0)
            maxs = corners.max(axis=0)
            extent = (maxs - mins) / 2.0
            self.location = Location(float(center[0]), float(center[1]), float(center[2]))
            self.extent = Vector3D(float(extent[0]), float(extent[1]), float(extent[2]))
            return

        if len(args) == 2 and isinstance(args[0], Location) and isinstance(args[1], Vector3D):
            self.location = args[0]
            self.extent = args[1]
            return

        raise TypeError("Unsupported BoundingBox signature. Use BoundingBox(corners) or BoundingBox(location, extent).")

    def __repr__(self):
        return f"BoundingBox(location={self.location!r}, extent={self.extent!r})"

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            return False
        return self.location == other.location and self.extent == other.extent


class Vehicle:
    """A mock class for carla.Vehicle.

    Parameters
    ----------
    actor_id : int
        Vehicle actor ID.
    corners : array-like, optional
        Custom bounding box corners (8, 3). If None, uses deterministic default.
    seed : int, optional
        If provided and corners is None, generates random corners with this seed.
    """

    def __init__(self, actor_id=1, corners=None, seed=None):
        self.id = actor_id

        if corners is not None:
            bbox_corners = np.asarray(corners, dtype=np.float64)
        elif seed is not None:
            rng = np.random.default_rng(seed)
            bbox_corners = rng.random((8, 3))
        else:
            bbox_corners = _DEFAULT_CORNERS.copy()

        self.bounding_box = BoundingBox(bbox_corners)
        self.transform = Transform(x=12, y=12, z=12)

    def get_transform(self):
        return self.transform

    def get_world(self):
        return None

    def destroy(self):
        return True


class Camera:
    """A minimal mock for a CARLA camera sensor."""

    def __init__(self, attributes=None):
        self.attributes = attributes if attributes is not None else {}
        self.transform = Transform(x=10, y=10, z=10)
        self._callback = None
        self.is_listening = False

    def get_transform(self):
        return self.transform

    def listen(self, callback):
        self._callback = callback
        self.is_listening = True

    def stop(self):
        self.is_listening = False

    def trigger(self, data):
        """Test helper: manually trigger callback."""
        if self._callback and self.is_listening:
            self._callback(data)

    def destroy(self):
        self.stop()
        return True


class Lidar:
    """A minimal mock for a CARLA lidar sensor."""

    def __init__(self, attributes=None):
        self.attributes = attributes if attributes is not None else {}
        self.transform = Transform(x=11, y=11, z=11)
        self._callback = None
        self.is_listening = False

    def get_transform(self):
        return self.transform

    def listen(self, callback):
        self._callback = callback
        self.is_listening = True

    def stop(self):
        self.is_listening = False

    def trigger(self, data):
        """Test helper: manually trigger callback."""
        if self._callback and self.is_listening:
            self._callback(data)

    def destroy(self):
        self.stop()
        return True
