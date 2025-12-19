# -*- coding: utf-8 -*-
"""
Mock Carla for unit tests.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: MIT

import numpy as np


class Location(object):
    """A mock class for CARLA's Location.
    This class represents a 3D point in the simulation world.
    Args:
        x: X-coordinate (forward/backward)
        y: Y-coordinate (left/right)
        z: Z-coordinate (up/down)
    """
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class Transform(object):
    """A mock class for CARLA's Transform.
    This class combines location and rotation into a single transform.
    Args:
        x: X-coordinate
        y: Y-coordinate
        z: Z-coordinate
        pitch: Rotation around Y axis (default: 0)
        yaw: Rotation around Z axis (default: 0)
        roll: Rotation around X axis (default: 0)
    """
    def __init__(self, x: float, y: float, z: float, pitch: float = 0, yaw: float = 0, roll: float = 0) -> None:
        self.location = Location(x, y, z)
        self.rotation = Rotation(pitch, yaw, roll)


class Rotation(object):
    """A mock class for CARLA's Rotation.
    This class represents a 3D rotation in the simulation world.
    Args:
        pitch: Rotation around Y axis
        yaw: Rotation around Z axis
        roll: Rotation around X axis
    """
    def __init__(self, pitch: float, yaw: float, roll: float) -> None:
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class Vector3D(object):
    """A mock class for CARLA's Vector3D.
        This class represents a 3D vector.
        Args:
            x: X component
            y: Y component
            z: Z component
        """
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class Camera(object):
    """A mock class for CARLA's Camera.
    This class represents a camera sensor in the simulation world.
    Args:
        attributes: Dictionary of camera attributes
    """
    def __init__(self, attributes: dict):
        self.attributes = attributes
        self.transform = Transform(x=10, y=10, z=10)

    def get_transform(self):
        return self.transform


class Lidar(object):
    """A mock class for CARLA's LiDAR sensor.
    Args:
        attributes: Dictionary containing LiDAR attributes
    """
    def __init__(self, attributes: dict):
        self.attributes = attributes
        self.transform = Transform(x=11, y=11, z=11)

    def get_transform(self):
        """Get the current transform of the LiDAR.
        Returns:
            Transform: The current transform of the LiDAR
        """
        return self.transform


class BoundingBox(object):
    """A mock class for CARLA's bounding box.
    """

    def __init__(self, corners):
        """
        Construct class.
        Args:
            corners (nd.nparray): Eight corners of the bounding box. shape:(8, 3)
        """
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        center_z = np.mean(corners[:, 2])

        extent_x = (np.max(corners[:, 0]) - np.min(corners[:, 0])) / 2
        extent_y = (np.max(corners[:, 1]) - np.min(corners[:, 1])) / 2
        extent_z = (np.max(corners[:, 2]) - np.min(corners[:, 2])) / 2

        self.location = Location(x=center_x, y=center_y, z=center_z)
        self.transform = Transform(x=center_x, y=center_y, z=center_z)
        self.extent = Vector3D(x=extent_x, y=extent_y, z=extent_z)


class Vehicle(object):
    """A mock class for CARLA's vehicle.
    This class represents a vehicle in the simulation with a transform and bounding box.
    """

    def __init__(self):
        corner = np.random.random((8, 3))
        self.transform = Transform(x=12, y=12, z=12)
        self.bounding_box = BoundingBox(corner)

    def get_transform(self):
        """Get the current transform of the vehicle."""
        return self.transform
