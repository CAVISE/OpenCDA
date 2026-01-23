"""
Mock CARLA classes for unit testing.

This module provides lightweight mock implementations of CARLA classes
(Location, Transform, Rotation, Camera, LiDAR, Vehicle, etc.) to enable
unit testing without requiring the full CARLA simulator.
"""

import numpy as np

from typing import Dict


class Location(object):
    """
    Mock class for CARLA's Location.

    Represents a 3D point in the simulation world.

    Parameters
    ----------
    x : float
        X-coordinate (forward/backward).
    y : float
        Y-coordinate (left/right).
    z : float
        Z-coordinate (up/down).

    Attributes
    ----------
    x : float
        X-coordinate.
    y : float
        Y-coordinate.
    z : float
        Z-coordinate.
    """

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class Transform(object):
    """
    Mock class for CARLA's Transform.

    Combines location and rotation into a single transform.

    Parameters
    ----------
    x : float
        X-coordinate.
    y : float
        Y-coordinate.
    z : float
        Z-coordinate.
    pitch : float, optional
        Rotation around Y axis. Default is 0.
    yaw : float, optional
        Rotation around Z axis. Default is 0.
    roll : float, optional
        Rotation around X axis. Default is 0.

    Attributes
    ----------
    location : Location
        3D position.
    rotation : Rotation
        3D rotation.
    """

    def __init__(self, x: float, y: float, z: float, pitch: float = 0, yaw: float = 0, roll: float = 0):
        self.location = Location(x, y, z)
        self.rotation = Rotation(pitch, yaw, roll)


class Rotation(object):
    """
    Mock class for CARLA's Rotation.

    Represents a 3D rotation in the simulation world.

    Parameters
    ----------
    pitch : float
        Rotation around Y axis in degrees.
    yaw : float
        Rotation around Z axis in degrees.
    roll : float
        Rotation around X axis in degrees.

    Attributes
    ----------
    pitch : float
        Pitch rotation.
    yaw : float
        Yaw rotation.
    roll : float
        Roll rotation.
    """

    def __init__(self, pitch: float, yaw: float, roll: float):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class Vector3D(object):
    """
    Mock class for CARLA's Vector3D.

    Represents a 3D vector.

    Parameters
    ----------
    x : float
        X component.
    y : float
        Y component.
    z : float
        Z component.

    Attributes
    ----------
    x : float
        X component.
    y : float
        Y component.
    z : float
        Z component.
    """

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class Camera(object):
    """
    Mock class for CARLA's Camera sensor.

    Represents a camera sensor in the simulation world.

    Parameters
    ----------
    attributes : Dict[str, any]
        Camera configuration (image_size_x, image_size_y, fov).

    Attributes
    ----------
    attributes : Dict[str, any]
        Camera attributes.
    transform : Transform
        Camera's current transform.
    """

    def __init__(self, attributes: Dict):
        self.attributes = attributes
        self.transform = Transform(x=10, y=10, z=10)

    def get_transform(self):
        return self.transform


class Lidar(object):
    """
    Mock class for CARLA's LiDAR sensor.

    Parameters
    ----------
    attributes : Dict[str, any]
        LiDAR configuration (channels, range).

    Attributes
    ----------
    attributes : Dict[str, any]
        LiDAR attributes.
    transform : Transform
        LiDAR's current transform.
    """

    def __init__(self, attributes: Dict):
        self.attributes = attributes
        self.transform = Transform(x=11, y=11, z=11)

    def get_transform(self):
        """
        Get the current transform of the LiDAR.

        Returns
        -------
        Transform
            Current LiDAR transform.
        """
        return self.transform


class BoundingBox(object):
    """
    Mock class for CARLA's bounding box.

    Computes center location and extents from corner points.

    Parameters
    ----------
    corners : nd.nparray
        Eight corners of the bounding box with shape (8, 3).

    Attributes
    ----------
    location : Location
        Center location of the bounding box.
    transform : Transform
        Transform at the center of the bounding box.
    extent : Vector3D
        Half-extents (width, length, height) of the bounding box.
    """

    def __init__(self, corners):
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
    """
    Mock class for CARLA's vehicle.

    Represents a vehicle with a transform and bounding box.

    Attributes
    ----------
    transform : Transform
        Vehicle's current transform.
    bounding_box : BoundingBox
        Vehicle's bounding box.
    """

    def __init__(self):
        corner = np.random.random((8, 3))
        self.transform = Transform(x=12, y=12, z=12)
        self.bounding_box = BoundingBox(corner)

    def get_transform(self) -> Transform:
        """
        Get the current transform of the vehicle.

        Returns
        -------
        Transform
            Current vehicle transform.
        """
        return self.transform
