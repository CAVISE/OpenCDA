"""
Static obstacle base classes for 3D object representation.

This module provides classes for representing static obstacles including bounding
boxes, general static obstacles, and traffic lights in the CARLA simulator.
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import carla


class BoundingBox(object):
    """
    Bounding box class for obstacle representation.

    Computes the center location and extent of a 3D bounding box from its
    eight corner points.

    Parameters
    ----------
    corners : NDArray[np.float64]
        Eight corners of the bounding box with shape (8, 3).

    Attributes
    ----------
    location : carla.Location
        The center location of the bounding box.
    extent : carla.Vector3D
        The half-extents of the bounding box along x, y, z axes.
    """

    def __init__(self, corners: npt.NDArray[np.float64]):
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        center_z = np.mean(corners[:, 2])

        extent_x = (np.max(corners[:, 0]) - np.min(corners[:, 0])) / 2
        extent_y = (np.max(corners[:, 1]) - np.min(corners[:, 1])) / 2
        extent_z = (np.max(corners[:, 2]) - np.min(corners[:, 2])) / 2

        self.location = carla.Location(x=center_x, y=center_y, z=center_z)
        self.extent = carla.Vector3D(x=extent_x, y=extent_y, z=extent_z)


class StaticObstacle(object):
    """
    The general class for obstacles. Currently, we regard all static obstacles
     such as stop signs and traffic light as the same class.

    Parameters
    ----------
    corner : nd.nparray
        Eight corners of the bounding box (shape:(8, 3)).
    o3d_bbx : open3d.AlignedBoundingBox
        The bounding box object in Open3d.This is
        mainly used for visualization.

    Attributes
    ----------
    bounding_box : BoundingBox
        Bounding box of the osbject vehicle.
    """

    def __init__(self, corner: npt.NDArray[np.float64], o3d_bbx: Any):
        self.bounding_box = BoundingBox(corner)
        self.o3d_bbx = o3d_bbx


class TrafficLight(object):
    """
    The class for traffic light. Currently, we retrieve the traffic light info
    from the server directly and assign to this class.

    Parameters
    ---------
    tl : carla.Actor
        The CARLA traffic actor

    trigger_location : carla.Vector3D
        The trigger location of te traffic light.

    pos : carla.Location
        The location of this traffic light.

    light_state : carla.TrafficLightState
        Current state of the traffic light.

    """

    def __init__(self, tl: carla.Actor, trigger_location: carla.Vector3D, light_state: carla.TrafficLightState) -> None:
        self._location = trigger_location
        self.state = light_state
        self.actor = tl

    def get_location(self) -> carla.Vector3D:
        return self._location

    def get_state(self) -> carla.TrafficLightState:
        return self.state

    @staticmethod
    def get_trafficlight_trigger_location(traffic_light: carla.Actor) -> carla.Vector3D:  # pylint: disable=invalid-name
        """
        Calculate the trigger location of a traffic light.

        Computes the location that represents the trigger volume of the traffic
        light by rotating and transforming the trigger volume relative to the
        traffic light's base transform.

        Parameters
        ----------
        traffic_light : carla.Actor
            The CARLA traffic light actor.

        Returns
        -------
        carla.Location
            The calculated trigger location.
        """

        def rotate_point(point: carla.Vector3D, angle: float) -> carla.Vector3D:
            """
            Rotate a point by a given angle around the z-axis.

            Parameters
            ----------
            point : carla.Vector3D
                The point to rotate.
            angle : float
                The rotation angle in degrees.

            Returns
            -------
            carla.Vector3D
                The rotated point.
            """
            x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
            y_ = math.sin(math.radians(angle)) * point.x - math.cos(math.radians(angle)) * point.y

            return carla.Vector3D(x_, y_, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)
