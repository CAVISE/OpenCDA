"""Module with auxiliary functions."""

import math
import importlib
from typing import Dict, List, Tuple, Union
import numpy as np
import carla


def draw_trajetory_points(
    world: carla.World,
    waypoints: List[Union[carla.Waypoint, carla.Transform, Tuple[carla.Waypoint, carla.Transform], List[carla.Transform]]],
    z: float = 0.25,
    color: carla.Color = carla.Color(255, 0, 0),
    lt: float = 5,
    size: float = 0.1,
    arrow_size: float = 0.1,
) -> None:
    """
    Draw a list of trajectory points

    Parameters
    ----------
    size : float
        Time step between updating visualized waypoint.
    lt : int
        Number of waypoints being visualized.
    color : carla.Color
        The trajectory color.
    world : carla.world
        The simulation world.
    waypoints : list
        The waypoints of the current plan.
    z : float
        The height of the visualized waypoint.
    """
    for i in range(len(waypoints)):
        wpt = waypoints[i]
        if isinstance(wpt, tuple) or isinstance(wpt, list):
            wpt = wpt[0]
        if hasattr(wpt, "is_junction"):
            wpt_t = wpt.transform
        else:
            wpt_t = wpt

        world.debug.draw_point(wpt_t.location, size=size, color=color, life_time=lt)


def get_speed(vehicle: carla.Vehicle, meters: bool = False) -> float:
    """
    Compute speed of a vehicle in Km/h.

    Parameters
    ----------
    meters : bool
        Whether to use m/s (True) or km/h (False).
    vehicle : carla.vehicle
        The vehicle for which speed is calculated.

    Returns
    -------
    speed : float
        The vehicle speed.
    """
    vel = vehicle.get_velocity()
    vel_meter_per_second = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    return vel_meter_per_second if meters else 3.6 * vel_meter_per_second


def cal_distance_angle(target_location: carla.Location, current_location: carla.Location, orientation: carla.Rotation) -> Tuple[float, float]:
    """
    Calculate the vehicle current relative distance to target location.

    Parameters
    ----------
    target_location : carla.Location
        The target location.
    current_location : carla.Location
        The current location .
    orientation : carla.Rotation
        Orientation of the reference object.

    Returns
    -------
    distance : float
        The measured distance from current location to target location.
    d_angle : float)
        The measured rotation (angle) froM current location
        to target location.
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector) + 1e-10

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1.0, 1.0)))

    return norm_target, d_angle


def distance_vehicle(waypoint: carla.Waypoint, vehicle_transform: carla.transform) -> float:
    """
    Returns the 2D distance from a waypoint to a vehicle

    Parameters
    ----------
    waypoint : carla.Waypoint
        Actual waypoint.
    vehicle_transform : carla.transform
        Transform of the target vehicle.

    Returns
    -------
    float
        2D Euclidean distance between waypoint and vehicle.
    """
    loc = vehicle_transform.location
    if hasattr(waypoint, "is_junction"):
        x = waypoint.transform.location.x - loc.x
        y = waypoint.transform.location.y - loc.y
    else:
        x = waypoint.location.x - loc.x
        y = waypoint.location.y - loc.y

    return math.sqrt(x * x + y * y)


def vector(location_1: carla.Location, location_2: carla.Location) -> List[float]:
    """
    Returns the unit vector from location_1 to location_2.

    Parameters
    ----------
    location_1 : carla.location
        Start location of the vector.
    location_2 : carla.location
        End location of the vector.

    Returns
    -------
    List[float]
        Unit vector as [x, y, z] components.
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def compute_distance(location_1: carla.Location, location_2: carla.Location) -> float:
    """
    Compute Euclidean distance between 3D points.

    Parameters
    ----------
    location_1 : carla.Location
        Start point of the measurement.
    location_2 : carla.Location
        End point of the measurement.

    Returns
    -------
    float
        Euclidean distance between the two locations.
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def positive(num: float) -> float:
    """
    Return the given number if positive, else 0.

    Parameters
    ----------
    num : float
        Input number.

    Returns
    -------
    float
        The number if positive, otherwise 0.0.
    """
    return num if num > 0.0 else 0.0


def get_speed_sumo(sumo2carla_ids: Dict[str, int], carla_id: int) -> float:
    """
    Get the speed of the vehicles controlled by sumo.

    Parameters
    ----------
    sumo2carla_ids : Dict[str, int]
        Sumo-carla mapping dictionary.
    carla_id : int
        Carla actor id.

    Returns
    -------
    float
        The speed retrieved from the sumo server, -1 if the carla_id not found.
    """
    # python will only import this once and then save it in cache. so the
    # efficiency won't affected during the loop.
    traci = importlib.import_module("traci")

    for key, value in sumo2carla_ids.items():
        if int(value) == carla_id:
            vehicle_speed = traci.vehicle.getSpeed(key)
            return vehicle_speed

    return -1
