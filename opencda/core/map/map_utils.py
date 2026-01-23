"""
HDMap utility functions.

This module provides utility functions for handling high-definition map data,
including coordinate conversions, waypoint transformations, and traffic light
status conversions for CARLA simulation.
"""

from typing import List
import carla
import numpy as np
import numpy.typing as npt


def lateral_shift(transform: carla.Transform, shift: float) -> carla.Location:
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


def list_loc2array(list_location: List[carla.Location]) -> npt.NDArray[np.float64]:
    """
    Convert list of carla location to np.array

    Parameters
    ----------
    list_location : list
        List of carla locations.

    Returns
    -------
    loc_array : np.array
        Numpy array of shape (N, 3)
    """
    loc_array = np.zeros((len(list_location), 3))
    for i, carla_location in enumerate(list_location):
        loc_array[i, 0] = carla_location.x
        loc_array[i, 1] = carla_location.y
        loc_array[i, 2] = carla_location.z

    return loc_array


def list_wpt2array(list_wpt: List[carla.Waypoint]) -> npt.NDArray[np.float64]:
    """
    Convert list of carla transform to np.array

    Parameters
    ----------
    list_wpt : list
        List of carla waypoint.

    Returns
    -------
    loc_array : np.array
        Numpy array of shape (N, 3)
    """
    loc_array = np.zeros((len(list_wpt), 3))
    for i, carla_wpt in enumerate(list_wpt):
        loc_array[i, 0] = carla_wpt.transform.location.x
        loc_array[i, 1] = carla_wpt.transform.location.y
        loc_array[i, 2] = carla_wpt.transform.location.z

    return loc_array


def convert_tl_status(status: carla.TrafficLightState) -> str:
    """
    Convert carla.TrafficLightState to str.

    Parameters
    ----------
    status : carla.TrafficLightState

    Returns
    -------
    status_str : str
    """
    if status == carla.TrafficLightState.Red:
        return "red"
    elif status == carla.TrafficLightState.Green:
        return "green"
    elif status == carla.TrafficLightState.Yellow:
        return "yellow"
    else:
        return "normal"
