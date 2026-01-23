"""
Global route planner data access object.

This module provides the data access layer for fetching topology and waypoint
data from the CARLA server instance for use by the GlobalRoutePlanner.
"""

from typing import List, Dict, Any
import numpy as np


class GlobalRoutePlannerDAO(object):
    """
    Data access layer for GlobalRoutePlanner.

    This class fetches data from the CARLA server instance, including map
    topology and waypoint information for route planning.

    Parameters
    ----------
    wmap : carla.Map
        The CARLA world map object.
    sampling_resolution : float
        Sampling distance between waypoints in meters.

    Attributes
    ----------
    _sampling_resolution : float
        Sampling distance between consecutive waypoints.
    _wmap : carla.Map
        The CARLA world map reference.
    """

    def __init__(self, wmap: Any, sampling_resolution: float):
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap

    def get_topology(self) -> List[Dict[str, Any]]:
        """
        Retrieve and process map topology.

        This function retrieves topology from the server as a list of road
        segments (pairs of waypoint objects), and processes them into a list
        of dictionary objects with detailed segment information.

        Returns
        -------
        List[Dict[str, Any]]
            List of topology dictionaries, each containing:
            - entry : carla.Waypoint
                Waypoint at entry point of road segment.
            - entryxyz : Tuple[float, float, float]
                (x, y, z) coordinates of entry point.
            - exit : carla.Waypoint
                Waypoint at exit point of road segment.
            - exitxyz : Tuple[float, float, float]
                (x, y, z) coordinates of exit point.
            - path : List[carla.Waypoint]
                List of waypoints separated by sampling_resolution meters
                from entry to exit.
        """
        topology = []
        # Retrieving waypoints to construct a detailed topology
        for segment in self._wmap.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            # Rounding off to avoid floating point imprecision
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            seg_dict = dict()
            seg_dict["entry"], seg_dict["exit"] = wp1, wp2
            seg_dict["entryxyz"], seg_dict["exitxyz"] = (x1, y1, z1), (x2, y2, z2)
            seg_dict["path"] = []
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict["path"].append(w)
                    w = w.next(self._sampling_resolution)[0]
            else:
                seg_dict["path"].append(wp1.next(self._sampling_resolution)[0])
            topology.append(seg_dict)
        return topology

    def get_waypoint(self, location: Any) -> Any:
        """
        Get waypoint at specified location.

        Parameters
        ----------
        location : carla.Location
            Vehicle location in the world.

        Returns
        -------
        carla.Waypoint
            Waypoint object closest to the specified location.
        """
        waypoint = self._wmap.get_waypoint(location)
        return waypoint

    def get_resolution(self) -> float:
        """
        Get the sampling resolution.

        Returns
        -------
        float
            Sampling distance between waypoints in meters.
        """
        return self._sampling_resolution
