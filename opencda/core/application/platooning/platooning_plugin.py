# -*- coding: utf-8 -*-
"""
Platooning plugin for V2X communication and finite state machine management.

This module provides the platooning application plugin that manages platoon
membership, status tracking, and vehicle coordination within a platoon through
V2X communication.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import warnings
from typing import Optional, Dict, List, Tuple, Any

from opencda.core.common.misc import compute_distance, cal_distance_angle
from opencda.core.application.platooning.fsm import FSM


class PlatooningPlugin(object):
    """
    Platooning plugin for V2X manager.

    Manages platoon membership, finite state machine transitions, and
    vehicle-to-vehicle coordination for cooperative adaptive cruise control
    in platoons.

    Parameters
    ----------
    search_range : float
        Search range of the communication equipment in meters.
    cda_enabled : bool
        Whether cooperative driving automation connectivity is supported.

    Attributes
    ----------
    search_range : float
        Maximum communication range for platoon searching.
    cda_enabled : bool
        Connectivity support flag.
    leader : bool
        True if this vehicle is the platoon leader.
    platooning_object : Any or None
        Current platoon object reference.
    platooning_id : int or None
        ID of the current platoon.
    in_id : int or None
        Position index within the platoon.
    status : str or None
        Current platooning FSM status.
    ego_pos : carla.Transform or None
        Current position and rotation of the ego vehicle.
    ego_spd : float or None
        Current speed of the ego vehicle in km/h.
    platooning_blacklist : List[int]
        List of platoon IDs that won't be considered for joining.
    front_vehicle : Any or None
        Vehicle manager of the vehicle in front.
    rear_vechile : Any or None
        Vehicle manager of the vehicle behind.
    """

    def __init__(self, search_range: float, cda_enabled: bool):
        self.search_range = search_range
        self.cda_enabled = cda_enabled

        # whether leader in a platoon
        self.leader = False
        self.platooning_object = None
        self.platooning_id = None
        self.in_id = None
        self.status = None

        # ego speed and position
        self.ego_pos = None
        self.ego_spd = None

        # the platoon in the black list won't be considered again
        self.platooning_blacklist = []

        # used to label the front and rear vehicle position
        self.front_vehicle = None
        self.rear_vechile = None

    def update_info(self, ego_pos: Any, ego_spd: float) -> None:
        """
        Update ego vehicle position and speed.

        Parameters
        ----------
        ego_pos : carla.Transform
            Ego vehicle pose (location and rotation).
        ego_spd : float
            Ego vehicle speed in km/h.
        """
        self.ego_pos = ego_pos
        self.ego_spd = ego_spd

    def reset(self) -> None:
        """
        Reset platooning plugin to initial state.

        Clears platoon membership, leader status, and neighboring vehicle
        references.
        """
        self.front_vehicle = None
        self.rear_vechile = None

        self.leader = False
        self.platooning_object = None
        self.platooning_id = None
        self.in_id = None

    def set_platoon(
        self,
        in_id: Optional[int],
        platooning_object: Optional[Any] = None,
        platooning_id: Optional[int] = None,
        leader: bool = False,
    ) -> None:
        """
        Set platoon membership status.

        Parameters
        ----------
        in_id : int or None
            Position index within the platoon. None to start searching.
        platooning_object : Any, optional
            Current platoon object reference. Default is None.
        platooning_id : int, optional
            Current platoon ID. Default is None.
        leader : bool, optional
            Whether this vehicle is the platoon leader. Default is False.
        """
        if in_id is None:
            if not self.cda_enabled:
                self.set_status(FSM.DISABLE)
                warnings.warn("CDA feature is disabled, can not activate platooning application ")
            else:
                self.set_status(FSM.SEARCHING)
            return

        if platooning_object:
            self.platooning_object = platooning_object
        if platooning_id:
            self.platooning_id = platooning_id

        self.in_id = in_id
        if leader:
            self.leader = leader
            self.set_status(FSM.LEADING_MODE)
        else:
            self.set_status(FSM.MAINTINING)

    def set_status(self, status: str) -> None:
        """
        Set finite state machine status.

        Parameters
        ----------
        status : str
            Current platooning FSM status (e.g., SEARCHING, LEADING_MODE,
            MAINTINING, DISABLE).
        """
        self.status = status

    def search_platoon(self, ego_loc: Any, cav_nearby: Dict[Any, Any]) -> Tuple[Optional[int], Optional[Any]]:
        """
        Search for platoon candidates within communication range.

        Parameters
        ----------
        ego_loc : carla.Location
            Ego vehicle current location.
        cav_nearby : Dict[Any, Any]
            Dictionary of nearby connected and automated vehicles.

        Returns
        -------
        pmid : int or None
            Platoon manager ID of the closest platoon, or None if not found.
        pm : Any or None
            Platoon manager object of the closest platoon, or None if not found.
        """
        pm = None
        pmid = None
        min_dist = 1000

        for _, vm in cav_nearby.items():
            if vm.v2x_manager.in_platoon is None:
                continue

            platoon_manager, _ = vm.v2x_manager.get_platoon_manager()
            if pmid and pmid == platoon_manager.pmid:
                continue

            distance = compute_distance(ego_loc, vm.v2x_manager.get_ego_pos().location)
            if distance < min_dist:
                pm = platoon_manager
                pmid = platoon_manager.pmid
                min_dist = distance

        return pmid, pm

    def match_platoon(self, cav_nearby: Dict[Any, Any]) -> Tuple[bool, int, List[Any]]:
        """
        Find the best position to join a platoon.

        Uses a naive matching algorithm to determine the optimal insertion
        point in a nearby platoon based on distance and angle.

        Parameters
        ----------
        cav_nearby : Dict[Any, Any]
            Dictionary of nearby connected and automated vehicles.

        Returns
        -------
        matched : bool
            True if a suitable platoon position was found.
        min_index : int
            Index position in the platoon to join (-1 if no match).
        platoon_vehicle_list : List[Any]
            List of vehicle managers in the matched platoon.
        """

        # make sure the previous status won't influence current one
        self.reset()

        cur_loc = self.ego_pos.location
        cur_yaw = self.ego_pos.rotation.yaw

        pmid, pm = self.search_platoon(cur_loc, cav_nearby)

        if not pmid or pmid in self.platooning_blacklist:
            return False, -1, []

        # used to search the closest platoon member in the searched platoon
        min_distance = float("inf")
        min_index = -1
        min_angle = 0

        # if the platooning is not open to joining
        if not pm.response_joining_request(self.ego_pos.location):
            return False, -1, []

        platoon_vehicle_list = []

        for i, vehicle_manager in enumerate(pm.vehicle_manager_list):
            distance, angle = cal_distance_angle(vehicle_manager.vehicle.get_location(), cur_loc, cur_yaw)
            platoon_vehicle_list.append(vehicle_manager)

            if distance < min_distance:
                min_distance = distance
                min_index = i
                min_angle = angle

        # if the ego is in front of the platooning
        if min_index == 0 and min_angle > 90:
            self.front_vehicle = None
            self.rear_vechile = pm.vehicle_manager_list[0]
            return True, min_index, platoon_vehicle_list

        self.front_vehicle = pm.vehicle_manager_list[min_index]

        if min_index < len(pm.vehicle_manager_list) - 1:
            self.rear_vechile = pm.vehicle_manager_list[min_index + 1]

        return True, min_index, platoon_vehicle_list
