"""
Platooning management for cooperative driving.

This module provides platooning management functionality for coordinating
multiple vehicles in a platoon formation, including leader-follower dynamics,
member management, and performance evaluation.
"""

import uuid
import weakref
import logging
from typing import List, Any

import carla

import numpy as np
import matplotlib.pyplot as plt

import opencda.core.plan.drive_profile_plotting as open_plt


logger = logging.getLogger("cavise.platooning_manager")


class PlatooningManager(object):
    """
    Platoon manager for coordinating vehicle managers.

    Manages all vehicle managers within a platoon, handling member coordination,
    speed control, joining requests, and performance evaluation.

    Parameters
    ----------
    config_yaml : dict
        Configuration dictionary for platoon parameters.
    cav_world : Any
        CAV world object that stores all CAV information.

    Attributes
    ----------
    pmid : str
        Unique platooning manager ID.
    vehicle_manager_list : List[Any]
        List of all vehicle managers within the platoon.
    maximum_capacity : int
        Maximum number of vehicles allowed in the platoon.
    destination : carla.Location or None
        Destination of the current plan.
    center_loc : carla.Location or None
        Center location of the platoon.
    leader_target_speed : float
        Current target speed of the leader vehicle.
    origin_leader_target_speed : float
        Original planned target speed of the platoon leader.
    recover_speed_counter : int
        Counter recording number of speed recovery attempts.
    cav_world : Any
        Reference to the CAV world object.
    """

    def __init__(self, config_yaml: dict, cav_world: Any):
        self.pmid = str(uuid.uuid1())

        self.vehicle_manager_list = []
        self.maximum_capacity = config_yaml["max_capacity"]

        self.destination = None
        self.center_loc = None  # noqa: DC05

        # this is used to control platooning speed during joining
        self.leader_target_speed = 0
        self.origin_leader_target_speed = 0
        self.recover_speed_counter = 0

        cav_world.update_platooning(self)
        self.cav_world = weakref.ref(cav_world)()

    def set_lead(self, vehicle_manager: Any) -> None:
        """
        Set the leader of the platoon.

        Parameters
        ----------
        vehicle_manager : Any
            The vehicle manager to be designated as leader.
        """
        self.add_member(vehicle_manager, leader=True)

        # this variable is used to control leader speed
        self.origin_leader_target_speed = vehicle_manager.agent.max_speed - vehicle_manager.agent.speed_lim_dist

    def add_member(self, vehicle_manager: Any, leader: bool = False) -> None:
        """
        Add member to the current platoon.

        Parameters
        ----------
        vehicle_manager : Any
            The vehicle manager to add to the platoon.
        leader : bool, optional
            Indicator of whether this CAV is a leader. Default is False.
        """
        self.vehicle_manager_list.append(vehicle_manager)
        vehicle_manager.v2x_manager.set_platoon(len(self.vehicle_manager_list) - 1, platooning_object=self, platooning_id=self.pmid, leader=leader)

    def set_member(self, vehicle_manager: Any, index: int, lead: bool = False) -> None:
        """
        Set member at specific index in the platoon.

        Parameters
        ----------
        vehicle_manager : Any
            The vehicle manager to insert.
        index : int
            The platoon index position for the vehicle.
        lead : bool, optional
            Indicator of whether this CAV is a leader. Default is False.
        """
        self.vehicle_manager_list.insert(index, vehicle_manager)
        vehicle_manager.v2x_manager.set_platoon(index, platooning_object=self, platooning_id=self.pmid, leader=lead)

    def cal_center_loc(self):
        """
        Calculate and update center location of the platoon.

        The center location is computed as the midpoint between the first
        and last vehicle in the platoon.
        """
        v1_ego_transform = self.vehicle_manager_list[0].v2x_manager.get_ego_pos()
        v2_ego_transform = self.vehicle_manager_list[-1].v2x_manager.get_ego_pos()

        if any(t is None for t in (v1_ego_transform, v2_ego_transform)):
            return
        self.center_loc = carla.Location(  # noqa: DC05
            x=(v1_ego_transform.location.x + v2_ego_transform.location.x) / 2,
            y=(v1_ego_transform.location.y + v2_ego_transform.location.y) / 2,
            z=(v1_ego_transform.location.z + v2_ego_transform.location.z) / 2,
        )

    def update_member_order(self):
        """
        Update member front and rear vehicle relationships.

        This method should be called whenever a new member is added to the
        platoon list to maintain correct leader-follower relationships.
        """
        for i, vm in enumerate(self.vehicle_manager_list):
            if i != 0:
                vm.v2x_manager.set_platoon(i, leader=False)
                vm.v2x_manager.set_platoon_front(self.vehicle_manager_list[i - 1])
            if i != len(self.vehicle_manager_list) - 1:
                leader = True if i == 0 else False
                vm.v2x_manager.set_platoon(i, leader=leader)
                vm.v2x_manager.set_platoon_rear(self.vehicle_manager_list[i + 1])

    def reset_speed(self) -> None:
        """
        Reset platoon speed to original after joining request.

        After a joining request is accepted for a certain number of steps,
        the platoon returns to its original speed..
        """
        if self.recover_speed_counter <= 0:
            self.leader_target_speed = self.origin_leader_target_speed
        else:
            self.recover_speed_counter -= 1

    def response_joining_request(self, request_loc: carla.Location) -> bool:
        """
        Process joining request based on capacity and location.

        Parameters
        ----------
        request_loc : carla.Location
            Location of the requesting vehicle.

        Returns
        -------
        bool
            True if joining request is accepted, False otherwise.

        """
        if len(self.vehicle_manager_list) >= self.maximum_capacity:
            return False
        else:
            # when the platoon accept a joining request,by default
            # it will decrease the speed so the merging vehicle
            # can better catch up with
            self.leader_target_speed = self.origin_leader_target_speed - 5
            self.recover_speed_counter = 200

            # find the corresponding vehicle manager and add it to the leader's
            # whitelist
            request_vm = self.cav_world.locate_vehicle_manager(request_loc)
            self.vehicle_manager_list[0].agent.add_white_list(request_vm)

            return True

    def set_destination(self, destination: carla.Location) -> None:
        """
        Set destination for all vehicle managers in the platoon.

        Parameters
        ----------
        destination : carla.Location
            Target destination location.
        """
        self.destination = destination
        for i in range(len(self.vehicle_manager_list)):
            self.vehicle_manager_list[i].set_destination(self.vehicle_manager_list[i].vehicle.get_location(), destination, clean=True)

    def update_information(self) -> None:
        """
        Update CAV world information for every member in the platoon.

        This method updates speed settings, member information, and
        recalculates the center location of the platoon.
        """
        self.reset_speed()
        for i in range(len(self.vehicle_manager_list)):
            self.vehicle_manager_list[i].update_info()
        # update the center location of the platoon
        self.cal_center_loc()

    def run_step(self) -> List[carla.VehicleControl]:
        """
        Execute one control step for each vehicle in the platoon.

        Returns
        -------
        List[carla.VehicleControl]
            List of control commands for all vehicles.
        """
        control_list = []
        for i in range(len(self.vehicle_manager_list)):
            control = self.vehicle_manager_list[i].run_step(self.leader_target_speed)
            control_list.append(control)

        for i, control in enumerate(control_list):
            self.vehicle_manager_list[i].vehicle.apply_control(control)

        return control_list

    def evaluate(self):
        """
        Evaluate and save statistics for all platoon members.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure with performance curves including velocity, acceleration,
            time gap, and distance gap profiles.
        perform_txt : str
            String containing all evaluation results with mean and standard
            deviation statistics for each member.
        """

        velocity_list = []
        acceleration_list = []
        time_gap_list = []
        distance_gap_list = []

        perform_txt = ""

        for i in range(len(self.vehicle_manager_list)):
            vm = self.vehicle_manager_list[i]
            debug_helper = vm.agent.debug_helper

            # we need to filter out the first 100 data points
            # since the vehicles spawn at the beginning have
            # no velocity and thus make the time gap close to infinite

            velocity_list += debug_helper.speed_list
            acceleration_list += debug_helper.acc_list
            time_gap_list += debug_helper.time_gap_list
            distance_gap_list += debug_helper.dist_gap_list

            time_gap_list_tmp = np.array(debug_helper.time_gap_list)
            time_gap_list_tmp = time_gap_list_tmp[time_gap_list_tmp < 100]
            distance_gap_list_tmp = np.array(debug_helper.dist_gap_list)
            distance_gap_list_tmp = distance_gap_list_tmp[distance_gap_list_tmp < 100]

            perform_txt += "\n Platoon member ID:%d, Actor ID:%d : \n" % (i, vm.vehicle.id)
            perform_txt += "Time gap mean: %f, std: %f \n" % (np.mean(time_gap_list_tmp), np.std(time_gap_list_tmp))
            perform_txt += "Distance gap mean: %f, std: %f \n" % (np.mean(distance_gap_list_tmp), np.std(distance_gap_list_tmp))

        figure = plt.figure()

        plt.subplot(411)
        open_plt.draw_velocity_profile_single_plot(velocity_list)

        plt.subplot(412)
        open_plt.draw_acceleration_profile_single_plot(acceleration_list)

        plt.subplot(413)
        open_plt.draw_time_gap_profile_singel_plot(time_gap_list)

        plt.subplot(414)
        open_plt.draw_dist_gap_profile_singel_plot(distance_gap_list)

        label = []
        for i in range(1, len(velocity_list) + 1):
            label.append("Leading Vehicle, id: %d" % int(i - 1) if i == 1 else "Platoon member, id: %d" % int(i - 1))

        figure.legend(label, loc="upper right")

        return figure, perform_txt

    def destroy(self) -> None:
        """
        Destroy platoon vehicles actors inside simulation world.
        """
        for vm in self.vehicle_manager_list:
            vm.destroy()
