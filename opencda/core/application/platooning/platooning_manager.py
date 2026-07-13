"""Platooning Manager"""

import uuid
import weakref
import logging

import carla
from opencda.metrics_tools.report_models import EntityMetricCollections


logger = logging.getLogger("cavise.opencda.opencda.core.application.platooning.platooning_manager")


class PlatooningManager(object):
    """
    Platoon manager. Used to manage all vehicle managers inside the platoon.

    Parameters
    ----------
    config_yaml : dict
        The configuration dictionary for platoon.

    cav_world : opencda object
        CAV world that stores all CAV information.

    Attributes
    ----------
    pmid : int
        The  platooning manager ID.
    agent_manager_list : list
        A list of all vehicle agent managers within the platoon.
    destination : carla.location
        The destiantion of the current plan.
    center_loc : carla.location
        The center location of the platoon.
    leader_target_speed : float
        The speed of the leader vehicle.
    origin_leader_target_speed : float
        The original planned target speed of the platoon leader.
    recover_speed_counter : int
        The counter that record the number of speed recovery attempts.
    """

    def __init__(self, config_yaml, cav_world):
        self.pmid = str(uuid.uuid1())

        self.agent_manager_list = []
        self.maximum_capacity = config_yaml["max_capacity"]

        self.destination = None
        self.center_loc = None  # noqa: DC05

        # this is used to control platooning speed during joining
        self.leader_target_speed = 0
        self.origin_leader_target_speed = 0
        self.recover_speed_counter = 0

        cav_world.update_platooning(self)
        self.cav_world = weakref.ref(cav_world)()

    def set_lead(self, agent_manager):
        """
        Set the leader of the platooning

        Parameters
        __________
        agent_manager : opencda object
            The universal agent manager.
        """
        self.add_member(agent_manager, leader=True)

        # this variable is used to control leader speed
        behavior_agent = agent_manager.agent.behavior_agent
        self.origin_leader_target_speed = behavior_agent.max_speed - behavior_agent.speed_lim_dist

    def add_member(self, agent_manager, leader=False):
        """
        Add memeber to the current platooning

        Parameters
        __________
        leader : boolean
            Indicator of whether this cav is a leader.

        agent_manager : opencda object
            The universal agent manager.
        """
        self.agent_manager_list.append(agent_manager)
        agent_manager.agent.v2x_manager.set_platoon(
            len(self.agent_manager_list) - 1,
            platooning_object=self,
            platooning_id=self.pmid,
            leader=leader,
        )

    def set_member(self, agent_manager, index, lead=False):
        """
        Set member at specific index

        Parameters
        ----------
        lead : boolean
            Indicator of whether this cav is a leader.

        agent_manager : opencda object
            The universal agent manager.

        index : int
            The platoon index of the current vehicle.
        """
        self.agent_manager_list.insert(index, agent_manager)
        agent_manager.agent.v2x_manager.set_platoon(index, platooning_object=self, platooning_id=self.pmid, leader=lead)

    def cal_center_loc(self):
        """
        Calculate and update center location of the platoon.
        """
        v1_ego_transform = self.agent_manager_list[0].agent.v2x_manager.get_ego_pos()
        v2_ego_transform = self.agent_manager_list[-1].agent.v2x_manager.get_ego_pos()

        if any(t is None for t in (v1_ego_transform, v2_ego_transform)):
            return
        self.center_loc = carla.Location(  # noqa: DC05
            x=(v1_ego_transform.location.x + v2_ego_transform.location.x) / 2,
            y=(v1_ego_transform.location.y + v2_ego_transform.location.y) / 2,
            z=(v1_ego_transform.location.z + v2_ego_transform.location.z) / 2,
        )

    def update_member_order(self):
        """
        Update the members' front and rear vehicle.
        This should be called whenever new member added to the platoon list.
        """
        for i, manager in enumerate(self.agent_manager_list):
            if i != 0:
                manager.agent.v2x_manager.set_platoon(i, leader=False)
                manager.agent.v2x_manager.set_platoon_front(self.agent_manager_list[i - 1])
            if i != len(self.agent_manager_list) - 1:
                leader = True if i == 0 else False
                manager.agent.v2x_manager.set_platoon(i, leader=leader)
                manager.agent.v2x_manager.set_platoon_rear(self.agent_manager_list[i + 1])

    def reset_speed(self):
        """
        After joining request accepted for certain steps,
        the platoon will return to the origin speed.
        """
        if self.recover_speed_counter <= 0:
            self.leader_target_speed = self.origin_leader_target_speed
        else:
            self.recover_speed_counter -= 1

    def response_joining_request(self, request_loc):
        """
        Identify whether to accept the joining request based on capacity.

        Parameters
        ----------
        request_loc : carla.Location)
            Request vehicle location.

        Returns
        -------
        response : boolean
        Indicator of whether the joining request is accepted.

        """
        if len(self.agent_manager_list) >= self.maximum_capacity:
            return False
        else:
            # when the platoon accept a joining request,by default
            # it will decrease the speed so the merging vehicle
            # can better catch up with
            self.leader_target_speed = self.origin_leader_target_speed - 5
            self.recover_speed_counter = 200

            # find the corresponding vehicle manager and add it to the leader's
            # whitelist
            request_manager = self.cav_world.locate_agent_manager(request_loc)
            self.agent_manager_list[0].agent.behavior_agent.add_white_list(request_manager)

            return True

    def set_destination(self, destination):
        """
        Set desination of the vehicle managers in the platoon.
        """
        self.destination = destination
        for manager in self.agent_manager_list:
            manager.agent.set_destination(manager.agent.vehicle.get_location(), destination, clean=True)

    def update_information(self):
        """
        Update CAV world information for every member in the list.
        """
        self.reset_speed()
        for manager in self.agent_manager_list:
            manager.agent.update()
        # update the center location of the platoon
        self.cal_center_loc()

    def run_step(self):
        """
        Run one control step for each vehicles.

        Returns
        -------
        control_list : list
            The control command list for all vehicles.
        """
        control_list = []
        for manager in self.agent_manager_list:
            control = manager.agent.plan_control(self.leader_target_speed)
            control_list.append(control)

        for i, control in enumerate(control_list):
            self.agent_manager_list[i].agent.vehicle.apply_control(control)

        return control_list

    def get_metric_collections(self) -> tuple[EntityMetricCollections, ...]:
        """Return raw metric collections for all platoon members."""
        return tuple(
            EntityMetricCollections(
                entity_id=manager.agent.vehicle.id,
                context_id=index,
                collections=(
                    manager.agent.behavior_agent.metrics_collector.get_raw(),
                    manager.agent.behavior_agent.platooning_metrics_collector.get_raw(),
                ),
            )
            for index, manager in enumerate(self.agent_manager_list)
        )

    def destroy(self):
        """
        Destroy platoon vehicles actors inside simulation world.
        """
        for manager in self.agent_manager_list:
            manager.destroy()
