from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from opencda.core.map.map_data import MapDataCache, SharedMapData

if TYPE_CHECKING:
    import carla

    from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
    from opencda.core.common.agent_manager import AgentManager

logger = logging.getLogger("cavise.opencda.opencda.core.common.cav_world")


class CavWorld(object):
    """
    A customized world object to save all CDA vehicle
    information and shared ML models. During co-simulation,
    it is also used to save the sumo-carla id mapping.

    Parameters
    ----------
    apply_ml : bool
        Whether apply ml/dl models in this simulation, please make sure
        you have install torch/sklearn before setting this to True.

    Attributes
    ----------
    vehicle_id_set : set
        A set that stores vehicle IDs.

    _agent_manager_dict : dict
        A dictionary that stores all universal agent managers.

    ml_manager : opencda object.
        The machine learning manager class.
    """

    def __init__(self, apply_ml: bool = False) -> None:
        self.vehicle_id_set: set[int] = set()
        self._agent_manager_dict: dict[str, AgentManager] = {}
        self._map_data_cache = MapDataCache()
        self.ml_manager: Any | None = None

        if apply_ml:
            # we import in this way so the user don't need to install ml
            # packages unless they require to
            ml_manager = getattr(importlib.import_module("opencda.customize.ml_libs.ml_manager"), "MLManager")
            # initialize the ml manager to load the DL/ML models into memory
            self.ml_manager = ml_manager()

        # this is used only when co-simulation activated.
        self.sumo2carla_ids: dict[str, int] = {}

    def get_shared_map_data(
        self,
        world: carla.World,
        carla_map: carla.Map,
        config: Mapping[str, Any],
    ) -> SharedMapData:
        """Return map geometry shared by all matching agents in this simulation."""
        return self._map_data_cache.get_or_build(world, carla_map, config)

    def update_agent_manager(self, agent_manager: AgentManager) -> None:
        """Register a universal agent manager."""
        if agent_manager.agent.is_vehicle:
            self.vehicle_id_set.add(agent_manager.agent.actor.id)
        self._agent_manager_dict[agent_manager.id] = agent_manager
        logger.debug(
            "Registered agent manager node_id=%r agent_type=%s behavior_services=%s.",
            agent_manager.id,
            agent_manager.agent.agent_type.value,
            [service.service_type for service in agent_manager.behavior_services],
        )

    def update_sumo_vehicles(self, sumo2carla_ids: dict[str, int]) -> None:
        """
        Update the sumo carla mapping dict. This is only called
        when cosimulation is conducted.

        Parameters
        ----------
        sumo2carla_ids : dict
            Key is sumo id and value is carla id.
        """
        self.sumo2carla_ids = sumo2carla_ids

    def get_agent_managers(self) -> dict[str, AgentManager]:  # noqa DC04
        """Return all registered agent managers."""
        return self._agent_manager_dict

    def get_vehicle_agent_managers(self) -> dict[str, AgentManager]:
        """Return only managers whose agents are vehicles."""
        return {agent_id: manager for agent_id, manager in self._agent_manager_dict.items() if manager.agent.is_vehicle}

    def _get_behavior_services_for_node(self, node_id: str) -> tuple[BehaviorService[Any, Any], ...]:
        agent_manager = self._agent_manager_dict.get(node_id)
        if agent_manager is not None:
            available_services = [service.service_type for service in agent_manager.behavior_services]
            logger.debug(
                "Found agent manager for node_id=%r with behavior_services=%s.",
                node_id,
                available_services,
            )
            return tuple(agent_manager.behavior_services)

        return ()

    def resolve_behavior_services(self, node_id: str, service_type: str | None = None) -> tuple[BehaviorService[Any, Any], ...]:
        """
        Resolve behavior service instances by node ID and optional service name.
        """
        logger.debug(
            "Resolving behavior services node_id=%r service_type=%r. Known nodes=%s.",
            node_id,
            service_type,
            sorted(self._agent_manager_dict),
        )

        node_services = self._get_behavior_services_for_node(node_id)
        if not node_services:
            logger.warning(
                "Could not resolve behavior services node_id=%r service_type=%r. No matching agent manager found.",
                node_id,
                service_type,
            )
            return ()

        if service_type is None:
            logger.debug(
                "Resolved %d behavior service(s) for node_id=%r without service_type filtering.",
                len(node_services),
                node_id,
            )
            return node_services

        matched_services = tuple(service for service in node_services if service.service_type == service_type)
        if matched_services:
            logger.debug(
                "Resolved %d behavior service(s) for node_id=%r service_type=%r.",
                len(matched_services),
                node_id,
                service_type,
            )
            return matched_services

        logger.warning(
            "Node node_id=%r does not expose requested service_type=%r. Available services=%s.",
            node_id,
            service_type,
            [service.service_type for service in node_services],
        )
        return ()
