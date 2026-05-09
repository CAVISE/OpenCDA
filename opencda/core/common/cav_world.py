from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import carla

    from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
    from opencda.core.application.platooning.platooning_manager import PlatooningManager
    from opencda.core.common.rsu_manager import RSUManager
    from opencda.core.common.vehicle_manager import VehicleManager

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

    _vehicle_manager_dict : dict
        A dictionary that stores vehicle managers.

    _platooning_dict : dict
        A dictionary that stores platooning managers.

    _rsu_manager_dict : dict
        A dictionary that stores RSU managers.

    ml_manager : opencda object.
        The machine learning manager class.
    """

    def __init__(self, apply_ml: bool = False) -> None:
        self.vehicle_id_set: set[int] = set()
        self._vehicle_manager_dict: dict[str, VehicleManager] = {}
        self._platooning_dict: dict[str, PlatooningManager] = {}
        self._rsu_manager_dict: dict[str, RSUManager] = {}
        self.ml_manager: Any | None = None

        if apply_ml:
            # we import in this way so the user don't need to install ml
            # packages unless they require to
            ml_manager = getattr(importlib.import_module("opencda.customize.ml_libs.ml_manager"), "MLManager")
            # initialize the ml manager to load the DL/ML models into memory
            self.ml_manager = ml_manager()

        # this is used only when co-simulation activated.
        self.sumo2carla_ids: dict[str, int] = {}

    def update_vehicle_manager(self, vehicle_manager: VehicleManager) -> None:
        """
        Update created CAV manager to the world.

        Parameters
        ----------
        vehicle_manager : opencda object
            The vehicle manager class.
        """
        self.vehicle_id_set.add(vehicle_manager.vehicle.id)
        self._vehicle_manager_dict.update({vehicle_manager.id: vehicle_manager})
        logger.info(
            "Registered vehicle manager node_id=%r with behavior_services=%s.",
            vehicle_manager.id,
            [service.service_type for service in vehicle_manager.behavior_services],
        )

    def update_platooning(self, platooning_manager: PlatooningManager) -> None:
        """
        Add created platooning.

        Parameters
        ----------
        platooning_manger : opencda object
            The platooning manager class.
        """
        self._platooning_dict.update({platooning_manager.pmid: platooning_manager})

    def update_rsu_manager(self, rsu_manager: RSUManager) -> None:
        """
        Add rsu manager.

        Parameters
        ----------
        rsu_manager : opencda object
            The RSU manager class.
        """
        self._rsu_manager_dict.update({rsu_manager.id: rsu_manager})
        logger.info(
            "Registered RSU manager node_id=%r with behavior_services=%s.",
            rsu_manager.id,
            [service.service_type for service in rsu_manager.behavior_services],
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

    def get_vehicle_managers(self) -> dict[str, VehicleManager]:
        """
        Return vehicle manager dictionary.
        """
        return self._vehicle_manager_dict

    def get_rsu_managers(self) -> dict[str, RSUManager]:  # noqa: DC04
        """
        Return RSU manager dictionary.
        """
        return self._rsu_manager_dict

    def resolve_behavior_service(self, node_id: str, service_type: str) -> BehaviorService[Any, Any] | None:
        """
        Resolve a behavior service instance by node ID and service name.
        """
        logger.info(
            "Resolving behavior service node_id=%r service_type=%r. Known vehicle_nodes=%s known_rsu_nodes=%s.",
            node_id,
            service_type,
            sorted(self._vehicle_manager_dict),
            sorted(self._rsu_manager_dict),
        )

        vehicle_manager = self._vehicle_manager_dict.get(node_id)
        if vehicle_manager is not None:
            available_vehicle_services = [service.service_type for service in vehicle_manager.behavior_services]
            logger.info(
                "Found vehicle manager for node_id=%r with behavior_services=%s.",
                node_id,
                available_vehicle_services,
            )
            for service in vehicle_manager.behavior_services:
                if service.service_type == service_type:
                    logger.info(
                        "Resolved vehicle behavior service node_id=%r service_type=%r.",
                        node_id,
                        service_type,
                    )
                    return service
            logger.warning(
                "Vehicle manager node_id=%r does not expose requested service_type=%r. Available services=%s.",
                node_id,
                service_type,
                available_vehicle_services,
            )
            return None

        rsu_manager = self._rsu_manager_dict.get(node_id)
        if rsu_manager is not None:
            available_rsu_services = [service.service_type for service in rsu_manager.behavior_services]
            logger.info(
                "Found RSU manager for node_id=%r with behavior_services=%s.",
                node_id,
                available_rsu_services,
            )
            for service in rsu_manager.behavior_services:
                if service.service_type == service_type:
                    logger.info(
                        "Resolved RSU behavior service node_id=%r service_type=%r.",
                        node_id,
                        service_type,
                    )
                    return service
            logger.warning(
                "RSU manager node_id=%r does not expose requested service_type=%r. Available services=%s.",
                node_id,
                service_type,
                available_rsu_services,
            )

        logger.warning(
            "Could not resolve behavior service node_id=%r service_type=%r. No matching vehicle or RSU manager found.",
            node_id,
            service_type,
        )
        return None

    def get_platoon_dict(self) -> dict[str, PlatooningManager]:
        """
        Return existing platoons.
        """
        return self._platooning_dict

    def locate_vehicle_manager(self, loc: carla.Location) -> VehicleManager | None:
        """
        Locate the vehicle manager based on the given location.

        Parameters
        ----------
        loc : carla.Location
            Vehicle location.

        Returns
        -------
        target_vm : opencda object
            The vehicle manager at the give location.
        """

        target_vm = None
        for vm in self._vehicle_manager_dict.values():
            x = vm.localizer.get_ego_pos().location.x
            y = vm.localizer.get_ego_pos().location.y

            if loc.x == x and loc.y == y:
                target_vm = vm
                break

        return target_vm
