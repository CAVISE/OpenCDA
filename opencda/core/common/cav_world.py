from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import carla

    from opencda.core.application.platooning.platooning_manager import PlatooningManager
    from opencda.core.common.rsu_manager import RSUManager
    from opencda.core.common.vehicle_manager import VehicleManager


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

    def get_platoon_dict(self) -> dict[str, PlatooningManager]:
        """
        Return existing platoons.
        """
        return self._platooning_dict

    def get_rsu_managers(self) -> dict[str, RSUManager]:
        """
            Return RSU managers.
        """
        return self._rsu_manager_dict

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
