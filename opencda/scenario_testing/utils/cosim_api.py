"""
Co-simulation scenario manager for CARLA-SUMO integration.

This module provides functionality for synchronized co-simulation between CARLA
and SUMO traffic simulators, handling vehicle spawning, state synchronization,
and traffic light coordination.
"""

import os
import logging
from typing import Dict, Optional, Any, Set

import carla

from opencda.co_simulation.sumo_integration.constants import SPAWN_OFFSET_Z
from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper
from opencda.co_simulation.sumo_integration.constants import INVALID_ACTOR_ID
from opencda.co_simulation.sumo_integration.sumo_simulation import SumoSimulation
from opencda.scenario_testing.utils.sim_api import ScenarioManager
from opencda.core.common.cav_world import CavWorld

logger = logging.getLogger("cavise.cosim_api")


class CoScenarioManager(ScenarioManager):
    """
    The Scenario manager for co-simulation(CARLA-SUMO). All sumo-related
    functions and variables will start with self.sumo.xx, and all carla-related
    functions and members won't have such prefixes.

     Parameters
    ----------
    scenario_params : dict
        Dictionary containing all simulation configurations.
    apply_ml : bool
        Whether to load DL/ML model (PyTorch required) in this simulation.
    carla_version : str
        CARLA simulator version, currently supports 0.9.11 and 0.9.12.
    node_ids : dict
        Dictionary mapping node identifiers for different vehicle types.
    xodr_path : str, optional
        Path to the customized map XODR file. Default is None.
    town : str, optional
        Town name if not using customized map, e.g., 'Town06'. Default is None.
    cav_world : object, optional
        CAV world object. Default is None.
    sumo_file_parent_path : str, optional
        Parent path to SUMO configuration files. Default is None.
    carla_host : str, optional
        CARLA server host address. Default is "carla".
    carla_timeout : float, optional
        Timeout for CARLA connection in seconds. Default is 30.0.

    Attributes
    ----------
    _active_actors : set
        Set of currently active actor IDs.
    spawned_actors : set
        Set of newly spawned actor IDs in current tick.
    destroyed_actors : set
        Set of destroyed actor IDs in current tick.
    node_ids : dict
        Mapping of node identifiers for vehicle types.
    _tls : dict
        Dictionary of traffic light objects keyed by landmark ID.
    sumo : SumoSimulation
        SUMO simulation instance.
    sumo2carla_ids : dict
        Mapping from SUMO actor IDs to CARLA actor IDs.
    carla2sumo_ids : dict
        Mapping from CARLA actor IDs to SUMO actor IDs.
    """

    def __init__(
        self,
        scenario_params: Dict[str, Any],
        apply_ml: bool,
        carla_version: str,
        node_ids: Dict[str, Dict[int, str]],
        xodr_path: Optional[str] = None,
        town: Optional[str] = None,
        cav_world: Optional[CavWorld] = None,
        sumo_file_parent_path: Optional[str] = None,
        carla_host: str = "carla",
        carla_timeout: float = 30.0,
    ):
        # carla side initializations(partial init is already done in scenario manager
        super(CoScenarioManager, self).__init__(scenario_params, apply_ml, carla_version, xodr_path, town, cav_world, carla_host)

        # these following sets are used to track the vehicles controlled by sumo side
        self._active_actors: Set[int] = set()
        self.spawned_actors: Set[int] = set()
        self.destroyed_actors: Set[int] = set()
        self.node_ids = node_ids

        # contains all carla traffic lights objects
        self._tls = {}
        for landmark in self.carla_map.get_all_landmarks_of_type("1000001"):
            if landmark.id != "":
                traffic_ligth = self.world.get_traffic_light(landmark)
                if traffic_ligth is not None:
                    self._tls[landmark.id] = traffic_ligth
                else:
                    logging.warning(f"Landmark {landmark.id} is not linked to any traffic light")

        # sumo side initialization
        base_name: str = os.path.basename(sumo_file_parent_path)

        sumo_key = "sumo"

        sumo_cfg = os.path.join(sumo_file_parent_path, base_name + ".sumocfg")
        # todo: use yaml file to generate the route file
        assert os.path.isfile(sumo_cfg), (
            f"{sumo_cfg} does not exist, make sure your config file name has the same basename as the directory and use .sumocfg as extension"
        )

        sumo_port = scenario_params[sumo_key]["port"]
        sumo_host = scenario_params[sumo_key]["host"]
        sumo_gui = scenario_params[sumo_key]["gui"]
        sumo_client_order = scenario_params[sumo_key]["client_order"]
        # tick freq, the same as carla
        sumo_step_length = scenario_params[sumo_key]["step_length"]

        self.sumo = SumoSimulation(sumo_cfg, sumo_step_length, sumo_host, sumo_port, sumo_gui, sumo_client_order)
        # the sumo traffic light should be synchronized with carla
        self.sumo.switch_off_traffic_lights()

        # Mapped actor ids. All vehicles controlled by sumo is
        # in sumo2carla_ids, all vehicles controlled by carla
        # is saved in carla2sumo_ids
        self.sumo2carla_ids: Dict[str, int] = {}  # key: sumo id, value: carla id
        self.carla2sumo_ids: Dict[str, int] = {}  # key: carla id, value: sumo id

        BridgeHelper.blueprint_library = self.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

    def tick(self) -> None:
        """
        Execute a single step of co-simulation.

        Synchronizes vehicle positions and traffic light states between SUMO and
        CARLA. SUMO moves vehicles to new positions, then CARLA updates
        corresponding actors to match those locations.
        """
        # -----------------
        # sumo-->carla sync
        # -----------------
        self.sumo.tick()

        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())

        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            # given the sumo vehicle type, return the corresponding
            # carla vehicle type. If there is no such correspondence,
            # carla will choose a random vehicle type.
            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, False)

            if carla_blueprint is not None:
                # return the sumo-controlled vehicle position under
                # Carla coordinate system. There is a translation between
                # the two.
                carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform, sumo_actor.extent)
                carla_actor_id = self.spawn_actor(carla_blueprint, carla_transform)
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id

            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))

        # Updating sumo actors in carla.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform, sumo_actor.extent)
            assert self.synchronize_vehicle(carla_actor_id, carla_transform)

        # -----------------
        # carla-->sumo sync
        # -----------------
        self.world.tick()

        # Update data structures for the current frame.
        current_actors = set(
            [
                vehicle.id
                for vehicle in filter(
                    lambda actor: actor.type_id.startswith("vehicle.") or actor.type_id.startswith("static."),
                    self.world.get_actors(),
                )
            ]
        )
        self.spawned_actors = current_actors.difference(self._active_actors)
        self.destroyed_actors = self._active_actors.difference(current_actors)
        self._active_actors = current_actors

        # Spawning new carla actors (not controlled by sumo). For example, the CAV we created on the carla side.
        carla_spawned_actors = self.spawned_actors - set(self.sumo2carla_ids.values())

        for carla_actor_id in carla_spawned_actors:
            carla_actor = self.world.get_actor(carla_actor_id)
            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = carla_actor.attributes.get("color", None)
            if type_id is not None:
                if carla_actor_id in self.node_ids["platoon"]:
                    sumo_actor_id = self.sumo.spawn_actor(type_id, self.node_ids["platoon"][carla_actor_id], color)
                elif carla_actor_id in self.node_ids["cav"]:
                    sumo_actor_id = self.sumo.spawn_actor(type_id, self.node_ids["cav"][carla_actor_id], color)
                elif carla_actor_id in self.node_ids["rsu"]:
                    sumo_actor_id = self.sumo.spawn_actor(type_id, self.node_ids["rsu"][carla_actor_id], color)
                else:
                    sumo_actor_id = self.sumo.spawn_actor(type_id, f"unknown-{carla_actor_id}", color)

                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.carla2sumo_ids[carla_actor_id] = sumo_actor_id
                    self.sumo.subscribe(sumo_actor_id)

        # Destroying required carla actors in sumo.
        for carla_actor_id in self.destroyed_actors:
            if carla_actor_id in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(carla_actor_id))

        # updating carla actors in sumo. For instance, update the new CAV
        # position in sumo
        for carla_actor_id in self.carla2sumo_ids:
            sumo_actor_id = self.carla2sumo_ids[carla_actor_id]

            carla_actor = self.world.get_actor(carla_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(), carla_actor.bounding_box.extent)
            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, None)

        # Updates traffic lights in sumo based on carla information.
        # todo make sure the tl is synced
        common_landmarks = self.sumo.traffic_light_ids & self.traffic_light_ids
        for landmark_id in common_landmarks:
            carla_tl_state = self.get_traffic_light_state(landmark_id)
            sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(carla_tl_state)

            # Updates all the sumo links related to this landmark.
            self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)

        # update the sumo2carla dict to cav world
        self.cav_world.update_sumo_vehicles(self.sumo2carla_ids)

    @property
    def traffic_light_ids(self) -> Set[str]:
        return set(self._tls.keys())

    def get_traffic_light_state(self, landmark_id: str) -> Optional[carla.TrafficLightState]:
        """
        Get the state of a traffic light by landmark ID.

        Parameters
        ----------
        landmark_id : str
            The landmark ID of the traffic light.

        Returns
        -------
        carla.TrafficLightState or None
            Traffic light state if exists, None otherwise.
        """
        if landmark_id not in self._tls:
            return None
        return self._tls[landmark_id].state

    def spawn_actor(self, blueprint: carla.ActorBlueprint, transform: carla.Transform) -> int:
        """
        Spawns a new carla actor based on the given coordinate.

        Parameters
        ----------
        blueprint : carla.blueprint
            Blueprint of the actor to be spawned.
        transform : carla.transform
            Transform where the actor will be spawned.

        Returns
        -------
        actor_id : int
            The carla actor id the actor is successfully spawned. Otherwise,
            INVALID_ACTOR_ID will be return.
        """
        transform = carla.Transform(transform.location + carla.Location(0, 0, SPAWN_OFFSET_Z), transform.rotation)

        batch = [carla.command.SpawnActor(blueprint, transform).then(carla.command.SetSimulatePhysics(carla.command.FutureActor, False))]
        response = self.client.apply_batch_sync(batch, False)[0]
        if response.error:
            logging.error(f"Spawn carla actor failed. {response.error}")
            return INVALID_ACTOR_ID

        return response.actor_id

    def synchronize_vehicle(self, vehicle_id: int, transform: carla.Transform) -> bool:
        """
        The key function of co-simulation. Given the updated location in sumo,
        carla will move the corresponding vehicle to the same location.

        Parameters
        ----------
        vehicle_id : int
            The id of the carla actor to be updated.

        transform : carla.Transform
            The new vehicle transform.

        Returns
        -------
        success : bool
            Whether update is successful.
        """
        vehicle = self.world.get_actor(vehicle_id)
        if vehicle is None:
            return False

        vehicle.set_transform(transform)
        return True

    def destroy_actor(self, actor_id: int) -> bool:
        """
        Destroys the given carla actor.

        Parameters
        ----------
        actor_id : int
            The actor id in carla.

        Returns
        -------
        bool
            True if destruction is successful, False otherwise.
        """
        actor = self.world.get_actor(actor_id)
        if actor is not None:
            return actor.destroy()
        return False

    def close(self) -> None:
        """
        Close the co-simulation and clean up resources.

        Destroys all synchronized actors in both CARLA and SUMO, unfreezes
        traffic lights, and restores original simulation settings.
        """
        # restore to origin setting
        self.world.apply_settings(self.origin_settings)

        # Destroying synchronized actors.
        logger.info("Destroying carla actor")
        for carla_actor_id in self.sumo2carla_ids.values():
            self.destroy_actor(carla_actor_id)

        logger.info("Destroying sumo actor")
        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # unfreeze traffic lights, since sumo may freeze the traffic light
        for actor in self.world.get_actors():
            if actor.type_id == "traffic.traffic_light":
                actor.freeze(False)

        self.sumo.close()
