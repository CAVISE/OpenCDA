"""
Utilize scenario manager to manage CARLA simulation construction. This script
is used for carla simulation only, and if you want to manage the Co-simulation,
please use cosim_api.py.
"""

from __future__ import annotations

import math
import random
import logging
import sys
import json

from collections.abc import Callable, Mapping
from random import shuffle
from typing import Any, TypeAlias, cast

from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

import carla
import numpy as np
import torch

from opencda.core.application.platooning.platooning_manager import PlatooningManager
from opencda.core.common.agent import AgentType
from opencda.core.common.agent_manager import AgentManager
from opencda.core.common.cav_world import CavWorld
from opencda.core.sensing.perception.perception_manager import PerceptionRequirements
from opencda.scenario_testing.utils.customized_map_api import load_customized_world, bcolors

logger = logging.getLogger("cavise.opencda.opencda.scenario_testing.utils.sim_api")

ConfigDict: TypeAlias = dict[str, Any]
BlueprintMeta: TypeAlias = Mapping[str, Mapping[str, Any]]
MapHelper: TypeAlias = Callable[..., carla.Transform]


def car_blueprint_filter(blueprint_library: Any, carla_version: str = "0.9.15") -> list[Any]:
    """
    Exclude the uncommon vehicles from the default CARLA blueprint library
    (i.e., isetta, carlacola, cybertruck, t2).

    Parameters
    ----------
    blueprint_library : carla.blueprint_library
        The blueprint library that contains all models.

    carla_version : str
        CARLA simulator version, currently support 0.9.11 and 0.9.12. We need
        this as since CARLA 0.9.12 the blueprint name has been changed a lot.

    Returns
    -------
    blueprints : list
        The list of suitable blueprints for vehicles.
    """

    if carla_version == "0.9.15" or carla_version == "0.9.14":
        logger.info(f"Carla {carla_version} version is selected")
        blueprints = [
            blueprint_library.find("vehicle.audi.a2"),
            blueprint_library.find("vehicle.audi.tt"),
            blueprint_library.find("vehicle.ford.ambulance"),
            blueprint_library.find("vehicle.ford.crown"),
            blueprint_library.find("vehicle.mini.cooper_s_2021"),
            blueprint_library.find("vehicle.nissan.micra"),
            blueprint_library.find("vehicle.nissan.patrol"),
            blueprint_library.find("vehicle.nissan.patrol_2021"),
            blueprint_library.find("vehicle.tesla.cybertruck"),
            blueprint_library.find("vehicle.volkswagen.t2"),
            blueprint_library.find("vehicle.volkswagen.t2_2021"),
            blueprint_library.find("vehicle.micro.microlino"),
            blueprint_library.find("vehicle.dodge.charger_police"),
            blueprint_library.find("vehicle.dodge.charger_police_2020"),
            blueprint_library.find("vehicle.dodge.charger_2020"),
            blueprint_library.find("vehicle.lincoln.mkz_2020"),
            blueprint_library.find("vehicle.seat.leon"),
            blueprint_library.find("vehicle.nissan.patrol"),
            blueprint_library.find("vehicle.nissan.micra"),
        ]
    else:
        sys.exit(
            "Since v0.1.4, we do not support version earlier than "
            "CARLA v0.9.14. If you want to use early CARLA version including"
            "0.9.11 and 0.9.12, please use OpenCDA v0.1.3."
        )

    return blueprints


def multi_class_vehicle_blueprint_filter(label: str, blueprint_library: Any, bp_meta: BlueprintMeta) -> list[Any]:
    """
    Get a list of blueprints that have the class equals the specified label.

    Parameters
    ----------
    label : str
        Specified blueprint.

    blueprint_library : carla.blueprint_library
        The blueprint library that contains all models.

    bp_meta : dict
        Dictionary of {blueprint name: blueprint class}.

    Returns
    -------
    blueprints : list
        List of blueprints that have the class equals the specified label.

    """
    blueprints = [blueprint_library.find(k) for k, v in bp_meta.items() if v["class"] == label]
    return blueprints


class ScenarioManager:
    """
    The manager that controls simulation construction, backgound traffic
    generation and CAVs spawning.

    Parameters
    ----------
    scenario_params : dict
        The dictionary contains all simulation configurations.

    carla_version : str
        CARLA simulator version, it currently supports 0.9.11 and 0.9.12

    xodr_path : str
        The xodr file to the customized map, default: None.

    town : str
        Town name if not using customized map, eg. 'Town06'.

    apply_ml : bool
        Whether need to load dl/ml model(pytorch required) in this simulation.

    Attributes
    ----------
    client : carla.client
        The client that connects to carla server.

    world : carla.world
        Carla simulation server.

    origin_settings : dict
        The origin setting of the simulation server.

    cav_world : opencda object
        CAV World that contains the information of all CAVs.

    carla_map : carla.map
        Car;a HD Map.

    """

    def __init__(
        self,
        scenario_params: ConfigDict,
        apply_ml: bool,
        carla_version: str,
        xodr_path: str | None = None,
        town: str | None = None,
        cav_world: CavWorld | None = None,
        carla_host: str = "carla",
        carla_timeout: float = 30.0,
    ) -> None:
        self.scenario_params = scenario_params
        self.carla_version = carla_version
        self.bp_meta: dict[str, dict[str, Any]] = {}
        self.bp_class_sample_prob: dict[str, float] = {}

        simulation_config = cast(ConfigDict, scenario_params["world"])

        # set random seed if stated
        if "seed" in simulation_config:
            seed = int(simulation_config["seed"])
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.client = carla.Client(carla_host, simulation_config["client_port"])
        self.client.set_timeout(carla_timeout)

        world: carla.World | None
        if xodr_path:
            world = load_customized_world(xodr_path, self.client)
        elif town:
            try:
                world = self.client.load_world(town)
            except RuntimeError as error:
                logger.error(
                    f"{bcolors.FAIL}{town} probably is not in your CARLA repo! Please download all town maps to your CARLA repo!{bcolors.ENDC}"
                )
                logger.error(error)
                world = None
        else:
            world = self.client.get_world()

        if world is None:
            sys.exit("- World loading failed")

        self.world = world
        self.origin_settings = self.world.get_settings()
        new_settings = self.world.get_settings()

        if simulation_config["sync_mode"]:
            fixed_delta_seconds = float(simulation_config["fixed_delta_seconds"])
            substepping = simulation_config.get("substepping", True)
            max_substep_delta_time = float(simulation_config.get("max_substep_delta_time", 0.01))
            max_substeps = simulation_config.get("max_substeps", 10)

            if not isinstance(substepping, bool):
                raise ValueError("Config key 'substepping' must be a boolean.")
            if not isinstance(max_substeps, int) or isinstance(max_substeps, bool) or not 1 <= max_substeps <= 16:
                raise ValueError("Config key 'max_substeps' must be an integer in the range [1, 16].")
            if fixed_delta_seconds <= 0 or max_substep_delta_time <= 0:
                raise ValueError("Config keys 'fixed_delta_seconds' and 'max_substep_delta_time' must be positive.")
            max_physics_delta = max_substep_delta_time * max_substeps
            if substepping and fixed_delta_seconds > max_physics_delta and not math.isclose(fixed_delta_seconds, max_physics_delta):
                raise ValueError(
                    "Invalid CARLA physics settings: fixed_delta_seconds must be less than or equal to "
                    "max_substep_delta_time * max_substeps "
                    f"({fixed_delta_seconds} > {max_substep_delta_time} * {max_substeps})."
                )

            new_settings.synchronous_mode = True  # noqa: DC05
            new_settings.fixed_delta_seconds = fixed_delta_seconds
            new_settings.substepping = substepping
            new_settings.max_substep_delta_time = max_substep_delta_time
            new_settings.max_substeps = max_substeps
        else:
            sys.exit("ERROR: Current version only supports sync simulation mode")

        self.world.apply_settings(new_settings)

        traffic_config = self.scenario_params.get("carla_traffic_manager")
        if traffic_config is not None:
            self._configure_traffic_manager(
                self.client.get_trafficmanager(),
                cast(ConfigDict, traffic_config),
            )

        # set weather
        weather = self.set_weather(simulation_config["weather"])
        self.world.set_weather(weather)

        # Define probabilities for each type of blueprint
        self.use_multi_class_bp = scenario_params["blueprint"]["use_multi_class_bp"] if "blueprint" in scenario_params else False
        if self.use_multi_class_bp:
            # bbx/blueprint meta
            with open(scenario_params["blueprint"]["bp_meta_path"]) as f:
                self.bp_meta = cast(dict[str, dict[str, Any]], json.load(f))
            self.bp_class_sample_prob = scenario_params["blueprint"]["bp_class_sample_prob"]

            # normalize probability
            self.bp_class_sample_prob = {k: v / sum(self.bp_class_sample_prob.values()) for k, v in self.bp_class_sample_prob.items()}

        self.cav_world = cav_world if cav_world is not None else CavWorld(apply_ml)
        self.carla_map = self.world.get_map()
        self.apply_ml = apply_ml

    @staticmethod
    def set_weather(weather_settings: Mapping[str, float]) -> carla.WeatherParameters:
        """
        Set CARLA weather params.

        Parameters
        ----------
        weather_settings : dict
            The dictionary that contains all parameters of weather.

        Returns
        -------
        The CARLA weather setting.
        """
        weather = carla.WeatherParameters(
            sun_altitude_angle=weather_settings["sun_altitude_angle"],
            cloudiness=weather_settings["cloudiness"],
            precipitation=weather_settings["precipitation"],
            precipitation_deposits=weather_settings["precipitation_deposits"],
            wind_intensity=weather_settings["wind_intensity"],
            fog_density=weather_settings["fog_density"],
            fog_distance=weather_settings["fog_distance"],
            fog_falloff=weather_settings["fog_falloff"],
            wetness=weather_settings["wetness"],
        )
        return weather

    def spawn_custom_actor(self, spawn_transform: carla.Transform, config: Mapping[str, Any], fallback_model: str) -> carla.Actor:
        model = config.get("model", fallback_model)
        cav_vehicle_bp = self.world.get_blueprint_library().find(model)

        color = config.get("color")
        if color is not None:
            try:
                cav_vehicle_bp.set_attribute("color", ",".join(map(str, color)))
            except IndexError:
                logger.warning(f"Vehicle model {cav_vehicle_bp.id} does not support the 'color' attribute. Skipping.")

        return self.world.spawn_actor(cav_vehicle_bp, spawn_transform)

    # TODO: make a custom_actor_manager for inanimated objects
    def create_custom_actor_manager(
        self,
        application: list[str],
        map_helper: MapHelper | None = None,
        data_dump: bool = False,
        fallback_model: str = "vehicle.lincoln.mkz_2017",
    ) -> tuple[list[Any], dict[int, Any]] | None:
        if self.scenario_params.get("scenario") is None or self.scenario_params["scenario"].get("custom_actor_list", None) is None:
            logger.info("No custom actor was created")
            return [], {}
        for i, config in enumerate(self.scenario_params["scenario"].get("custom_actor_list", {})):
            actor_config = cast(ConfigDict, OmegaConf.create(config))
            # if the spawn position is a single scalar, we need to use map
            # helper to transfer to spawn transform
            if "spawn_special" not in actor_config:
                spawn_transform = carla.Transform(
                    carla.Location(x=actor_config["spawn_position"][0], y=actor_config["spawn_position"][1], z=actor_config["spawn_position"][2]),
                    carla.Rotation(
                        pitch=actor_config["spawn_position"][5], yaw=actor_config["spawn_position"][4], roll=actor_config["spawn_position"][3]
                    ),
                )
            else:
                spawn_transform = cast(MapHelper, map_helper)(self.carla_version, *actor_config["spawn_special"])

            self.spawn_custom_actor(spawn_transform, actor_config, fallback_model)

        return None

    def create_vehicle_agents(
        self,
        application: list[str],
        map_helper: MapHelper | None = None,
        perception_requirements: PerceptionRequirements | None = None,
        fallback_model: str = "vehicle.lincoln.mkz_2017",
    ) -> tuple[list[AgentManager], dict[int, Any]]:
        """
        Create a list of single CAVs.

        Parameters
        ----------
        application : list
            The application purpose, a list, eg. ['single'], ['platoon'].

        map_helper : function
            A function to help spawn vehicle on a specific position in
            a specific map.

        perception_requirements : PerceptionRequirements
            Sensor/runtime requirements used to initialize perception-related modules.

        fallback_model: str
            Fallback cav model if none provided by config.

        Returns
        -------
        single_cav_list : list
            A list contains all single CAVs' vehicle manager.
        """
        single_cav_list: list[AgentManager] = []
        cav_carla_list: dict[int, Any] = {}
        perception_requirements = perception_requirements or PerceptionRequirements()

        if self.scenario_params.get("scenario") is None or self.scenario_params["scenario"].get("single_cav_list", None) is None:
            logger.info("No CAV was created")
            return single_cav_list, cav_carla_list

        for i, cav_config in enumerate(self.scenario_params["scenario"]["single_cav_list"]):
            # in case the cav wants to join a platoon later
            # it will be empty dictionary for single cav application
            platoon_base = cast(ConfigDict, OmegaConf.create({"platoon": self.scenario_params.get("platoon_base", {})}))
            cav_config = cast(ConfigDict, OmegaConf.merge(self.scenario_params["vehicle_base"], platoon_base, cav_config))
            # if the spawn position is a single scalar, we need to use map
            # helper to transfer to spawn transform
            if "spawn_special" not in cav_config:
                spawn_transform = carla.Transform(
                    carla.Location(x=cav_config["spawn_position"][0], y=cav_config["spawn_position"][1], z=cav_config["spawn_position"][2]),
                    carla.Rotation(pitch=cav_config["spawn_position"][5], yaw=cav_config["spawn_position"][4], roll=cav_config["spawn_position"][3]),
                )
            else:
                spawn_transform = cast(MapHelper, map_helper)(self.carla_version, *cav_config["spawn_special"])

            vehicle = self.spawn_custom_actor(spawn_transform, cav_config, fallback_model)

            agent_manager = AgentManager.create(
                actor=vehicle,
                config_yaml=cav_config,
                carla_map=self.carla_map,
                cav_world=self.cav_world,
                agent_type=AgentType.CAV,
                application=application,
                current_time=self.scenario_params["current_time"],
                perception_requirements=perception_requirements,
                id_prefix="cav",
            )

            cav_carla_list[vehicle.id] = agent_manager.id

            agent_manager.agent.v2x_manager.set_platoon(None)
            agent_manager.agent.update()
            if agent_manager.agent.use_carla_autopilot:
                if "destination" in cav_config:
                    logger.info("CAV %s is using CARLA autopilot; configured destination is ignored.", agent_manager.id)
            else:
                if "destination" not in cav_config:
                    raise ValueError(f"CAV {agent_manager.id} requires 'destination' unless carla_autopilot is enabled.")
                destination = carla.Location(x=cav_config["destination"][0], y=cav_config["destination"][1], z=cav_config["destination"][2])
                agent_manager.agent.set_destination(agent_manager.agent.vehicle.get_location(), destination, clean=True)

            single_cav_list.append(agent_manager)
            logger.info(f"Created CAV with id {agent_manager.id}")

        for agent_manager in single_cav_list:
            if agent_manager.agent.use_carla_autopilot:
                self.configure_carla_autopilot_vehicle(agent_manager.agent.vehicle)

        return single_cav_list, cav_carla_list

    def configure_carla_autopilot_vehicle(self, vehicle: carla.Vehicle) -> None:
        """
        Apply scenario Traffic Manager settings to an OpenCDA-managed vehicle
        that is driven by CARLA autopilot.

        Parameters
        ----------
        vehicle : carla.Vehicle
            Vehicle driven by CARLA autopilot.
        """
        if self.scenario_params.get("carla_traffic_manager") is None:
            return

        traffic_config = cast(ConfigDict, self.scenario_params["carla_traffic_manager"])
        tm = self.client.get_trafficmanager()
        self._configure_traffic_manager_vehicle(tm, traffic_config, vehicle, randomize_speed=True)

    @staticmethod
    def _configure_traffic_manager(tm: carla.TrafficManager, traffic_config: ConfigDict) -> None:
        """
        Apply global scenario settings to CARLA Traffic Manager.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic Manager instance to configure.
        traffic_config : ConfigDict
            Scenario Traffic Manager settings.
        """
        tm.set_global_distance_to_leading_vehicle(traffic_config["global_distance"])
        tm.set_synchronous_mode(traffic_config["sync_mode"])
        tm.set_osm_mode(traffic_config["set_osm_mode"])
        tm.global_percentage_speed_difference(traffic_config["global_speed_perc"])

    @staticmethod
    def _configure_traffic_manager_vehicle(
        tm: carla.TrafficManager,
        traffic_config: ConfigDict,
        vehicle: carla.Vehicle,
        *,
        randomize_speed: bool,
    ) -> None:
        """
        Apply scenario Traffic Manager settings to a vehicle.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic Manager instance controlling the vehicle.
        traffic_config : ConfigDict
            Scenario Traffic Manager settings.
        vehicle : carla.Vehicle
            Vehicle to configure.
        randomize_speed : bool
            Whether to add a random offset to the configured speed difference.
        """
        tm.auto_lane_change(vehicle, traffic_config["auto_lane_change"])
        tm.ignore_lights_percentage(vehicle, traffic_config["ignore_lights_percentage"])
        tm.ignore_signs_percentage(vehicle, traffic_config["ignore_signs_percentage"])
        tm.ignore_vehicles_percentage(vehicle, traffic_config["ignore_vehicles_percentage"])
        tm.ignore_walkers_percentage(vehicle, traffic_config["ignore_walkers_percentage"])

        if traffic_config["random_left_lanechange_percentage"] != 0:
            tm.random_left_lanechange_percentage(vehicle, traffic_config["random_left_lanechange_percentage"])
        if traffic_config["random_right_lanechange_percentage"] != 0:
            tm.random_right_lanechange_percentage(vehicle, traffic_config["random_right_lanechange_percentage"])

        speed_difference = traffic_config["global_speed_perc"]
        if randomize_speed:
            speed_difference += random.randint(-30, 30)
        tm.vehicle_percentage_speed_difference(vehicle, speed_difference)

    def create_platoon_manager(
        self,
        map_helper: MapHelper | None = None,
        perception_requirements: PerceptionRequirements | None = None,
        fallback_model: str = "vehicle.lincoln.mkz_2017",
    ) -> tuple[list[PlatooningManager], dict[int, Any]]:
        """
        Create a list of platoons.

        Parameters
        ----------
        map_helper : function
            A function to help spawn vehicle on a specific position in a
            specific map.

        perception_requirements : PerceptionRequirements
            Sensor/runtime requirements used to initialize perception-related modules.

        fallback_model: str
            Fallback cav model if none provided by config.

        Returns
        -------
        single_cav_list : list
            A list contains all single CAVs' vehicle manager.
        """
        platoon_list: list[PlatooningManager] = []
        platoon_carla_ids: dict[int, Any] = {}
        perception_requirements = perception_requirements or PerceptionRequirements()

        if self.scenario_params.get("scenario") is None or self.scenario_params["scenario"].get("platoon_list", None) is None:
            logger.info("No platoon was created")
            return platoon_list, platoon_carla_ids

        # create platoons
        for i, platoon in enumerate(self.scenario_params["scenario"]["platoon_list"]):
            platoon = cast(ConfigDict, OmegaConf.merge(self.scenario_params["platoon_base"], platoon))
            platoon_manager = PlatooningManager(platoon, self.cav_world)

            for j, cav_config in enumerate(platoon["members"]):
                platoon_base = cast(ConfigDict, OmegaConf.create({"platoon": platoon}))
                cav_config = cast(ConfigDict, OmegaConf.merge(self.scenario_params["vehicle_base"], platoon_base, cav_config))
                if "spawn_special" not in cav_config:
                    spawn_transform = carla.Transform(
                        carla.Location(x=cav_config["spawn_position"][0], y=cav_config["spawn_position"][1], z=cav_config["spawn_position"][2]),
                        carla.Rotation(
                            pitch=cav_config["spawn_position"][5], yaw=cav_config["spawn_position"][4], roll=cav_config["spawn_position"][3]
                        ),
                    )
                else:
                    spawn_transform = cast(MapHelper, map_helper)(self.carla_version, *cav_config["spawn_special"])

                vehicle = self.spawn_custom_actor(spawn_transform, cav_config, fallback_model)

                agent_manager = AgentManager.create(
                    actor=vehicle,
                    config_yaml=cav_config,
                    carla_map=self.carla_map,
                    cav_world=self.cav_world,
                    agent_type=AgentType.CAV,
                    application=["platoon"],
                    current_time=self.scenario_params["current_time"],
                    perception_requirements=perception_requirements,
                    id_prefix="platoon",
                )

                platoon_carla_ids[vehicle.id] = agent_manager.id

                if j == 0:
                    platoon_manager.set_lead(agent_manager)
                else:
                    platoon_manager.add_member(agent_manager, leader=False)

            destination = carla.Location(x=platoon["destination"][0], y=platoon["destination"][1], z=platoon["destination"][2])

            platoon_manager.set_destination(destination)
            platoon_manager.update_member_order()
            platoon_list.append(platoon_manager)

        return platoon_list, platoon_carla_ids

    def create_rsu_agents(self, perception_requirements: PerceptionRequirements | None = None) -> tuple[list[AgentManager], dict[int, Any]]:
        """
        Create a list of RSU.

        Parameters
        ----------
        perception_requirements : PerceptionRequirements
            Sensor/runtime requirements used to initialize perception-related modules.

        Returns
        -------
        rsu_list : list
            A list contains all rsu managers..
        """
        rsu_list: list[AgentManager] = []
        rsu_carla_ids: dict[int, Any] = {}
        perception_requirements = perception_requirements or PerceptionRequirements()

        if self.scenario_params.get("scenario") is None or self.scenario_params["scenario"].get("rsu_list", None) is None:
            logger.info("No RSU was created")
            return rsu_list, rsu_carla_ids

        for rsu_config in self.scenario_params["scenario"]["rsu_list"]:
            rsu_config = cast(ConfigDict, OmegaConf.merge(self.scenario_params["rsu_base"], rsu_config))
            default_model = "static.prop.gnome"
            static_bp = self.world.get_blueprint_library().find(default_model)

            spawn_transform = carla.Transform(
                carla.Location(x=rsu_config["spawn_position"][0], y=rsu_config["spawn_position"][1], z=rsu_config["spawn_position"][2]),
                carla.Rotation(pitch=rsu_config["spawn_position"][5], yaw=rsu_config["spawn_position"][4], roll=rsu_config["spawn_position"][3]),
            )

            actor = self.world.spawn_actor(static_bp, spawn_transform)

            try:
                agent_manager = AgentManager.create(
                    actor=actor,
                    config_yaml=rsu_config,
                    carla_map=self.carla_map,
                    cav_world=self.cav_world,
                    agent_type=AgentType.RSU,
                    current_time=self.scenario_params["current_time"],
                    perception_requirements=perception_requirements,
                )
            except Exception:
                actor.destroy()
                raise

            rsu_carla_ids[actor.id] = agent_manager.id

            rsu_list.append(agent_manager)
            logger.info(f"Created RSU with id {agent_manager.id}")

        return rsu_list, rsu_carla_ids

    def spawn_vehicles_by_list(self, tm: carla.TrafficManager, traffic_config: ConfigDict, bg_list: list[carla.Actor]) -> list[carla.Actor]:
        """
        Spawn the traffic vehicles by the given list.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        bg_list : list
            The list contains all background traffic.

        Returns
        -------
        bg_list : list
            Update traffic list.
        """

        blueprint_library = self.world.get_blueprint_library()
        if not self.use_multi_class_bp:
            ego_vehicle_random_list = car_blueprint_filter(blueprint_library, self.carla_version)
        else:
            label_list = list(self.bp_class_sample_prob.keys())
            prob = [self.bp_class_sample_prob[itm] for itm in label_list]

        # if not random select, we always choose lincoln.mkz with green color
        color = "0, 255, 0"
        default_model = "vehicle.lincoln.mkz_2020"
        ego_vehicle_bp = blueprint_library.find(default_model)

        for i, vehicle_config in enumerate(traffic_config["vehicle_list"]):
            vehicle_config = cast(ConfigDict, vehicle_config)
            spawn_transform = carla.Transform(
                carla.Location(x=vehicle_config["spawn_position"][0], y=vehicle_config["spawn_position"][1], z=vehicle_config["spawn_position"][2]),
                carla.Rotation(
                    pitch=vehicle_config["spawn_position"][5], yaw=vehicle_config["spawn_position"][4], roll=vehicle_config["spawn_position"][3]
                ),
            )

            if not traffic_config["random"]:
                ego_vehicle_bp.set_attribute("color", color)

            else:
                # sample a bp from various classes
                if self.use_multi_class_bp:
                    label = np.random.choice(label_list, p=prob)
                    # Given the label (class), find all associated blueprints in CARLA
                    ego_vehicle_random_list = multi_class_vehicle_blueprint_filter(label, blueprint_library, self.bp_meta)
                ego_vehicle_bp = random.choice(ego_vehicle_random_list)

                if ego_vehicle_bp.has_attribute("color"):
                    color = random.choice(ego_vehicle_bp.get_attribute("color").recommended_values)
                    ego_vehicle_bp.set_attribute("color", color)

            vehicle = self.world.spawn_actor(ego_vehicle_bp, spawn_transform)
            vehicle.set_autopilot(True, 8000)

            if "vehicle_speed_perc" in vehicle_config:
                tm.vehicle_percentage_speed_difference(vehicle, vehicle_config["vehicle_speed_perc"])
            tm.auto_lane_change(vehicle, traffic_config["auto_lane_change"])

            bg_list.append(vehicle)

        return bg_list

    def spawn_vehicle_by_range(self, tm: carla.TrafficManager, traffic_config: ConfigDict, bg_list: list[carla.Actor]) -> list[carla.Actor]:
        """
        Spawn the traffic vehicles by the given range.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        bg_list : list
            The list contains all background traffic.

        Returns
        -------
        bg_list : list
            Update traffic list.
        """
        blueprint_library = self.world.get_blueprint_library()
        if not self.use_multi_class_bp:
            ego_vehicle_random_list = car_blueprint_filter(blueprint_library, self.carla_version)
        else:
            label_list = list(self.bp_class_sample_prob.keys())
            prob = [self.bp_class_sample_prob[itm] for itm in label_list]

        # if not random select, we always choose lincoln.mkz with green color
        color = "0, 255, 0"
        default_model = "vehicle.lincoln.mkz_2020"
        ego_vehicle_bp = blueprint_library.find(default_model)

        spawn_ranges = traffic_config["range"]
        spawn_set: set[tuple[float, float, float, float, float, float]] = set()
        spawn_num = 0
        cav_spawn_clearance = float(traffic_config.get("cav_spawn_clearance", 0))
        if cav_spawn_clearance < 0:
            raise ValueError("Config key 'cav_spawn_clearance' must be non-negative.")
        cav_locations = (
            [manager.agent.vehicle.get_location() for manager in self.cav_world.get_vehicle_agent_managers().values()]
            if cav_spawn_clearance > 0
            else []
        )

        for spawn_range in spawn_ranges:
            spawn_range = cast(list[Any], spawn_range)
            spawn_num += cast(int, spawn_range[6])
            x_min, x_max, y_min, y_max = math.floor(spawn_range[0]), math.ceil(spawn_range[1]), math.floor(spawn_range[2]), math.ceil(spawn_range[3])

            for x in range(x_min, x_max, int(spawn_range[4])):
                for y in range(y_min, y_max, int(spawn_range[5])):
                    location = carla.Location(x=x, y=y, z=0.3)
                    way_point = self.carla_map.get_waypoint(location).transform

                    spawn_set.add(
                        (
                            way_point.location.x,
                            way_point.location.y,
                            way_point.location.z,
                            way_point.rotation.roll,
                            way_point.rotation.yaw,
                            way_point.rotation.pitch,
                        )
                    )
        count = 0
        spawn_list = list(spawn_set)
        shuffle(spawn_list)

        for coordinates in spawn_list:
            if count >= spawn_num:
                break

            spawn_transform = carla.Transform(
                carla.Location(x=coordinates[0], y=coordinates[1], z=coordinates[2] + 0.3),
                carla.Rotation(roll=coordinates[3], yaw=coordinates[4], pitch=coordinates[5]),
            )
            if any(spawn_transform.location.distance(cav_location) < cav_spawn_clearance for cav_location in cav_locations):
                continue

            if not traffic_config["random"]:
                ego_vehicle_bp.set_attribute("color", color)

            else:
                # sample a bp from various classes
                if self.use_multi_class_bp:
                    label = np.random.choice(label_list, p=prob)
                    # Given the label (class), find all associated blueprints in CARLA
                    ego_vehicle_random_list = multi_class_vehicle_blueprint_filter(label, blueprint_library, self.bp_meta)
                    if not ego_vehicle_random_list:
                        logger.warning(f"No blueprints found for label: {label}, using default blueprint")
                        ego_vehicle_random_list = blueprint_library.filter("vehicle.*")
                ego_vehicle_bp = random.choice(ego_vehicle_random_list)
                if ego_vehicle_bp.has_attribute("color"):
                    color = random.choice(ego_vehicle_bp.get_attribute("color").recommended_values)
                    ego_vehicle_bp.set_attribute("color", color)

            vehicle = self.world.try_spawn_actor(ego_vehicle_bp, spawn_transform)

            if not vehicle:
                continue

            vehicle.set_autopilot(True, 8000)
            self._configure_traffic_manager_vehicle(tm, traffic_config, vehicle, randomize_speed=True)

            bg_list.append(vehicle)
            count += 1

        if count < spawn_num:
            logger.warning(
                "Spawned %d of %d requested range vehicles; no more valid spawn points are available (cav_spawn_clearance=%.1f m).",
                count,
                spawn_num,
                cav_spawn_clearance,
            )

        return bg_list

    def create_traffic_carla(self) -> tuple[carla.TrafficManager | None, list[carla.Actor]]:
        """
        Create traffic flow.

        Returns
        -------
        tm : carla.traffic_manager
            Carla traffic manager.

        bg_list : list
            The list that contains all the background traffic vehicles.
        """
        bg_list: list[carla.Actor] = []

        if self.scenario_params.get("carla_traffic_manager") is None:
            logger.info("No Carla traffic flow was created")
            return None, bg_list

        traffic_config = cast(ConfigDict, self.scenario_params["carla_traffic_manager"])
        tm = self.client.get_trafficmanager()

        if isinstance(traffic_config["vehicle_list"], list) or isinstance(traffic_config["vehicle_list"], ListConfig):
            bg_list = self.spawn_vehicles_by_list(tm, traffic_config, bg_list)
        else:
            bg_list = self.spawn_vehicle_by_range(tm, traffic_config, bg_list)

        if not bg_list:
            logger.info("No Carla traffic flow was created")
        else:
            logger.info("CARLA traffic flow generated")
        return tm, bg_list

    def tick(self) -> int:
        """
        Tick the server.
        """
        return self.world.tick()

    def sumo_tick(self) -> None:
        return None

    # TODO: Use this function instead of destroy in scenario.py
    # NOTE: This function crashes Carla
    def destroyActors(self) -> None:  # noqa: DC04
        """
        Destroy all actors in the world.
        """

        actor_list = self.world.get_actors()
        for actor in actor_list:
            actor.destroy()

    def close(self) -> None:
        """
        Simulation close.
        """
        # restore to origin setting
        self.world.apply_settings(self.origin_settings)
