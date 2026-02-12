"""
Scene manager for CARLA-based log replay of cooperative driving scenarios.

This module provides the SceneManager class that handles spawning, moving,
and destroying vehicles during log replay, along with sensor data collection
and HD map generation for each scene.
"""

import os
import random
import sys
from collections import OrderedDict

import carla
import numpy as np

from logreplay.assets.utils import find_town, find_blue_print
from logreplay.assets.presave_lib import bcolors
from logreplay.map.map_manager import MapManager
from logreplay.sensors.sensor_manager import SensorManager
from opencood.hypes_yaml.yaml_utils import load_yaml
from typing import Dict, List, Any, Optional, cast


class SceneManager:
    """
    Manager for each scene for spawning, moving and destroying.

    Parameters
    ----------
    folder : str
        The folder to the current scene.

    scene_name : str
        The scene's name.

    collection_params : dict
        The collecting protocol information.

    scenario_params : dict
        The replay params.

    Attributes
    ----------
    scene_name : str
        Current scene name.
    town_name : str
        CARLA town name
    database : OrderedDict
        Nested dict
    timestamps : list of str
        Sorted list of all timestamps in the scene.
    cav_id_list : list of str
        List of all CAV IDs in the scene.
    veh_dict : OrderedDict
        Dynamic storage for spawned vehicle objects.
    cur_count : int
        Current timestamp index.
    client : carla.Client
        CARLA simulator client.
    world : carla.World
        CARLA world instance.
    map_manager : MapManager
        HD map generation manager.
    """

    def __init__(self, folder: str, scene_name: str, collection_params: Dict[str, Any], scenario_params: Dict[str, Any]):
        self.scene_name = scene_name
        self.town_name = find_town(scene_name)
        self.collection_params = collection_params
        self.scenario_params = scenario_params
        self.cav_id_list: List[str] = []

        # dumping related
        self.output_root = os.path.join(scenario_params["output_dir"], scene_name)

        if "seed" in collection_params["world"]:
            np.random.seed(collection_params["world"]["seed"])
            random.seed(collection_params["world"]["seed"])

        # at least 1 cav should show up
        cav_list = sorted([x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))])
        assert len(cav_list) > 0

        self.database: OrderedDict = OrderedDict()
        # we want to save timestamp as the parent keys for cavs
        cav_sample = os.path.join(folder, cav_list[0])

        yaml_files = sorted([os.path.join(cav_sample, x) for x in os.listdir(cav_sample) if x.endswith(".yaml") and "additional" not in x])
        self.timestamps = self.extract_timestamps(yaml_files)

        # loop over all timestamps
        for timestamp in self.timestamps:
            self.database[timestamp] = OrderedDict()
            # loop over all cavs
            for j, cav_id in enumerate(cav_list):
                if cav_id not in self.cav_id_list:
                    self.cav_id_list.append(str(cav_id))

                self.database[timestamp][cav_id] = OrderedDict()
                cav_path = os.path.join(folder, cav_id)

                yaml_file = os.path.join(cav_path, timestamp + ".yaml")
                self.database[timestamp][cav_id]["yaml"] = yaml_file

        # this is used to dynamically save all information of the objects
        self.veh_dict: OrderedDict = OrderedDict()
        # used to count timestamp
        self.cur_count = 0
        # used for HDMap
        self.map_manager: Optional[MapManager] = None

    def start_simulator(self) -> None:
        """
        Connect to the CARLA simulator and initialize world for log replay.

        Establishes client connection, loads the appropriate map, configures
        synchronous mode with fixed time step, applies weather settings, and
        initializes the HD map manager.

        Raises
        ------
        RuntimeError
            If the specified town map is not found in CARLA.
        SystemExit
            If world loading fails.
        """
        simulation_config = self.collection_params["world"]

        # simulation sync mode time step
        fixed_delta_seconds = simulation_config["fixed_delta_seconds"]
        weather_config = simulation_config["weather"] if "weather" in simulation_config else None

        # setup the carla client
        self.client = carla.Client("localhost", simulation_config["client_port"])
        self.client.set_timeout(10.0)

        # load the map
        if self.town_name != "Culver_City":
            try:
                self.world = self.client.load_world(self.town_name)
            except RuntimeError:
                print(
                    f"{bcolors.FAIL} %s is not found in your CARLA repo! "
                    f"Please download all town maps to your CARLA "
                    f"repo!{bcolors.ENDC}" % self.town_name
                )
        else:
            self.world = self.client.get_world()

        if not self.world:
            sys.exit("World loading failed")

        # setup the new setting
        self.origin_settings = self.world.get_settings()
        new_settings = self.world.get_settings()

        new_settings.synchronous_mode = True  # noqa: DC05
        new_settings.fixed_delta_seconds = fixed_delta_seconds

        self.world.apply_settings(new_settings)
        # set weather if needed
        if weather_config is not None:
            weather = self.set_weather(weather_config)
            self.world.set_weather(weather)
        # get map
        self.carla_map = self.world.get_map()
        # spectator
        self.spectator = self.world.get_spectator()
        # hd map manager per scene
        self.map_manager = MapManager(self.world, self.scenario_params["map"], self.output_root, self.scene_name)

    def tick(self) -> bool:
        """
        Execute one simulation step: spawn/move vehicles and collect data.

        Processes current timestamp by spawning new vehicles, moving existing
        ones, managing sensors, and collecting data. Advances simulation by
        one tick.

        Returns
        -------
        success : bool
            True if tick executed successfully, False if no more timestamps
            remain (end of scenario).
        """
        if self.cur_count >= len(self.timestamps):
            return False

        cur_timestamp = self.timestamps[self.cur_count]
        cur_database = self.database[cur_timestamp]

        for i, (cav_id, cav_yml) in enumerate(cur_database.items()):
            cav_content = load_yaml(cav_yml["yaml"])
            if cav_id not in self.veh_dict:
                self.spawn_cav(cav_id, cav_content, cur_timestamp)
            else:
                self.move_vehicle(cav_id, cur_timestamp, self.structure_transform_cav((cav_content["true_ego_pos"])))

            self.veh_dict[cav_id]["cav"] = True
            # spawn the sensor on each cav
            if "sensor_manager" not in self.veh_dict[cav_id]:
                self.veh_dict[cav_id]["sensor_manager"] = SensorManager(
                    cav_id, self.veh_dict[cav_id], self.world, self.scenario_params["sensor"], self.output_root
                )

            # set the spectator to the first cav
            if i == 0:
                transform = self.structure_transform_cav(cav_content["true_ego_pos"])
                self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=70), carla.Rotation(pitch=-90)))

            for bg_veh_id, bg_veh_content in cav_content["vehicles"].items():
                if str(bg_veh_id) not in self.veh_dict:
                    self.spawn_bg_vehicles(bg_veh_id, bg_veh_content, cur_timestamp)
                else:
                    self.move_vehicle(
                        str(bg_veh_id), cur_timestamp, self.structure_transform_bg_veh(bg_veh_content["location"], bg_veh_content["angle"])
                    )
        # remove the vehicles that are not in any cav's scope
        self.destroy_vehicle(cur_timestamp)

        self.cur_count += 1
        self.world.tick()

        # we dump data after tick() so the agent can retrieve the newest info
        self.sensor_dumping(cur_timestamp)
        self.map_dumping()

        return True

    def map_dumping(self) -> None:
        """
        Generate and save BEV maps for all CAVs.

        Invokes HDMap manager to rasterize and save bird's eye view maps
        for each connected autonomous vehicle in the current frame.
        """
        if self.map_manager is None:
            return
        for veh_id, veh_content in self.veh_dict.items():
            if "cav" in veh_content:
                self.map_manager.run_step(veh_id, veh_content, self.veh_dict)

    def sensor_dumping(self, cur_timestamp: str) -> None:
        """
        Trigger sensor data collection for all equipped vehicles.

        Parameters
        ----------
        cur_timestamp : str
            Current simulation timestamp for data association.
        """
        for veh_id, veh_content in self.veh_dict.items():
            if "sensor_manager" in veh_content:
                veh_content["sensor_manager"].run_step(cur_timestamp)

    def spawn_cav(self, cav_id: str, cav_content: Dict[str, Any], cur_timestamp: str) -> None:
        """
        Spawn the cav based on current content.

        Parameters
        ----------
        cav_id : str
            The saved cav_id.

        cav_content : dict
            The information in the cav's folder.

        cur_timestamp : str
            This is used to judge whether this vehicle has been already
            called in this timestamp.
        """

        # cav always use lincoln
        model = "vehicle.lincoln.mkz_2017"

        # retrive the blueprint library
        blueprint_library = self.world.get_blueprint_library()
        cav_bp = blueprint_library.find(model)
        # cav is always green
        color = "0, 0, 255"
        cav_bp.set_attribute("color", color)

        cur_pose = cav_content["true_ego_pos"]
        # convert to carla needed format
        cur_pose = self.structure_transform_cav(cur_pose)

        # spawn the vehicle
        vehicle = self.world.try_spawn_actor(cav_bp, cur_pose)

        while not vehicle:
            cur_pose.location.z += 0.01
            vehicle = self.world.try_spawn_actor(cav_bp, cur_pose)

        self.veh_dict.update(
            {
                str(cav_id): {
                    "cur_pose": cur_pose,
                    "model": model,
                    "color": color,
                    "actor_id": vehicle.id,
                    "actor": vehicle,
                    "cur_count": cur_timestamp,
                }
            }
        )

    def spawn_bg_vehicles(self, bg_veh_id: str, bg_veh_content: Dict[str, Any], cur_timestamp: str) -> None:
        """
        Spawn the background vehicle.

        Parameters
        ----------
        bg_veh_id : str
            The id of the bg vehicle.
        bg_veh_content : dict
            The contents of the bg vehicle
        cur_timestamp : str
            This is used to judge whether this vehicle has been already
            called in this timestamp.
        """
        # retrieve the blueprint library
        blueprint_library = self.world.get_blueprint_library()

        cur_pose = self.structure_transform_bg_veh(bg_veh_content["location"], bg_veh_content["angle"])
        model: str
        if str(bg_veh_id) in self.cav_id_list:
            model = "vehicle.lincoln.mkz_2017"
            veh_bp = blueprint_library.find(model)
            color = "0, 0, 255"
        else:
            model_candidate = find_blue_print(bg_veh_content["extent"])
            if not model_candidate:
                print("model net found for %s" % bg_veh_id)
                model_candidate = "vehicle.lincoln.mkz_2017"
            model = model_candidate
            veh_bp = blueprint_library.find(model)

            color = random.choice(veh_bp.get_attribute("color").recommended_values)

        veh_bp.set_attribute("color", color)

        # spawn the vehicle
        vehicle = self.world.try_spawn_actor(veh_bp, cur_pose)

        while not vehicle:
            cur_pose.location.z += 0.01
            vehicle = self.world.try_spawn_actor(veh_bp, cur_pose)

        self.veh_dict.update(
            {
                str(bg_veh_id): {
                    "cur_pose": cur_pose,
                    "model": model,
                    "color": color,
                    "actor_id": vehicle.id,
                    "actor": vehicle,
                    "cur_count": cur_timestamp,
                }
            }
        )

    def move_vehicle(self, veh_id: str, cur_timestamp: str, transform: carla.Transform) -> None:
        """
        Updates vehicle position if it hasn't been moved in current timestamp.
        Prevents duplicate movements within the same simulation tick.

        Parameters
        ----------
        veh_id : str
            Vehicle's unique identifier.
        cur_timestamp : str
            Current timestamp to check if vehicle already moved.
        transform : carla.Transform
            Target pose (location + rotation) for the vehicle.
        """
        # this represent this vehicle is already moved in this round
        if self.veh_dict[veh_id]["cur_count"] == cur_timestamp:
            return

        self.veh_dict[veh_id]["actor"].set_transform(transform)
        self.veh_dict[veh_id]["cur_count"] = cur_timestamp
        self.veh_dict[veh_id]["cur_pose"] = transform

    def close(self) -> None:
        self.world.apply_settings(self.origin_settings)
        actor_list = self.world.get_actors()
        for actor in actor_list:
            actor = cast(carla.Actor, actor)
            actor.destroy()
        cast(MapManager, self.map_manager).destroy()  # Tell type checker that self.map_manager is MapManager
        self.sensor_destory()

    def sensor_destory(self) -> None:
        for veh_id, veh_content in self.veh_dict.items():
            if "sensor_manager" in veh_content:
                veh_content["sensor_manager"].destroy()

    def destroy_vehicle(self, cur_timestamp: str) -> None:
        """
        Destroy vehicles that are out of scope of all CAVs.

        Removes vehicles that haven't been updated in the current timestamp,
        indicating they're no longer within any CAV's perception range.

        Parameters
        ----------
        cur_timestamp : str
            Current timestamp to compare against vehicle's last update time.
        """
        destroy_list = []
        for veh_id, veh_content in self.veh_dict.items():
            if veh_content["cur_count"] != cur_timestamp:
                veh_content["actor"].destroy()
                destroy_list.append(veh_id)

        for veh_id in destroy_list:
            self.veh_dict.pop(veh_id)

    def structure_transform_cav(self, pose: List[float]) -> carla.Transform:
        """
        Convert CAV pose list to CARLA Transform.

        Parameters
        ----------
        pose : list of float
        Pose as [x, y, z, roll, yaw, pitch].
        - x, y, z : float
        Position in meters.
        - roll, yaw, pitch : float
        Rotation in degrees.

        Returns
        -------
        transform : carla.Transform
        CARLA transform object with location and rotation.
        """
        cur_pose = carla.Transform(carla.Location(x=pose[0], y=pose[1], z=pose[2]), carla.Rotation(roll=pose[3], yaw=pose[4], pitch=pose[5]))

        return cur_pose

    @staticmethod
    def structure_transform_bg_veh(location: List[float], rotation: List[float]) -> carla.Transform:
        """
        Convert location and rotation lists to CARLA Transform.

        Parameters
        ----------
        location : list of float
            Position as [x, y, z] in meters.
        rotation : list of float
            Rotation as [roll, yaw, pitch] in degrees.

        Returns
        -------
        transform : carla.Transform
            CARLA transform object with location and rotation.
        """
        cur_pose = carla.Transform(
            carla.Location(x=location[0], y=location[1], z=location[2]), carla.Rotation(roll=rotation[0], yaw=rotation[1], pitch=rotation[2])
        )

        return cur_pose

    @staticmethod
    def extract_timestamps(yaml_files: List[str]) -> List[str]:
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split("/")[-1]

            timestamp = res.replace(".yaml", "")
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def set_weather(weather_settings: Dict[str, float]) -> carla.WeatherParameters:
        """
        Set CARLA weather params.

        Parameters
        ----------
        weather_settings : dict
            The dictionary that contains all parameters of weather.

        Returns
        -------
        weather : carla.WeatherParameters
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
