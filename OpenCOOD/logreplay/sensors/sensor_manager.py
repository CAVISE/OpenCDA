"""
Sensor management for cooperative autonomous vehicles in CARLA simulation.

This module provides the SensorManager class that handles sensor data streaming,
visualization, and data dumping for each connected autonomous vehicle (CAV)
during log replay operations.
"""

import importlib
import os
from collections import OrderedDict

from typing import Dict, Any
import carla


class SensorManager:
    """
    The manager controls all sensor data streaming and dumping for each cav.
    todo: rsu not considered yet.

    Parameters
    ----------
    agent_id : str
        The cav's original id.

    vehicle_content : dict
        The content of the cav.

    world : carla.World
        Carla simulation server object.

    config_yaml : dict
        Configurations for the sensor manager.

    output_root : str
        Output directory for data dumping.

    Attributes
    ----------
    agent_id : str
        The CAV's unique identifier.
    output_root : str
        Root output directory for data dumping.
    vehicle : carla.Actor
        CARLA vehicle actor instance.
    world : carla.World
        CARLA world instance.
    sensor_list : List
        List of instantiated sensor objects for this vehicle.
    sensor_meta : OrderedDict
        Dictionary storing metadata returned from sensors, keyed by sensor name.
    """

    def __init__(self, agent_id: str, vehicle_content: Dict[str, Any], world: carla.World, config_yaml: Dict[str, Any], output_root: str):
        self.agent_id = agent_id
        self.output_root = output_root
        self.vehicle = vehicle_content["actor"]
        self.world = world
        self.sensor_list = []
        # this is used to gather the meta information return from sensors
        self.sensor_meta: OrderedDict = OrderedDict()

        for sensor_content in config_yaml["sensor_list"]:
            sensor = None
            sensor_name = sensor_content["name"]

            # find the relative path
            sensor_filename = "logreplay.sensors." + sensor_name
            sensor_lib = importlib.import_module(sensor_filename)
            target_sensor_name = sensor_name.replace("_", "")

            # the sensor corresponding class has to have the same
            # name pattern as the file
            for name, cls in sensor_lib.__dict__.items():
                if name.lower() == target_sensor_name.lower():
                    sensor = cls

            assert sensor is not None, "The sensor class name has to be thesame as the file name. e.g. bev_semantic_camera -> BevSemanticCamera"
            # todo: rsu is not considered yet
            sensor_instance = sensor(self.agent_id, self.vehicle, self.world, sensor_content["args"], None)
            self.sensor_list.append(sensor_instance)

    def run_step(self, cur_timestamp: str) -> None:
        """
        Execute one step of sensor data collection and dumping.

        For each sensor, this method visualizes data, collects metadata,
        and dumps data to disk with the current timestamp.

        Parameters
        ----------
        cur_timestamp : str
            Current simulation timestamp for data association and file naming.
        """
        for sensor_instance in self.sensor_list:
            sensor_name = sensor_instance.name
            sensor_instance.visualize_data()

            meta_info = sensor_instance.tick()
            self.sensor_meta.update({sensor_name: meta_info})

            # for data dumping
            output_folder = os.path.join(self.output_root, self.agent_id)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            sensor_instance.data_dump(output_folder, cur_timestamp)

    def destroy(self) -> None:
        """
        Destroy all managed sensor instances.

        Calls the destroy method on each sensor to properly clean up CARLA
        actors and free resources.
        """
        for sensor_instance in self.sensor_list:
            sensor_instance.destroy()
