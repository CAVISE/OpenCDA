"""
Customized perception manager for vehicle sensor processing.

This module provides a template for implementing custom perception algorithms
that process camera and LiDAR data for object detection in autonomous driving.
"""

from typing import Dict, List, Any
import numpy.typing as npt
import cv2
import numpy as np
from opencda.core.sensing.perception.perception_manager import PerceptionManager
from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import TrafficLight
from opencda.core.sensing.localization.localization_manager import CustomizedLocalizationManager


class CustomziedPeceptionManager(PerceptionManager):
    """
    Customized perception manager for multi-sensor object detection.

    Extends the base PerceptionManager to allow custom perception algorithms
    using camera images and LiDAR point clouds.

    Parameters
    ----------
    vehicle : Any
        CARLA vehicle actor.
    config_yaml : Dict[str, Any]
        Configuration dictionary from YAML file.
    cav_world : Any
        Connected and automated vehicle world object.
    data_dump : bool, optional
        Whether to dump sensor data to disk. Default is False.

    Attributes
    ----------
    Inherits all attributes from PerceptionManager.
    """

    def __init__(
        self,
        vehicle: Any,
        config_yaml: Dict[str, Any],
        cav_world: Any,
        data_dump: bool = False,
    ):
        super(CustomizedLocalizationManager, self).__init__(vehicle, config_yaml, cav_world, data_dump)

    def detect(self, ego_pos: Any) -> Dict[str, List[Any]]: #NOTE The function repeats twice
        def detect(self, ego_pos: Any) -> Dict[str, List[Any]]:
            """
            Perform object detection using sensor data.

            Processes RGB camera images and LiDAR data to detect vehicles,
            traffic lights, and other objects in the environment.

            Parameters
            ----------
            ego_pos : Any
                Ego vehicle position for coordinate transformations.

            Returns
            -------
            Dict[str, List[Any]]
                Dictionary containing detected objects with keys:
                - 'vehicles': List of ObstacleVehicle objects
                - 'traffic_lights': List of TrafficLight objects
                - 'other_objects_you_wanna_add': List of additional detected objects
            """

        objects: Dict[str, List[Any]] = {"vehicles": [], "traffic_lights": [], "other_objects_you_wanna_add": []}

        # retrieve current rgb images from all cameras
        rgb_images: List[npt.NDArray[np.uint8]] = []
        for rgb_camera in self.rgb_camera:
            while rgb_camera.image is None:
                continue
            rgb_images.append(cv2.cvtColor(np.array(rgb_camera.image), cv2.COLOR_BGR2RGB))

        # retrieve lidar data from the sensor
        # lidar_data = self.lidar.data

        ########################################
        # this is where you put your algorithm #
        ########################################
        # objects = your_algorithm(rgb_images, lidar_data)
        assert isinstance(type(objects["vehicles"]), ObstacleVehicle)
        assert isinstance(type(objects["traffic_lights"]), TrafficLight)

        return objects
