import cv2
import numpy as np
from opencda.core.sensing.perception.perception_manager import PerceptionManager, PerceptionRequirements
from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import TrafficLight


class CustomziedPeceptionManager(PerceptionManager):
    def __init__(self, vehicle, config_yaml, cav_world, data_dump=False):
        super().__init__(
            vehicle=vehicle,
            config_yaml=config_yaml,
            cav_world=cav_world,
            infra_id=vehicle.id,
            perception_requirements=PerceptionRequirements.from_runtime_flags(data_dump=data_dump),
        )

    def detect(self, ego_pos):
        objects = {"vehicles": [], "traffic_lights": [], "other_objects_you_wanna_add": []}

        # retrieve current rgb images from all cameras
        rgb_images = []
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
