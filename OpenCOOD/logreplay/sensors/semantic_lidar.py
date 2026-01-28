"""
Semantic LiDAR sensor for filtering objects within camera field of view.

This module provides a semantic LiDAR sensor implementation that identifies
and filters vehicles based on point cloud density, primarily used to determine
which objects are visible to the vehicle's perception system.
"""

import weakref
from typing import List, Optional, Dict, Any

import carla
import numpy as np
from logreplay.sensors.base_sensor import BaseSensor


class SemanticLidar(BaseSensor):
    """
    Semantic LiDAR sensor for vehicle detection and visibility filtering.

    Captures semantic point cloud data and filters vehicles based on point
    density to determine visibility.

    Parameters
    ----------
    agent_id : str
        Unique identifier for the agent.
    vehicle : Optional[carla.Actor]
        CARLA vehicle actor to attach sensor to. None for fixed position.
    world : carla.World
        CARLA world instance.
    config : Dict[str, Any]
        Sensor configuration with LiDAR parameters (fov, channels, range,
        points_per_second, rotation_frequency, relative_pose, thresh).
    global_position : Optional[List[float]]
        Global position [x, y, z] if not attached to vehicle.

    Attributes
    ----------
    sensor : carla.Actor
        CARLA semantic LiDAR sensor actor.
    name : str
        Sensor name with relative position.
    thresh : int
        Minimum points required to consider a vehicle visible.
    points : Optional[NDArray]
        Point cloud data (N, 3) array of [x, y, z].
    obj_idx : Optional[NDArray]
        Object instance IDs for each point.
    obj_tag : Optional[NDArray]
        Semantic tags for each point (10 = vehicle).
    """

    def __init__(
        self, agent_id: str, vehicle: Optional[carla.Actor], world: carla.World, config: Dict[str, Any], global_position: Optional[List[float]]
    ) -> None:
        super().__init__(agent_id, vehicle, world, config, global_position)

        if vehicle is not None:
            world = vehicle.get_world()

        self.agent_id = agent_id

        blueprint = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        # set attribute based on the configuration
        blueprint.set_attribute("upper_fov", str(config["upper_fov"]))
        blueprint.set_attribute("lower_fov", str(config["lower_fov"]))
        blueprint.set_attribute("channels", str(config["channels"]))
        blueprint.set_attribute("range", str(config["range"]))
        blueprint.set_attribute("points_per_second", str(config["points_per_second"]))
        blueprint.set_attribute("rotation_frequency", str(config["rotation_frequency"]))

        relative_position = config["relative_pose"]
        spawn_point = self.spawn_point_estimation(relative_position, global_position)
        self.name = "semantic_lidar" + str(relative_position)
        self.thresh = config["thresh"]

        if vehicle is not None:
            self.sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.points = None
        self.obj_idx = None
        self.obj_tag = None

        self.timestamp = None
        self.frame = 0

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: SemanticLidar._on_data_event(weak_self, event))

    @staticmethod
    def _on_data_event(weak_self: weakref.ref, event: carla.SensorData) -> None:
        """
        Callback for semantic LiDAR data reception.

        Processes incoming semantic LiDAR data, extracting point coordinates,
        object indices, and semantic tags.

        Parameters
        ----------
        weak_self : weakref.ref
            Weak reference to the SemanticLidar instance.
        event : carla.SensorData
            Semantic LiDAR data event from CARLA.

        Returns
        -------
        None
        """
        self = weak_self()
        if not self:
            return

        # shape:(n, 6)
        data = np.frombuffer(
            event.raw_data,
            dtype=np.dtype(
                [("x", np.float32), ("y", np.float32), ("z", np.float32), ("CosAngle", np.float32), ("ObjIdx", np.uint32), ("ObjTag", np.uint32)]
            ),
        )

        # (x, y, z, intensity)
        self.points = np.array([data["x"], data["y"], data["z"]]).T
        self.obj_tag = np.array(data["ObjTag"])
        self.obj_idx = np.array(data["ObjIdx"])

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp

    @staticmethod
    def spawn_point_estimation(relative_position: str, global_position: Optional[List[float]]) -> carla.Transform:
        """
        Calculate sensor spawn point based on mounting position.

        Determines the sensor's location and orientation relative to the vehicle
        or at a global position.

        Parameters
        ----------
        relative_position : str
            Mounting position: "front", "left", "right", or "back".
        global_position : Optional[List[float]]
            Global position [x, y, z] if not attached to vehicle. If None,
            position is relative to vehicle origin.

        Returns
        -------
        carla.Transform
            Spawn point with location and rotation for the sensor.
        """
        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)

        if global_position is not None:
            carla_location = carla.Location(x=global_position[0], y=global_position[1], z=global_position[2])
            pitch = -35

        if relative_position == "front":
            carla_location = carla.Location(x=carla_location.x + 2.5, y=carla_location.y, z=carla_location.z + 1.0)
            yaw = 0

        elif relative_position == "right":
            carla_location = carla.Location(x=carla_location.x + 0.0, y=carla_location.y + 0.3, z=carla_location.z + 1.8)
            yaw = 100

        elif relative_position == "left":
            carla_location = carla.Location(x=carla_location.x + 0.0, y=carla_location.y - 0.3, z=carla_location.z + 1.8)
            yaw = -100
        else:
            carla_location = carla.Location(x=carla_location.x - 2.0, y=carla_location.y, z=carla_location.z + 1.5)
            yaw = 180

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    def tick(self) -> Any: #NOTE Used 'Any' as hack for override mismatch (returns List[int] here, but base_sensor.py tick -> None)
        """
        Process current LiDAR data and return visible vehicle IDs.

        Filters vehicles based on point density threshold. Only vehicles with
        sufficient LiDAR points (above thresh) are considered visible.

        Returns
        -------
        List[int]
            List of unique vehicle instance IDs that meet the visibility
            threshold.
        """
        while self.obj_idx is None or self.obj_tag is None or self.obj_idx.shape[0] != self.obj_tag.shape[0]:
            continue

        # label 10 is the vehicle
        vehicle_idx = self.obj_idx[self.obj_tag == 10]
        # each individual instance id
        vehicle_unique_id = list(np.unique(vehicle_idx))
        vehicle_id_filter = []

        for veh_id in vehicle_unique_id:
            if vehicle_idx[vehicle_idx == veh_id].shape[0] > self.thresh:
                vehicle_id_filter.append(veh_id)

        # these are the ids that are visible
        return vehicle_id_filter
