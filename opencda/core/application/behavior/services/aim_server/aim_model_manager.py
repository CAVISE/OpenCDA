# from __future__ import annotations
from collections.abc import Sequence

import carla
import torch
import logging
import numpy as np
import pickle as pkl
from scipy.spatial import distance

from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper
from AIM import AIMModel

from .messages import AIMServerRequestMessage
from .results import AIMServerResult, AIMServerMessage

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_server.aim_model_manager")


class AIMModelManager:
    def __init__(
        self,
        model: AIMModel,
        control_center: carla.Transform,
        service_name: str,
        owner_id: str,
    ):
        """
        Initialize the standalone AIM model manager.

        Parameters
        ----------
        model : AIMModel
            Loaded AIM model used for trajectory prediction.
        control_center : carla.Transform
            Intersection control point used to normalize vehicle positions.
        service_name : str
            Service identifier used in AIM result messages.
        owner_id : str
            Identifier for the owner of the AIM model.
        """
        self.CONTROL_RADIUS = 50
        self.THRESHOLD = 10
        self.FORCE_VALUE = 20

        self.cav_data = dict()

        self.trajs = dict()

        control_center_location = BridgeHelper.get_sumo_transform(control_center, carla.Vector3D(0, 0, 0)).location

        self.control_center_coords = np.array([control_center_location.x, control_center_location.y])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

        self._service_name = service_name
        self._owner_id = owner_id

        self.yaw_dict = self._load_yaw()
        self.yaw_id = {}

    def _load_yaw(self):
        """
        Loads yaw dictionary from a predefined address.

        :return: yaw_dict
        """
        with open("opencda/assets/yaw_dict_10m.pkl", "rb") as f:
            return pkl.load(f)

    def _get_distance_to_center(self, curr_pos: np.ndarray) -> float:
        return np.linalg.norm(curr_pos - self.control_center_coords)

    def _get_distance_to_center_by_vid(self, vehicle_id: str) -> float:
        curr_pos = self._get_cav_sumo_pos(vehicle_id)
        return self._get_distance_to_center(curr_pos)

    def _preprocess_cav_data(self, msg: AIMServerRequestMessage) -> None:
        """
        Normalize and cache incoming CAV data for the current inference step.
        """
        cav_pos = BridgeHelper.get_sumo_transform(msg.position, carla.Vector3D(0, 0, 0)).location
        curr_pos = np.array([cav_pos.x, cav_pos.y])
        distance_to_center = self._get_distance_to_center(curr_pos)

        if distance_to_center != -1 and distance_to_center < self.CONTROL_RADIUS:
            self.cav_data[msg.vehicle_id] = {
                "intention": self.get_opencda_intention(msg.waypoints, self.control_center_coords),
                "pos": msg.position.location,
                "sumo_pos": curr_pos,
                "speed": msg.speed,
                "yaw": msg.yaw,
                "waypoints": msg.waypoints,
            }

    def _get_cav_pos(self, vehicle_id: str) -> np.ndarray:
        return self.cav_data[vehicle_id]["pos"]

    def _get_cav_sumo_pos(self, vehicle_id: str) -> np.ndarray:
        return self.cav_data[vehicle_id]["sumo_pos"]

    def _make_result(self, messages: tuple[AIMServerMessage, ...] = ()) -> AIMServerResult:
        return AIMServerResult(
            service_name=self._service_name,
            owner_id=self._owner_id,
            messages=messages,
        )

    def process(self, messages: Sequence[AIMServerRequestMessage]) -> AIMServerResult:
        """
        Run AIM inference for the request batch and return predicted targets.
        """
        for msg in messages:
            self._preprocess_cav_data(msg)
        result_messages: list[AIMServerMessage] = []

        self.update_trajs()

        features, target_agent_ids = self.encoding_scenario_features()
        num_agents = features.shape[0]

        if num_agents == 0:
            self.cav_data = dict()
            return self._make_result()

        predictions = self.model.predict(features.copy(), target_agent_ids)

        for idx in range(num_agents):
            vehicle_id = target_agent_ids[idx]

            distance_to_center = self._get_distance_to_center_by_vid(vehicle_id)
            if distance_to_center == -1:
                continue

            if distance_to_center < self.CONTROL_RADIUS:
                pred_delta = predictions[idx].reshape(30, 2).detach().cpu().numpy()
                local_delta = pred_delta[0].reshape(2, 1)
                last_delta = pred_delta[-1].reshape(2, 1)

                if last_delta[1, 0] <= 1:
                    local_delta[1, 0] = 1e-8
                    local_delta[0, 0] = 1e-10
                else:
                    local_delta[1, 0] = max(1e-8, local_delta[1, 0])

                yaw = features[idx][3]
                rotation = self.rotation_matrix_back(yaw)
                global_delta = (rotation @ local_delta).squeeze()
                global_delta[1] *= -1

                pos = self._get_cav_pos(vehicle_id)

                global_delta = np.where(np.abs(global_delta) <= self.THRESHOLD, np.sign(global_delta) * self.FORCE_VALUE, global_delta)

                next_loc = carla.Location(
                    x=pos.x + global_delta[0],
                    y=pos.y - global_delta[1],
                    z=pos.z,
                )
                result_messages.append(
                    AIMServerMessage(
                        service_name=self._service_name,
                        vehicle_id=vehicle_id,
                        next_position=next_loc,
                    )
                )

        self.cav_data = dict()
        return self._make_result(messages=tuple(result_messages))

    def update_trajs(self):
        """
        Updates the self.trajs dictionary, which stores the trajectory history of each vehicle.
        Format:
        {
            'vehicle_id': [
                (rel_x, rel_y, speed, yaw_rad, yaw, intention),
                ...
            ],
            ...
        }
        """
        logger.debug(f"Updating trajectories for cavs: {self.cav_data.keys()}")
        for vehicle_id in self.cav_data:
            position = self._get_cav_sumo_pos(vehicle_id)

            distance_to_center = np.linalg.norm(position - self.control_center_coords)

            if distance_to_center < self.CONTROL_RADIUS:
                # Initialize trajectory if this is a new vehicle
                if vehicle_id not in self.trajs:
                    self.trajs[vehicle_id] = []

                # Get vehicle state
                speed = self.cav_data[vehicle_id]["speed"]
                yaw = self.cav_data[vehicle_id]["yaw"]
                yaw_rad = np.deg2rad(self.get_yaw(vehicle_id, position, self.yaw_dict))

                # Normalize position relative to control node
                node_x, node_y = self.control_center_coords
                rel_x = position[0] - node_x
                rel_y = position[1] - node_y

                if not self.trajs[vehicle_id] or self.trajs[vehicle_id][-1][-1] == "null":
                    intention = self.get_intention(vehicle_id)
                else:
                    intention = self.trajs[vehicle_id][-1][-1]

                self.trajs[vehicle_id] = [(rel_x, rel_y, speed, yaw_rad, yaw, intention)]

        for vehicle_id in list(self.trajs):
            if vehicle_id not in self.cav_data:
                del self.trajs[vehicle_id]

    def _get_intention_by_rotation(self, rotation: int) -> str:
        """
        Distinguishes vehicle intention by its rotation

        :param rotation: rotation degrees (from 0 to 360)
        :return: intention
        """
        if rotation < 30 or rotation > 330:
            intention = "straight"
        elif rotation < 135:
            intention = "right"
        elif rotation > 225:
            intention = "left"
        else:
            intention = "null"
        return intention

    def _get_distance(self, waypoint1, waypoint2):
        """
        Calculates Euclidean distance between two waypoints

        :param waypoint1: waypoint 2D-coordinates
        :param waypoint2: waypoint 2D-coordinates
        :return: distance
        """
        rel_x = waypoint1.location.x - waypoint2[0].transform.location.x
        rel_y = waypoint1.location.y - waypoint2[0].transform.location.y
        position = np.array([rel_x, rel_y])
        return np.linalg.norm(position)

    def get_opencda_intention(self, waypoints, mid):
        """
        Gets intention by averaged rotation to pass 3 next waypoints.

        :param waypoints: list of waypoint 2D-coordinates
        :param mid: middle of the turning path 2D-coordinates
        :param radius: search radius for waypoints nearby
        :return: intention
        """
        # Too few waypoints
        if len(waypoints) < 2:
            return "null"

        waypoint_index = 0
        location = carla.Location(mid[0], mid[1], 0)
        rotation = carla.Rotation(0)

        in_sumo_transform = carla.Transform(location, rotation)
        mid = BridgeHelper.get_carla_transform(in_sumo_transform, carla.Vector3D(0, 0, 0))
        if self._get_distance(mid, waypoints[0]) > self.CONTROL_RADIUS:
            logger.debug("Car not int radius")
            return "null"
        while self._get_distance(mid, waypoints[waypoint_index]) > self.CONTROL_RADIUS:
            waypoint_index += 1
            if waypoint_index >= len(waypoints):
                logger.debug("No waypoints in radius")
                return "null"

        first_waypoint = waypoints[waypoint_index]
        first_waypoint_index = waypoint_index
        while (self._get_distance(mid, waypoints[waypoint_index]) <= self.CONTROL_RADIUS) and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1

        # average the values of several points to reduce noise
        mean_yaw = 0
        if waypoint_index - first_waypoint_index < 3 and waypoint_index > 3:
            logger.warning("Too few waypoints in radius")
        else:
            for i in range(3):
                mean_yaw += waypoints[waypoint_index - i][0].transform.rotation.yaw
        mean_yaw //= 3
        rotation = (mean_yaw - first_waypoint[0].transform.rotation.yaw + 360) % 360
        return self._get_intention_by_rotation(rotation)

    def get_intention(self, vehicle_id):
        if self.cav_data.get(vehicle_id) and self.cav_data[vehicle_id].get("intention"):
            return self.cav_data[vehicle_id]["intention"]
        return "null"

    def encoding_scenario_features(self):
        """
        Encodes data on CAV movement and intentions for processing by ML models

        :return: x: list((motion features, intention vector)), target: agent ids list(vehicle id)
        """
        features = []
        target_agent_ids = []

        for vehicle_id, trajectory in self.trajs.items():
            last_position = trajectory[-1]
            position = np.array(last_position[:2])
            distance_to_origin = np.linalg.norm(position)

            if distance_to_origin < self.CONTROL_RADIUS:
                motion_features = np.array(last_position[:-2])
                intention_vector = self.get_intention_vector(last_position[-1])
                feature_vector = np.concatenate((motion_features, intention_vector)).reshape(1, -1)

                features.append(feature_vector)
                target_agent_ids.append(vehicle_id)

        features = np.vstack(features) if features else np.empty((0, 7))
        return features, target_agent_ids

    @staticmethod
    def rotation_matrix_back(yaw):
        """
        Rotate back.
        https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
        """
        rotation = np.array([[np.cos(-np.pi / 2 + yaw), -np.sin(-np.pi / 2 + yaw)], [np.sin(-np.pi / 2 + yaw), np.cos(-np.pi / 2 + yaw)]])
        return rotation

    def get_end(self, start, intention):
        """
        Determines the direction of exit from the turn given its start and vehicle intention.

        :param start: direction from which CAV enters the intersection
        :param intention: direction of turning relative to CAV movement
        :return: end
        """
        match intention:
            case "right":
                match start:
                    case "up":
                        return "left"
                    case "right":
                        return "up"
                    case "down":
                        return "right"
                    case "left":
                        return "down"
            case "left":
                match start:
                    case "up":
                        return "right"
                    case "right":
                        return "down"
                    case "down":
                        return "left"
                    case "left":
                        return "up"
            case "straight":
                match start:
                    case "up":
                        return "down"
                    case "right":
                        return "left"
                    case "down":
                        return "up"
                    case "left":
                        return "right"

    def get_yaw(self, vehicle_id: str, pos: np.ndarray, yaw_dict: dict):
        """
        Calculates optimal CAV yaw based on its position, using previously collected trajectory data.

        :param vehicle_id: vehicle identifier
        :param pos: 2D-position on simulation map
        :param yaw_dict: dictionary containing information about rotation at each stage of the turn
        :return: yaw (rotation angle)
        """
        nearest_node = "rsu"
        if not self.trajs[vehicle_id] or self.trajs[vehicle_id][-1][-1] == "null":
            logging.warning(f"Intention isn't defined for car {vehicle_id}")
            return 0.0
        else:
            intention = self.trajs[vehicle_id][-1][-1]

        diff = self.control_center_coords - pos
        if abs(diff[0]) > abs(diff[1]):
            if diff[0] < 0:
                start = "right"
            else:
                start = "left"
        else:
            if diff[1] < 0:
                start = "up"
            else:
                start = "down"

        end = self.get_end(start, intention)
        v = f"{start}_{end}"
        if vehicle_id not in self.yaw_id:
            self.yaw_id[vehicle_id] = {nearest_node: v}
        else:
            if nearest_node not in self.yaw_id[vehicle_id]:
                # With new nearest node intantion may changes, so we reset trajectory to default and get intention for new node
                intention = self.get_intention(vehicle_id)
                end = self.get_end(start, intention)
                v = f"{start}_{end}"
                self.yaw_id[vehicle_id] = {nearest_node: v}

                # Update intention in trajs
                previous_traj = self.trajs[vehicle_id][-1]
                self.trajs[vehicle_id] = [(previous_traj[0], previous_traj[1], previous_traj[2], previous_traj[3], previous_traj[4], intention)]
        route = self.yaw_id[vehicle_id][nearest_node]

        if route not in yaw_dict:
            logging.warning(f"Route '{route}' not found for vehicle {vehicle_id}. Using default yaw.")
            return 0.0
        yaws = yaw_dict[route]
        dists = distance.cdist(pos.reshape(1, 2), yaws[:, :-1])
        return yaws[np.argmin(dists), -1]

    @staticmethod
    def get_intention_vector(intention: str = "straight") -> np.ndarray:
        """
        Return a 3-bit one-hot format intention vector.
        """
        intention_feature = np.zeros(3)
        if intention == "left":
            intention_feature[0] = 1
        elif intention == "straight":
            intention_feature[1] = 1
        elif intention == "right":
            intention_feature[2] = 1
        elif intention == "null":
            pass  # return zero array
        else:
            raise NotImplementedError
        return intention_feature
