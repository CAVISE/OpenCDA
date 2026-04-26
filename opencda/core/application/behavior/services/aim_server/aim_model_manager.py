from collections.abc import Sequence
from pathlib import Path
from typing import Any

import logging
import numpy as np
from scipy.spatial import distance

from AIM import AIMModel

from .messages import AIMServerRequest, AIMServerResponse
from .types import AIMServerState, CavData
from opencda.core.application.behavior.types import Transform, Location, Rotation
from . import utils

from opencda.core.application.behavior.transport_message import TransportMessage

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_server.aim_model_manager")


class AIMModelManager:
    def __init__(
        self,
        model: AIMModel,
        control_center: Transform,
        service_name: str,
        owner_id: str,
    ):
        """
        Initialize the standalone AIM model manager.

        Parameters
        ----------
        model : AIMModel
            Loaded AIM model used for trajectory prediction.
        control_center : Transform
            Intersection control point used to normalize vehicle positions.
        service_name : str
            Service identifier used in AIM result messages.
        owner_id : str
            Identifier for the owner of the AIM model.
        """
        self.CONTROL_RADIUS = 50
        self.THRESHOLD = 10
        self.FORCE_VALUE = 20

        self.cav_data: dict[str, CavData] = {}

        self.trajs: dict[str, list[tuple[float, float, float, float, float, str]]] = {}

        control_center_location = utils.get_sumo_transform(control_center, Location(0, 0, 0)).location

        self.control_center_coords: np.ndarray = np.array([control_center_location.x, control_center_location.y])

        self.model = model

        self._service_name = service_name
        self._owner_id = owner_id
        self._last_state_snapshot: AIMServerState

        self.__yaw_dict_path = Path(__file__).parent / "assets" / "yaw_dict_10m.pkl"
        self.yaw_dict: dict[str, Any] = utils.load_yaw(self.__yaw_dict_path)
        self.yaw_id: dict[str, dict[str, str]] = {}

    def _get_distance_to_center(self, curr_pos: np.ndarray) -> float:
        return float(np.linalg.norm(curr_pos - self.control_center_coords))

    def _get_distance_to_center_by_vid(self, vehicle_id: str) -> float:
        curr_pos = self._get_cav_sumo_pos(vehicle_id)
        return self._get_distance_to_center(curr_pos)

    def _preprocess_cav_data(self, transport_message: TransportMessage[AIMServerRequest]) -> None:
        """
        Normalize and cache incoming CAV data for the current inference step.
        """
        message = transport_message.payload
        cav_pos = utils.get_sumo_transform(message.position, Location(0, 0, 0)).location
        curr_pos = np.array([cav_pos.x, cav_pos.y])
        distance_to_center = self._get_distance_to_center(curr_pos)

        if distance_to_center != -1 and distance_to_center < self.CONTROL_RADIUS:
            self.cav_data[message.vehicle_id] = CavData(
                intention=self.get_opencda_intention(message.waypoints, self.control_center_coords),
                pos=message.position.location,
                sumo_pos=curr_pos,
                speed=message.speed,
                yaw=message.yaw,
                waypoints=message.waypoints,
                src_owner_id=transport_message.src_owner_id,
                src_service_type=transport_message.src_service_type,
                dst_owner_id=transport_message.dst_owner_id,
                dst_service_type=transport_message.dst_service_type,
            )

    def _get_cav_pos(self, vehicle_id: str) -> Location:
        return self.cav_data[vehicle_id].pos

    def _get_cav_sumo_pos(self, vehicle_id: str) -> np.ndarray:
        return self.cav_data[vehicle_id].sumo_pos

    def _get_cav_intention(self, vehicle_id: str) -> str:
        return self.cav_data[vehicle_id].intention

    def get_state_snapshot(self) -> AIMServerState:
        """Return an immutable snapshot of the current AIM runtime state."""
        return self._last_state_snapshot

    def _finalize_tick_state(self) -> None:
        self._last_state_snapshot = AIMServerState(
            service_name=self._service_name,
            owner_id=self._owner_id,
            is_attached=True,
            tracked_vehicle_ids=tuple(sorted(self.cav_data)),
            trajectory_vehicle_ids=tuple(sorted(self.trajs)),
            tracked_vehicle_count=len(self.cav_data),
            trajectory_vehicle_count=len(self.trajs),
        )
        self.cav_data.clear()

    def process(self, messages: Sequence[TransportMessage[AIMServerRequest]]) -> Sequence[TransportMessage[AIMServerResponse]]:
        """
        Run AIM inference for the request batch and return predicted targets.
        """
        for message in messages:
            self._preprocess_cav_data(message)
        result_messages: list[TransportMessage[AIMServerResponse]] = []

        self.update_trajs()

        features, target_agent_ids = self.encoding_scenario_features()
        num_agents = features.shape[0]

        if num_agents == 0:
            self._finalize_tick_state()
            return result_messages

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
                rotation = utils.rotation_matrix_back(yaw)
                global_delta = (rotation @ local_delta).squeeze()
                global_delta[1] *= -1

                pos = self._get_cav_pos(vehicle_id)

                global_delta = np.where(np.abs(global_delta) <= self.THRESHOLD, np.sign(global_delta) * self.FORCE_VALUE, global_delta)

                next_location = Location(
                    x=pos.x + global_delta[0],
                    y=pos.y - global_delta[1],
                    z=pos.z,
                )

                payload = AIMServerResponse(
                    next_position=next_location,
                )
                result_messages.append(
                    TransportMessage(
                        src_owner_id=self._owner_id,
                        src_service_type=self._service_name,
                        dst_owner_id=self.cav_data[vehicle_id].src_owner_id,
                        dst_service_type=self.cav_data[vehicle_id].src_service_type,
                        payload=payload,
                    )
                )

        self._finalize_tick_state()
        return result_messages

    def update_trajs(self) -> None:
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
        for vehicle_id in self.cav_data:
            position = self._get_cav_sumo_pos(vehicle_id)

            distance_to_center = self._get_distance_to_center(position)

            if distance_to_center < self.CONTROL_RADIUS:
                # Initialize trajectory if this is a new vehicle
                if vehicle_id not in self.trajs:
                    self.trajs[vehicle_id] = []

                # Get vehicle state
                speed = self.cav_data[vehicle_id].speed
                yaw = self.cav_data[vehicle_id].yaw
                yaw_rad = np.deg2rad(self.get_yaw(vehicle_id, position, self.yaw_dict))

                # Normalize position relative to control node
                node_x, node_y = self.control_center_coords
                rel_x = position[0] - node_x
                rel_y = position[1] - node_y

                if not self.trajs[vehicle_id] or self.trajs[vehicle_id][-1][-1] == "null":
                    intention = self._get_cav_intention(vehicle_id)
                else:
                    intention = self.trajs[vehicle_id][-1][-1]

                self.trajs[vehicle_id] = [(rel_x, rel_y, speed, yaw_rad, yaw, intention)]

        for vehicle_id in list(self.trajs):
            if vehicle_id not in self.cav_data:
                del self.trajs[vehicle_id]

    def get_opencda_intention(self, waypoints: Sequence[Any], mid: np.ndarray) -> str:
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
        location = Location(mid[0], mid[1], 0)
        rotation = Rotation(0, 0, 0)

        in_sumo_transform = Transform(location, rotation)
        mid_carla = utils.get_carla_transform(in_sumo_transform, Location(0, 0, 0))
        if utils.get_distance(mid_carla, waypoints[0]) > self.CONTROL_RADIUS:
            logger.debug("Car not int radius")
            return "null"
        while utils.get_distance(mid_carla, waypoints[waypoint_index]) > self.CONTROL_RADIUS:
            waypoint_index += 1
            if waypoint_index >= len(waypoints):
                logger.debug("No waypoints in radius")
                return "null"

        first_waypoint = waypoints[waypoint_index]
        first_waypoint_index = waypoint_index
        while (utils.get_distance(mid_carla, waypoints[waypoint_index]) <= self.CONTROL_RADIUS) and waypoint_index < len(waypoints) - 1:
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
        return utils.get_intention_by_rotation(rotation)

    def encoding_scenario_features(self) -> tuple[np.ndarray, list[str]]:
        """
        Encodes data on CAV movement and intentions for processing by ML models

        :return: x: list((motion features, intention vector)), target: agent ids list(vehicle id)
        """
        features: list[np.ndarray] = []
        target_agent_ids: list[str] = []

        for vehicle_id, trajectory in self.trajs.items():
            last_position = trajectory[-1]
            position = np.array(last_position[:2])
            distance_to_origin = np.linalg.norm(position)

            if distance_to_origin < self.CONTROL_RADIUS:
                motion_features = np.array(last_position[:-2])
                intention_vector = utils.get_intention_vector(last_position[-1])
                feature_vector = np.concatenate((motion_features, intention_vector)).reshape(1, -1)

                features.append(feature_vector)
                target_agent_ids.append(vehicle_id)

        feature_matrix = np.vstack(features) if features else np.empty((0, 7))
        return feature_matrix, target_agent_ids

    def get_yaw(self, vehicle_id: str, pos: np.ndarray, yaw_dict: dict[str, Any]) -> float:
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

        end = utils.get_end(start, intention)
        v = f"{start}_{end}"
        if vehicle_id not in self.yaw_id:
            self.yaw_id[vehicle_id] = {nearest_node: v}
        else:
            if nearest_node not in self.yaw_id[vehicle_id]:
                # With new nearest node intantion may changes, so we reset trajectory to default and get intention for new node
                intention = self._get_cav_intention(vehicle_id)
                end = utils.get_end(start, intention)
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
        return float(yaws[np.argmin(dists), -1])
