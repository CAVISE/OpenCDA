from collections.abc import Sequence
from pathlib import Path
from typing import Any

import logging
import numpy as np

from AIM import AIMModelWrapper

from .messages import AIMServerRequest, AIMServerResponse
from .types import AIMServerState, CavData
from opencda.core.application.behavior.types import Location
from . import utils

from opencda.core.application.behavior.transport_message import TransportMessage

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_server.aim_model_manager")


class AIMModelManager:
    def __init__(
        self,
        model: AIMModelWrapper,
        control_center: Location,
        service_type: str,
        owner_id: str,
        control_radius: int = 15,
    ):
        """
        Initialize the standalone AIM model manager.

        Parameters
        ----------
        model : AIMModel
            Loaded AIM model used for trajectory prediction.
        control_center : Transform
            Intersection control point used to normalize vehicle positions.
        service_type : str
            Service identifier used in AIM result messages.
        owner_id : str
            Identifier for the owner of the AIM model.
        """
        self.CONTROL_RADIUS = control_radius

        self.cav_data: dict[str, CavData] = {}
        self.cav_state: dict[str, dict] = {}

        self.trajs: dict[str, list[tuple[float, float, float, float, float, str]]] = {}

        self.control_center_carla_location: Location = control_center
        control_center_location: Location = utils.get_sumo_location(control_center)
        self.control_center_coords: np.ndarray = np.array([control_center_location.x, control_center_location.y])

        self.model = model

        self._service_name = service_type
        self._owner_id = owner_id
        self._last_state_snapshot: AIMServerState | None = None

        self.__yaw_dict_path = Path(__file__).parent / "assets" / "yaw_dict_10m.pkl"
        self.yaw_dict: dict[str, Any] = utils.load_yaw(self.__yaw_dict_path)

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
        vehicle_id = message.vehicle_id

        if distance_to_center != -1 and distance_to_center < self.CONTROL_RADIUS:
            self.cav_data[vehicle_id] = CavData(
                intention=self.get_intention(vehicle_id, message.waypoints, self.control_center_carla_location),
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
            if vehicle_id not in self.cav_state:
                self.cav_state[vehicle_id] = {}
            self.cav_state[vehicle_id]["ttl"] = 3
        else:
            if vehicle_id in self.cav_state:
                del self.cav_state[vehicle_id]

    def _get_cav_pos(self, vehicle_id: str) -> Location:
        return self.cav_data[vehicle_id].pos

    def _get_cav_sumo_pos(self, vehicle_id: str) -> np.ndarray:
        return self.cav_data[vehicle_id].sumo_pos

    def _get_cav_intention(self, vehicle_id: str) -> str:
        return self.cav_data[vehicle_id].intention

    def get_state_snapshot(self) -> AIMServerState:
        """Return an immutable snapshot of the current AIM runtime state."""
        if self._last_state_snapshot is None:
            return AIMServerState(
                service_type=self._service_name,
                owner_id=self._owner_id,
                tracked_vehicle_ids=(),
                trajectory_vehicle_ids=(),
                tracked_vehicle_count=0,
                trajectory_vehicle_count=0,
            )
        return self._last_state_snapshot

    def _finalize_tick_state(self) -> None:
        self._last_state_snapshot = AIMServerState(
            service_type=self._service_name,
            owner_id=self._owner_id,
            tracked_vehicle_ids=tuple(sorted(self.cav_data)),
            trajectory_vehicle_ids=tuple(sorted(self.trajs)),
            tracked_vehicle_count=len(self.cav_data),
            trajectory_vehicle_count=len(self.trajs),
        )
        self.cav_data.clear()

    def process(self, messages: Sequence[TransportMessage[AIMServerRequest]]) -> tuple[TransportMessage[AIMServerResponse], ...]:
        """
        Run AIM inference for the request batch and return predicted targets.
        """
        for vehicle_id in self.cav_state:
            self.cav_state[vehicle_id]["ttl"] -= 1
            if self.cav_state[vehicle_id]["ttl"] == 0:
                del self.cav_state[vehicle_id]

        for message in messages:
            self._preprocess_cav_data(message)
        result_messages: list[TransportMessage[AIMServerResponse]] = []

        self.update_trajs()

        features, target_agent_ids = self.encoding_scenario_features()
        num_agents = features.shape[0]

        if num_agents == 0:
            self._finalize_tick_state()
            return ()

        predictions = self.model.predict(features.copy(), target_agent_ids)

        for idx in range(num_agents):
            vehicle_id = target_agent_ids[idx]

            distance_to_center = self._get_distance_to_center_by_vid(vehicle_id)

            if distance_to_center < self.CONTROL_RADIUS:
                pred_delta = predictions[idx].reshape(30, 2).detach().cpu().numpy()
                yaw_rad = features[idx][3] - np.deg2rad(90)  # convert to carla yaw

                trajectory = [self.predition_to_location(vehicle_id, pred_delta[i], yaw_rad) for i in range(30)]
                payload = AIMServerResponse(
                    trajectory=trajectory,
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
        return tuple(result_messages)

    def predition_to_location(self, vehicle_id: str, local_delta: np.ndarray, yaw: float) -> Location:
        rotation = utils.rotation_matrix_back(yaw)
        global_delta = (rotation @ local_delta).squeeze()
        position = self._get_cav_pos(vehicle_id)

        return Location(
            x=position.x + global_delta[0],
            y=position.y + global_delta[1],
            z=position.z,
        )

    def update_trajs(self) -> None:
        """
        Updates the self.trajs dictionary, which stores the trajectory history of each vehicle.
        """
        for vehicle_id in self.cav_data:
            position = self._get_cav_sumo_pos(vehicle_id)

            distance_to_center = self._get_distance_to_center(position)

            if distance_to_center < self.CONTROL_RADIUS:
                # Initialize trajectory if this is a new vehicle
                if vehicle_id not in self.trajs:
                    self.trajs[vehicle_id] = []

                if self.trajs[vehicle_id] == [] or self.trajs[vehicle_id][-1][-1] == "null":
                    intention = self._get_cav_intention(vehicle_id)
                else:
                    intention = self.trajs[vehicle_id][-1][-1]

                # Get vehicle state
                speed = self.cav_data[vehicle_id].speed
                yaw = self.cav_data[vehicle_id].yaw
                yaw_rad = np.deg2rad(self.get_yaw(vehicle_id, position, self.yaw_dict, intention))

                # Normalize position relative to control node
                node_x, node_y = self.control_center_coords
                rel_x = position[0] - node_x
                rel_y = position[1] - node_y

                self.trajs[vehicle_id] = [(rel_x, rel_y, speed, yaw_rad, yaw, intention)]

        for vehicle_id in list(self.trajs):
            if vehicle_id not in self.cav_data:
                del self.trajs[vehicle_id]

    def get_intention(self, vehicle_id: str, waypoints: Sequence[Any], mid: Location) -> str:
        if vehicle_id not in self.trajs or self.trajs[vehicle_id] == [] or self.trajs[vehicle_id][-1][-1] == "null":
            return self.get_opencda_intention(waypoints, mid)
        else:
            return self.trajs[vehicle_id][-1][-1]

    def get_opencda_intention(self, waypoints: Sequence[Any], mid: Location) -> str:
        """
        Gets intention by averaged rotation to pass 3 next waypoints.

        :param waypoints: list of waypoint 2D-coordinates
        :param mid: middle of the turning path 2D-coordinates
        :param radius: search radius for waypoints nearby
        :return: intention
        """
        # Too few waypoints
        if len(waypoints) < 2:
            logger.debug("Too few waypoints")
            return "null"

        waypoint_index = 0

        if utils.get_distance(mid, waypoints[0][0].transform.location) > self.CONTROL_RADIUS:
            logger.debug("Car not in radius")
            return "null"
        while utils.get_distance(mid, waypoints[waypoint_index][0].transform.location) > self.CONTROL_RADIUS:
            waypoint_index += 1
            if waypoint_index >= len(waypoints):
                logger.debug("No waypoints in radius")
                return "null"

        first_waypoint = waypoints[waypoint_index]
        first_waypoint_index = waypoint_index
        while utils.get_distance(mid, waypoints[waypoint_index][0].transform.location) <= self.CONTROL_RADIUS and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1

        # average the values of several points to reduce noise
        mean_yaw = 0
        waypoints_in_radius = waypoint_index - first_waypoint_index
        if waypoints_in_radius < 3 and waypoint_index > 3:
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

    def get_yaw(self, vehicle_id: str, position: np.ndarray, yaw_dict: dict[str, Any], intention: str) -> float:
        """
        Calculates optimal CAV yaw based on its position, using previously collected trajectory data.

        :param vehicle_id: vehicle identifier
        :param position: 2D-position on simulation map
        :param yaw_dict: dictionary containing information about rotation at each stage of the turn
        :return: yaw (rotation angle)
        """
        if intention == "null":
            logger.warning(f"Intention isn't defined for car {vehicle_id}")
            return 0.0

        if "route" not in self.cav_state[vehicle_id] or self.cav_state[vehicle_id]["route"] == "":
            diff = self.control_center_coords - position
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
            route = f"{start}_{end}"
            self.cav_state[vehicle_id]["route"] = route
        else:
            route = self.cav_state[vehicle_id]["route"]

        if route not in yaw_dict:
            logger.warning(f"Route '{route}' not found for vehicle {vehicle_id}. Using default yaw.")
            return 0.0
        yaws = yaw_dict[route]
        dists = np.linalg.norm(yaws[:, :-1] - position, axis=1)
        return float(yaws[np.argmin(dists), -1])
