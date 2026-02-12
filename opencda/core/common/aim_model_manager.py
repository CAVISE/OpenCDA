import carla
import traci
import torch
import logging
import numpy as np
import numpy.typing as npt
import pickle as pkl
from scipy.spatial import distance
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper
from AIM import AIMModel


logger = logging.getLogger("cavise.codriving_model_manager")


class AIMModelManager:
    def __init__(self, model: AIMModel, nodes: Sequence[Any], excluded_nodes: Optional[Set[Any]] = None) -> None:
        """
        :param model_name: model name contained in the filename
        :param pretrained: filepath to saved model state
        :param nodes: intersections present in the simulation
        :param excluded_nodes: nodes that do not use AIM

        :return: None
        """
        self.CONTROL_RADIUS = 15
        self.THRESHOLD = 10
        self.FORCE_VALUE = 20

        self.mtp_controlled_vehicles: Set[str] = set()

        self.cav_ids: List[str] = []
        self.carla_vmanagers: List[Any] = []

        self.trajs: Dict[str, List[Tuple[float, float, float, float, float, str]]] = {}

        self.nodes = nodes
        self.node_coords: npt.NDArray[np.float64] = np.array([node.getCoord() for node in nodes], dtype=np.float64)
        self.excluded_nodes = excluded_nodes  # Intersections where the MTP module is disabled

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.yaw_dict = self._load_yaw()
        self.yaw_id: Dict[str, Dict[Any, str]] = {}

    def _load_yaw(self) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Loads yaw dictionary from a predefined address.

        :return: yaw_dict
        """
        with open("opencda/assets/yaw_dict_10m.pkl", "rb") as f:
            return cast(Dict[str, npt.NDArray[np.float64]], pkl.load(f))

    def _get_nearest_node(self, pos: npt.NDArray[np.float64]) -> Any:
        """
        Selects the nearest node from chosen position

        :param pos: 2D-position on simulation map
        :return: position of the closest node
        """
        distances = np.linalg.norm(self.node_coords - pos, axis=1)
        return self.nodes[int(np.argmin(distances))]

    def _get_vmanager_by_vid(self, vid: str) -> Optional[Any]:
        """
        Returns vmanager with selected vid if exists.

        :param vid: virtual manager id (string)
        :return: virtual manager
        """
        for vmanager in self.carla_vmanagers:
            if vmanager.vid == vid:
                return vmanager
        return None

    def make_trajs(self, carla_vmanagers: Sequence[Any]) -> None:
        """
        Creates new trajectories based on model predictions, assigns CAVs new destinations.

        :param carla_vmanagers: carla virtual managers
        :return: None
        """
        # List of cars from CARLA
        self.carla_vmanagers = list(carla_vmanagers)
        self.cav_ids = [vmanager.vid for vmanager in self.carla_vmanagers]

        self.update_trajs()

        features, target_agent_ids = self.encoding_scenario_features()
        num_agents = features.shape[0]

        if num_agents == 0:
            return

        # Transform coordinates
        self.transform_sumo2carla(features)
        features_tensor = torch.tensor(features).float().to(self.device)

        predictions = self.model.predict(features, target_agent_ids)

        for idx in range(num_agents):
            vehicle_id = target_agent_ids[idx]

            pos_x, pos_y = traci.vehicle.getPosition(vehicle_id)
            curr_pos = np.array([pos_x, pos_y])

            nearest_node = self._get_nearest_node(curr_pos)
            if self.excluded_nodes and nearest_node in self.excluded_nodes:
                continue

            control_center = np.array(nearest_node.getCoord())
            distance_to_center = np.linalg.norm(curr_pos - control_center)

            if distance_to_center < self.CONTROL_RADIUS:
                self.mtp_controlled_vehicles.add(vehicle_id)

                pred_delta = predictions[idx].reshape(30, 2).detach().cpu().numpy()
                local_delta = pred_delta[0].reshape(2, 1)
                last_delta = pred_delta[-1].reshape(2, 1)

                if last_delta[1, 0] <= 1:
                    local_delta[1, 0] = 1e-8
                    local_delta[0, 0] = 1e-10
                else:
                    local_delta[1, 0] = max(1e-8, local_delta[1, 0])

                yaw = features_tensor[idx, 3].detach().cpu().item()
                rotation = self.rotation_matrix_back(yaw)
                global_delta = (rotation @ local_delta).squeeze()
                global_delta[1] *= -1

                cav = self._get_vmanager_by_vid(vehicle_id)
                if cav is None:
                    continue

                pos = cav.vehicle.get_location()

                global_delta = np.where(np.abs(global_delta) <= self.THRESHOLD, np.sign(global_delta) * self.FORCE_VALUE, global_delta)

                next_loc = carla.Location(
                    x=pos.x + global_delta[0],
                    y=pos.y - global_delta[1],
                    z=pos.z,
                )

                cav.set_destination(pos, next_loc, clean=True, end_reset=False)
                cav.update_info_v2x()

                if len(cav.agent.get_local_planner().get_waypoint_buffer()) == 0:
                    logger.warning(f"{vehicle_id}: waypoint buffer is empty after set_destination!")
            elif vehicle_id in self.mtp_controlled_vehicles:
                cav = self._get_vmanager_by_vid(vehicle_id)
                if cav is None:
                    continue
                cav.set_destination(cav.vehicle.get_location(), cav.agent.end_waypoint.transform.location, clean=True, end_reset=True)

                self.mtp_controlled_vehicles.remove(vehicle_id)

    def update_trajs(self) -> None:
        """
        Updates the self.trajs dictionary, which stores the trajectory history of each vehicle.
        Format:
        {
            'vehicle_id': [
                (rel_x, rel_y, speed, yaw_rad, yaw_deg_sumo, intention),
                ...
            ],
            ...
        }
        """
        for vehicle_id in self.cav_ids:
            position = np.array(traci.vehicle.getPosition(vehicle_id))
            nearest_node = self._get_nearest_node(position)

            if self.excluded_nodes and nearest_node in self.excluded_nodes:
                continue

            control_center = np.array(nearest_node.getCoord())
            distance_to_center = np.linalg.norm(position - control_center)

            if distance_to_center < self.CONTROL_RADIUS:
                # Initialize trajectory if this is a new vehicle
                if vehicle_id not in self.trajs:
                    self.trajs[vehicle_id] = []

                # Get vehicle state
                speed = traci.vehicle.getSpeed(vehicle_id)
                yaw_deg_sumo = traci.vehicle.getAngle(vehicle_id)
                yaw_rad = np.deg2rad(self.get_yaw(vehicle_id, position, self.yaw_dict))

                # Normalize position relative to control node
                node_x, node_y = nearest_node.getCoord()
                rel_x = position[0] - node_x
                rel_y = position[1] - node_y

                if not self.trajs[vehicle_id] or self.trajs[vehicle_id][-1][-1] == "null":
                    intention = self.get_intention(vehicle_id)
                else:
                    intention = self.trajs[vehicle_id][-1][-1]

                self.trajs[vehicle_id] = [(rel_x, rel_y, speed, yaw_rad, yaw_deg_sumo, intention)]

        for vehicle_id in list(self.trajs):
            if vehicle_id not in self.cav_ids:
                del self.trajs[vehicle_id]

    def get_intention_by_rotation(self, rotation: float) -> str:
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

    def get_distance(self, waypoint1: carla.Transform, waypoint2: Sequence[Any]) -> float:
        """
        Calculates Euclidean distance between two waypoints

        :param waypoint1: waypoint 2D-coordinates
        :param waypoint2: waypoint 2D-coordinates
        :return: distance
        """
        rel_x = waypoint1.location.x - waypoint2[0].transform.location.x
        rel_y = waypoint1.location.y - waypoint2[0].transform.location.y
        position = np.array([rel_x, rel_y])
        return float(np.linalg.norm(position))

    def get_opencda_intention(self, waypoints: Sequence[Any], mid: Sequence[float]) -> str:
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
        if self.get_distance(mid, waypoints[0]) > self.CONTROL_RADIUS:
            logger.debug("Car not int radius")
            return "null"
        while self.get_distance(mid, waypoints[waypoint_index]) > self.CONTROL_RADIUS:
            waypoint_index += 1
            if waypoint_index >= len(waypoints):
                logger.debug("No waypoints in radius")
                return "null"
                # raise IndexError("No waypoints in radius")

        first_waypoint = waypoints[waypoint_index]
        first_waypoint_index = waypoint_index
        while (self.get_distance(mid, waypoints[waypoint_index]) <= self.CONTROL_RADIUS) and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1

        # average the values of several points to reduce noise
        mean_yaw = 0
        if waypoint_index - first_waypoint_index < 3 and waypoint_index > 3:
            logger.warning("Too few waypoints in radius")
        else:
            for i in range(3):
                mean_yaw += waypoints[waypoint_index - i][0].transform.rotation.yaw
        mean_yaw = mean_yaw // 3
        rotation = (mean_yaw - first_waypoint[0].transform.rotation.yaw + 360) % 360
        return self.get_intention_by_rotation(rotation)

    def get_intention(self, vehicle_id: str) -> str:
        cav = self._get_vmanager_by_vid(vehicle_id)
        if cav is None:
            return "null"
        cav.set_destination(cav.vehicle.get_location(), cav.agent.end_waypoint.transform.location, clean=True, end_reset=True)
        waypoints = cav.agent.get_local_planner().get_waypoint_buffer()
        curr_pos = np.array(traci.vehicle.getPosition(vehicle_id))
        nearest_node = self._get_nearest_node(curr_pos)
        control_center = nearest_node.getCoord()
        return self.get_opencda_intention(waypoints, control_center)

    def encoding_scenario_features(self) -> Tuple[npt.NDArray[np.float64], List[str]]:
        """
        Encodes data on CAV movement and intentions for processing by ML models

        :return: x: list((motion features, intention vector)), target: agent ids list(vehicle id)
        """
        feature_list: List[npt.NDArray[np.float64]] = []
        target_agent_ids = []

        for vehicle_id, trajectory in self.trajs.items():
            last_position = trajectory[-1]
            position = np.array(last_position[:2])
            distance_to_origin = np.linalg.norm(position)

            if distance_to_origin < self.CONTROL_RADIUS:
                motion_features = np.array(last_position[:-2])
                intention_vector = self.get_intention_vector(last_position[-1])
                feature_vector = np.concatenate((motion_features, intention_vector)).reshape(1, -1)

                feature_list.append(feature_vector)
                target_agent_ids.append(vehicle_id)

        features = np.vstack(feature_list) if feature_list else np.empty((0, 7))
        return cast(npt.NDArray[np.float64], features), target_agent_ids

    @staticmethod
    def transform_sumo2carla(states: npt.NDArray[np.float64]) -> None:
        """
        In-place transform from sumo to carla: [x_carla, y_carla, yaw_carla] = [x_sumo, -y_sumo, yaw_sumo-90].
        Note:
            - the coordinate system in Carla is more convenient since the angle increases in the direction of rotation from +x to +y, while in sumo this is from +y to +x.
            - the coordinate system in Carla is a left-handed Cartesian coordinate system.
        """
        if states.ndim == 1:
            states[1] = -states[1]
            states[3] -= np.deg2rad(90)
        elif states.ndim == 2:
            states[:, 1] = -states[:, 1]
            states[:, 3] -= np.deg2rad(90)
        else:
            raise NotImplementedError

    @staticmethod
    def rotation_matrix_back(yaw: float) -> npt.NDArray[np.float64]:
        """
        Rotate back.
        https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
        """
        rotation = np.array([[np.cos(-np.pi / 2 + yaw), -np.sin(-np.pi / 2 + yaw)], [np.sin(-np.pi / 2 + yaw), np.cos(-np.pi / 2 + yaw)]])
        return rotation

    def get_end(self, start: str, intention: str) -> str:
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
        return "null"

    def get_yaw(self, vehicle_id: str, pos: npt.NDArray[np.float64], yaw_dict: Dict[str, npt.NDArray[np.float64]]) -> float:
        """
        Calculates optimal CAV yaw based on its position, using previously collected trajectory data.

        :param vehicle_id: vehicle identifier
        :param pos: 2D-position on simulation map
        :param yaw_dict: dictionary containing information about rotation at each stage of the turn
        :return: yaw (rotation angle)
        """
        nearest_node = self._get_nearest_node(pos)
        if not self.trajs[vehicle_id] or self.trajs[vehicle_id][-1][-1] == "null":
            logging.warning(f"Intention isn't defined for car {vehicle_id}")
            return 0.0
        else:
            intention = self.trajs[vehicle_id][-1][-1]

        control_center = nearest_node.getCoord()
        diff = control_center - pos
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
                cav = self._get_vmanager_by_vid(vehicle_id)
                if cav is None:
                    return 0.0
                cav.set_destination(cav.vehicle.get_location(), cav.agent.end_waypoint.transform.location, clean=True, end_reset=True)
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
        return float(yaws[np.argmin(dists), -1])

    @staticmethod
    def get_intention_vector(intention: str = "straight") -> npt.NDArray[np.float64]:
        """
        Return a 3-bit one-hot format intention vector.
        """
        intention_feature = np.zeros(3, dtype=np.float64)
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
