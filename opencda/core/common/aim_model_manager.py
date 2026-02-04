from typing import Any, Dict, List, Optional, Set, Tuple
import numpy.typing as npt
import carla
import traci
import torch
import logging
import numpy as np
import pickle as pkl
from scipy.spatial import distance

from CoDriving.scripts.constants import CONTROL_RADIUS, THRESHOLD, FORCE_VALUE
from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper
from AIM import AIMModel

logger = logging.getLogger("cavise.codriving_model_manager")


class AIMModelManager:
    def __init__(self, model: AIMModel, nodes: List[Any], excluded_nodes: Optional[List[Any]] = None):
        """
        :param model_name: model name contained in the filename
        :param pretrained: filepath to saved model state
        :param nodes: intersections present in the simulation
        :param excluded_nodes: nodes that do not use AIM

        :return: None
        """
        self.mtp_controlled_vehicles: Set = set()

        self.cav_ids: Set = set()
        self.carla_vmanagers: Set = set()

        self.trajs: Dict = dict()

        self.nodes = nodes
        self.node_coords = np.array([node.getCoord() for node in nodes])
        self.excluded_nodes = excluded_nodes  # Intersections where the MTP module is disabled

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.yaw_dict = self._load_yaw()
        self.yaw_id: Dict[str, Dict[Any, str]] = {}

    def _load_yaw(self) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Load yaw dictionary from predefined file.

        Returns
        -------
        Dict[str, NDArray[np.float64]]
            Dictionary mapping routes to yaw arrays.
        """
        with open("opencda/assets/yaw_dict_10m.pkl", "rb") as f:
            return pkl.load(f)

    def _get_nearest_node(self, pos: npt.NDArray[np.float64]) -> Any:
        """
        Find the nearest node to given position.

        Parameters
        ----------
        pos : NDArray[np.float64]
            2D position on simulation map.

        Returns
        -------
        Any
            Nearest node object.
        """
        distances = np.linalg.norm(self.node_coords - pos, axis=1)
        return self.nodes[np.argmin(distances)]

    def _get_vmanager_by_vid(self, vid: str) -> Optional[Any]:
        """
        Get vehicle manager by vehicle ID.

        Parameters
        ----------
        vid : str
            Vehicle manager ID.

        Returns
        -------
        Optional[Any]
            Vehicle manager if found, None otherwise.
        """
        for vmanager in self.carla_vmanagers:
            if vmanager.vid == vid:
                return vmanager
        return None

    def make_trajs(self, carla_vmanagers):
        """
        Create trajectories based on model predictions.

        Parameters
        ----------
        carla_vmanagers : Set[Any]
            Set of CARLA vehicle managers.
        """
        # List of cars from CARLA
        self.carla_vmanagers = carla_vmanagers
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

            if distance_to_center < CONTROL_RADIUS:
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

                global_delta = np.where(np.abs(global_delta) <= THRESHOLD, np.sign(global_delta) * FORCE_VALUE, global_delta)

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
            # Get current vehicle position and find nearest node
            position = np.array(traci.vehicle.getPosition(vehicle_id))
            nearest_node = self._get_nearest_node(position)

            # Skip excluded regions
            if self.excluded_nodes and nearest_node in self.excluded_nodes:
                continue

            control_center = np.array(nearest_node.getCoord())
            distance_to_center = np.linalg.norm(position - control_center)

            if distance_to_center < CONTROL_RADIUS:
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

                # Determine intention
                if not self.trajs[vehicle_id] or self.trajs[vehicle_id][-1][-1] == "null":
                    intention = self.get_intention(vehicle_id)
                else:
                    intention = self.trajs[vehicle_id][-1][-1]

                # Append current state to trajectory
                self.trajs[vehicle_id] = [(rel_x, rel_y, speed, yaw_rad, yaw_deg_sumo, intention)]

        # Remove trajectories of vehicles that have left the scene
        for vehicle_id in list(self.trajs):
            if vehicle_id not in self.cav_ids:
                del self.trajs[vehicle_id]

    def get_intention_by_rotation(self, rotation: float) -> str:
        """
        Determine vehicle intention from rotation angle.

        Parameters
        ----------
        rotation : float
            Rotation angle in degrees (0-360).

        Returns
        -------
        str
            Intention: 'left', 'straight', 'right', or 'null'.
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

    def get_distance(self, waypoint1: Any, waypoint2: Tuple[Any]) -> float:
        """
        Calculate Euclidean distance between waypoints.

        Parameters
        ----------
        waypoint1 : Any
            First waypoint with location attribute.
        waypoint2 : Tuple[Any]
            Second waypoint as tuple.

        Returns
        -------
        float
            Euclidean distance.
        """
        rel_x = waypoint1.location.x - waypoint2[0].transform.location.x
        rel_y = waypoint1.location.y - waypoint2[0].transform.location.y
        position = np.array([rel_x, rel_y])
        return np.linalg.norm(position)

    def get_opencda_intention(self, waypoints: List[Tuple[Any]], mid: Tuple[float, float], radius: float = CONTROL_RADIUS) -> str:
        """
        Get intention from OpenCDA waypoints.

        Parameters
        ----------
        waypoints : List[Tuple[Any]]
            List of waypoint coordinates.
        mid : Tuple[float, float]
            Middle of turning path coordinates.
        radius : float, optional
            Search radius, by default CONTROL_RADIUS.

        Returns
        -------
        str
            Intention: 'left', 'straight', 'right', or 'null'.
        """
        # Too few waypoints
        if len(waypoints) < 2:
            return "null"

        waypoint_index = 0
        location = carla.Location(mid[0], mid[1], 0)
        rotation = carla.Rotation(0)

        in_sumo_transform = carla.Transform(location, rotation)
        mid = BridgeHelper.get_carla_transform(in_sumo_transform, carla.Vector3D(0, 0, 0))
        if self.get_distance(mid, waypoints[0]) > radius:
            logger.debug("Car not int radius")
            return "null"
        while self.get_distance(mid, waypoints[waypoint_index]) > radius:
            waypoint_index += 1
            if waypoint_index >= len(waypoints):
                logger.debug("No waypoints in radius")
                return "null"
                # raise IndexError("No waypoints in radius")

        first_waypoint = waypoints[waypoint_index]
        first_waypoint_index = waypoint_index
        while (self.get_distance(mid, waypoints[waypoint_index]) <= radius) and waypoint_index < len(waypoints) - 1:
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
        return self.get_intention_by_rotation(rotation)

    def get_intention(self, vehicle_id):
        cav = self._get_vmanager_by_vid(vehicle_id)
        cav.set_destination(cav.vehicle.get_location(), cav.agent.end_waypoint.transform.location, clean=True, end_reset=True)
        waypoints = cav.agent.get_local_planner().get_waypoint_buffer()
        curr_pos = np.array(traci.vehicle.getPosition(vehicle_id))
        nearest_node = self._get_nearest_node(curr_pos)
        control_center = nearest_node.getCoord()
        return self.get_opencda_intention(waypoints, control_center, CONTROL_RADIUS)

    def encoding_scenario_features(self) -> Tuple[npt.NDArray[np.float64], List[str]]:
        """
        Encode scenario features for ML model.

        Returns
        -------
        features : NDArray[np.float64]
            Array of shape (n_agents, 7) with motion and intention features.
        target_agent_ids : List[str]
            List of vehicle IDs.
        """
        features = []
        target_agent_ids = []

        for vehicle_id, trajectory in self.trajs.items():
            last_position = trajectory[-1]
            position = np.array(last_position[:2])
            distance_to_origin = np.linalg.norm(position)

            if distance_to_origin < CONTROL_RADIUS:
                motion_features = np.array(last_position[:-2])
                intention_vector = self.get_intention_vector(last_position[-1])
                feature_vector = np.concatenate((motion_features, intention_vector)).reshape(1, -1)

                features.append(feature_vector)
                target_agent_ids.append(vehicle_id)

        features = np.vstack(features) if features else np.empty((0, 7))
        return features, target_agent_ids

    @staticmethod
    def transform_sumo2carla(states: npt.NDArray[np.float64]) -> None:
        """
        Transform coordinates from SUMO to CARLA in-place.

        Parameters
        ----------
        states : NDArray[np.float64]
            State array with x, y, yaw coordinates.

        Note:
        ----------
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
        Create rotation matrix for given yaw.

        Parameters
        ----------
        yaw : float
            Yaw angle in radians.

        Returns
        -------
        NDArray[np.float64]
            2x2 rotation matrix.):

        References
        -------
        Rotate back.
        https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
        """
        rotation = np.array([[np.cos(-np.pi / 2 + yaw), -np.sin(-np.pi / 2 + yaw)], [np.sin(-np.pi / 2 + yaw), np.cos(-np.pi / 2 + yaw)]])
        return rotation

    def get_end(self, start: str, intention: str) -> Any:
        """
        Determine exit direction from intersection.

        Parameters
        ----------
        start : str
            Entry direction: 'up', 'down', 'left', 'right'.
        intention : str
            Turn intention: 'left', 'straight', 'right'.

        Returns
        -------
        Optional[str]
            Exit direction or None.
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

    def get_yaw(self, vehicle_id: str, pos: npt.NDArray[np.float64], yaw_dict: Dict[str, npt.NDArray[np.float64]]) -> float:
        """
        Calculate optimal yaw for vehicle position.

        Parameters
        ----------
        vehicle_id : str
            Vehicle ID.
        pos : NDArray[np.float64]
            2D position array.
        yaw_dict : Dict[str, NDArray[np.float64]]
            Dictionary with yaw data for routes.

        Returns
        -------
        float
            Yaw angle in degrees.
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
