import carla
import traci
import torch
import logging
import numpy as np
import pickle as pkl
from scipy.spatial import distance

from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper
from AIM import AIMModel


logger = logging.getLogger("cavise.opencda.opencda.core.common.aim_model_manager")


class AIMModelManager:
    def __init__(self, model: AIMModel, nodes, excluded_nodes=None, payload_handler=None):
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

        self.mtp_controlled_vehicles = set()

        self.cav_ids = set()
        self.carla_vmanagers = set()

        self.trajs = dict()

        self.nodes = nodes
        self.node_coords = np.array([node.getCoord() for node in nodes])
        self.excluded_nodes = excluded_nodes  # Intersections where the MTP module is disabled

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.payload_handler = payload_handler
        self.module_name = "AIM.AIMModelManager"
        self.latest_output_payloads = {}

        self.yaw_dict = self._load_yaw()
        self.yaw_id = {}

    def _load_yaw(self):
        """
        Loads yaw dictionary from a predefined address.

        :return: yaw_dict
        """
        with open("opencda/assets/yaw_dict_10m.pkl", "rb") as f:
            return pkl.load(f)

    def _get_nearest_node(self, pos):
        """
        Selects the nearest node from chosen position

        :param pos: 2D-position on simulation map
        :return: position of the closest node
        """
        distances = np.linalg.norm(self.node_coords - pos, axis=1)
        return self.nodes[np.argmin(distances)]

    def _get_vmanager_by_vid(self, vid: str):
        """
        Returns vmanager with selected vid if exists.

        :param vid: virtual manager id (string)
        :return: virtual manager
        """
        for vmanager in self.carla_vmanagers:
            if vmanager.vid == vid:
                return vmanager
        return None

    def make_trajs(self, carla_vmanagers):
        """
        Creates new trajectories based on model predictions, assigns CAVs new destinations.

        :param carla_vmanagers: carla virtual managers
        :return: None
        """
        # List of cars from CARLA
        self.carla_vmanagers = carla_vmanagers
        self.cav_ids = [vmanager.vid for vmanager in self.carla_vmanagers]

        self.update_trajs()

        if self.payload_handler is None:
            self._make_trajs_without_messages()
            return

        self._make_trajs_with_messages()

    def extract_data(self, carla_vmanagers):
        """
        Export per-vehicle AIM features into the payload handler for CAPI exchange.
        """
        self.carla_vmanagers = carla_vmanagers
        self.cav_ids = [vmanager.vid for vmanager in self.carla_vmanagers]
        self.update_trajs()

        if self.payload_handler is None:
            return

        feature_map = self._encoding_feature_map()
        current_vehicle_ids = set(self.cav_ids)
        export_ids = (set(feature_map) | set(self.latest_output_payloads)) & current_vehicle_ids

        for vehicle_id in export_ids:
            with self.payload_handler.handle_opencda_payload(vehicle_id, self.module_name) as msg:
                if vehicle_id in feature_map:
                    msg["aim_features"] = feature_map[vehicle_id]
                if vehicle_id in self.latest_output_payloads:
                    msg["aim_output"] = self.latest_output_payloads[vehicle_id]

    def _make_trajs_without_messages(self):
        features, target_agent_ids = self.encoding_scenario_features()
        num_agents = features.shape[0]

        if num_agents == 0:
            return

        self._predict_and_apply(features, target_agent_ids)

    def _make_trajs_with_messages(self):
        for cav in self.carla_vmanagers:
            remote_output = self._get_remote_output(cav.vid)
            if remote_output is not None:
                self._apply_output(cav.vid, remote_output)
                continue

            features, target_agent_ids = self._encoding_features_from_messages(cav.vid)
            if features.shape[0] == 0:
                self.latest_output_payloads.pop(cav.vid, None)
                continue

            predictions, features_tensor = self._predict(features, target_agent_ids)
            ego_index = target_agent_ids.index(cav.vid)
            self._apply_prediction(cav.vid, predictions[ego_index], features_tensor[ego_index])

    def _predict_and_apply(self, features, target_agent_ids):
        predictions, features_tensor = self._predict(features, target_agent_ids)

        for idx, vehicle_id in enumerate(target_agent_ids):
            self._apply_prediction(vehicle_id, predictions[idx], features_tensor[idx])

    def _predict(self, features, target_agent_ids):
        self.transform_sumo2carla(features)
        features_tensor = torch.tensor(features).float().to(self.device)
        predictions = self.model.predict(features, target_agent_ids)
        return predictions, features_tensor

    def _apply_prediction(self, vehicle_id, prediction, feature_tensor):
        output_payload = self._build_output_payload(vehicle_id, prediction, feature_tensor)
        if output_payload is None:
            self.latest_output_payloads.pop(vehicle_id, None)
            return

        self.latest_output_payloads[vehicle_id] = output_payload
        self._apply_output(vehicle_id, output_payload)

    def _build_output_payload(self, vehicle_id, prediction, feature_tensor):
        pos_x, pos_y = traci.vehicle.getPosition(vehicle_id)
        curr_pos = np.array([pos_x, pos_y])

        nearest_node = self._get_nearest_node(curr_pos)
        if self.excluded_nodes and nearest_node in self.excluded_nodes:
            return None

        control_center = np.array(nearest_node.getCoord())
        distance_to_center = np.linalg.norm(curr_pos - control_center)

        if distance_to_center < self.CONTROL_RADIUS:
            pred_delta = prediction.reshape(30, 2).detach().cpu().numpy()
            local_delta = pred_delta[0].reshape(2, 1)
            last_delta = pred_delta[-1].reshape(2, 1)

            if last_delta[1, 0] <= 1:
                local_delta[1, 0] = 1e-8
                local_delta[0, 0] = 1e-10
            else:
                local_delta[1, 0] = max(1e-8, local_delta[1, 0])

            yaw = feature_tensor[3].detach().cpu().item()
            rotation = self.rotation_matrix_back(yaw)
            global_delta = (rotation @ local_delta).squeeze()
            global_delta[1] *= -1

            cav = self._get_vmanager_by_vid(vehicle_id)
            if cav is None:
                return None

            pos = cav.vehicle.get_location()

            global_delta = np.where(np.abs(global_delta) <= self.THRESHOLD, np.sign(global_delta) * self.FORCE_VALUE, global_delta)

            next_loc = carla.Location(
                x=pos.x + global_delta[0],
                y=pos.y - global_delta[1],
                z=pos.z,
            )

            return {
                "command": "set_destination",
                "prediction": pred_delta,
                "next_location": [float(next_loc.x), float(next_loc.y), float(next_loc.z)],
                "clean": True,
                "end_reset": False,
            }
        elif vehicle_id in self.mtp_controlled_vehicles:
            cav = self._get_vmanager_by_vid(vehicle_id)
            if cav is None:
                return None

            end_loc = cav.agent.end_waypoint.transform.location
            return {
                "command": "set_destination",
                "prediction": None,
                "next_location": [float(end_loc.x), float(end_loc.y), float(end_loc.z)],
                "clean": True,
                "end_reset": True,
            }

        return None

    def _apply_output(self, vehicle_id, output_payload):
        if output_payload.get("command") != "set_destination":
            return

        cav = self._get_vmanager_by_vid(vehicle_id)
        if cav is None:
            return

        next_location = output_payload.get("next_location")
        if not next_location:
            return

        next_loc = carla.Location(
            x=float(next_location[0]),
            y=float(next_location[1]),
            z=float(next_location[2]),
        )

        end_reset = bool(output_payload.get("end_reset", False))
        cav.set_destination(
            cav.vehicle.get_location(),
            next_loc,
            clean=bool(output_payload.get("clean", True)),
            end_reset=end_reset,
        )

        if end_reset:
            self.mtp_controlled_vehicles.discard(vehicle_id)
            return

        self.mtp_controlled_vehicles.add(vehicle_id)
        cav.update_info_v2x()

        if len(cav.agent.get_local_planner().get_waypoint_buffer()) == 0:
            logger.warning(f"{vehicle_id}: waypoint buffer is empty after set_destination!")

    def _get_remote_output(self, ego_id: str):
        if self.payload_handler is None:
            return None

        entity_payloads = self.payload_handler.current_artery_payload.get(ego_id, {})
        if ego_id not in entity_payloads:
            return None
        if self.module_name not in entity_payloads[ego_id]:
            return None

        with self.payload_handler.handle_artery_payload(ego_id, ego_id, self.module_name) as msg:
            return msg.get("aim_output")

    def _encoding_feature_map(self) -> dict[str, np.ndarray]:
        feature_map = {}

        for vehicle_id, trajectory in self.trajs.items():
            last_position = trajectory[-1]
            position = np.array(last_position[:2])
            distance_to_origin = np.linalg.norm(position)

            if distance_to_origin < self.CONTROL_RADIUS:
                motion_features = np.array(last_position[:-2], dtype=float)
                intention_vector = self.get_intention_vector(last_position[-1])
                feature_map[vehicle_id] = np.concatenate((motion_features, intention_vector)).astype(float)

        return feature_map

    def _encoding_features_from_messages(self, ego_id: str):
        feature_map = self._encoding_feature_map()
        if ego_id not in feature_map:
            return np.empty((0, 7)), []

        features = [feature_map[ego_id]]
        target_agent_ids = [ego_id]

        if ego_id in self.payload_handler.current_artery_payload:
            for cav_id in self.payload_handler.current_artery_payload[ego_id]:
                if cav_id == ego_id:
                    continue
                if self.module_name not in self.payload_handler.current_artery_payload[ego_id][cav_id]:
                    continue
                with self.payload_handler.handle_artery_payload(ego_id, cav_id, self.module_name) as msg:
                    if "aim_features" not in msg or msg["aim_features"] is None:
                        continue

                    features.append(np.asarray(msg["aim_features"], dtype=float))
                    target_agent_ids.append(cav_id)

        return np.vstack(features), target_agent_ids

    def update_trajs(self):
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

    def get_intention_by_rotation(self, rotation):
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

    def get_distance(self, waypoint1, waypoint2):
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
        return self.get_opencda_intention(waypoints, control_center)

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
    def transform_sumo2carla(states: np.ndarray):
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
