import carla
import traci
import torch
import logging
import importlib
import numpy as np
import pickle as pkl
from scipy.spatial import distance

from CoDriving.scripts.constants import CONTROL_RADIUS, HIDDEN_CHANNELS
from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper


logger = logging.getLogger("cavise.codriving_model_manager")


class CodrivingModelManager:
    def __init__(self, model_name, pretrained, nodes, excluded_nodes=None):
        self.mtp_controlled_vehicles = set()

        self.sumo_cavs_ids = set()  # ids
        self.carla_vmanagers = set()  # Vehicle Managers

        self.trajs = dict()

        self.nodes = nodes
        self.node_coords = np.array([node.getCoord() for node in nodes])
        self.excluded_nodes = excluded_nodes  # Перекрестки на которых отключен MTP модуль

        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self._import_model()(hidden_channels=HIDDEN_CHANNELS)
        checkpoint_dir = pretrained if len(pretrained) > 0 else None

        if checkpoint_dir:
            checkpoint = torch.load(checkpoint_dir, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint)
            self.model = self.model.to(self.device)

        self.model.eval()

        self.yaw_dict = self._load_yaw()
        self.yaw_id = {}

    def _load_yaw(self):
        with open("opencda/assets/yaw_dict_10m.pkl", "rb") as f:
            return pkl.load(f)

    def _get_nearest_node(self, pos):
        distances = np.linalg.norm(self.node_coords - pos, axis=1)
        return self.nodes[np.argmin(distances)]

    def _import_model(self):
        try:
            model_filename = "CoDriving.models." + self.model_name
            model_lib = importlib.import_module(model_filename)

            for name, cls in model_lib.__dict__.items():
                if name.lower() == self.model_name.lower():
                    model = cls

            return model

        except ModuleNotFoundError:
            logger.error(f"Model module {self.model_name} not found in CoDriving/models directory!")

    def _get_vmanager_by_vid(self, vid: str):
        for vmanager in self.carla_vmanagers:
            if vmanager.vid == vid:
                return vmanager
        return None

    def _is_carla_id(self, vid):
        for vmanager in self.carla_vmanagers:
            if vmanager.vid == vid:
                return True
        return False

    def make_trajs(self, carla_vmanagers):
        # Сохраняем список машин из SUMO и CARLA
        self.sumo_cavs_ids = traci.vehicle.getIDList()

        self.carla_vmanagers = carla_vmanagers

        # Обновляем траектории всех машин (SUMO + CARLA)
        self.trajs = self.update_trajs()

        # Получаем признаки агентов и список их идентификаторов
        x, target_agent_ids = self.encoding_scenario_features()
        num_agents = x.shape[0]

        if num_agents == 0:
            return

        # Подготовка графа агентов для GNN
        edge_index = torch.tensor([[i, j] for i in range(num_agents) for j in range(num_agents)]).T.to(self.device)

        # Преобразуем координаты и делаем предсказание модели
        self.transform_sumo2carla(x)
        x_tensor = torch.tensor(x).float().to(self.device)
        predictions = self.model(x_tensor[:, [0, 1, 4, 5, 6]], edge_index)

        # Обрабатываем каждого агента
        for idx in range(num_agents):
            vehicle_id = target_agent_ids[idx]

            # Получаем текущую позицию
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

                yaw = x_tensor[idx, 3].detach().cpu().item()
                rotation = self.rotation_matrix_back(yaw)
                global_delta = (rotation @ local_delta).squeeze()
                global_delta[1] *= -1
                if self._is_carla_id(vehicle_id):
                    cav = self._get_vmanager_by_vid(vehicle_id)
                    if cav is None:
                        continue

                    pos = cav.vehicle.get_location()

                    threshold = 10
                    force_value = 20

                    global_delta = np.where(np.abs(global_delta) <= threshold, np.sign(global_delta) * force_value, global_delta)

                    next_loc = carla.Location(
                        x=pos.x + global_delta[0],
                        y=pos.y - global_delta[1],
                        z=pos.z,
                    )

                    cav.set_destination(pos, next_loc, clean=True, end_reset=False)
                    cav.update_info_v2x()

                    if len(cav.agent.get_local_planner().get_waypoint_buffer()) == 0:
                        logger.warning(f"{vehicle_id}: waypoint buffer is empty after set_destination!")
                else:
                    try:
                        next_x = pos_x + global_delta[0]
                        next_y = pos_y + global_delta[1]
                        angle = self.get_yaw(vehicle_id, np.array([next_x, next_y]), self.yaw_dict)
                        traci.vehicle.moveToXY(vehicle_id, edgeID=-1, lane=-1, x=next_x, y=next_y, angle=angle, keepRoute=2)
                    except traci.TraCIException as e:
                        logger.error(f"Failed to move vehicle {vehicle_id}: {e}")
            elif vehicle_id in self.mtp_controlled_vehicles:
                if self._is_carla_id(vehicle_id):
                    cav = self._get_vmanager_by_vid(vehicle_id)
                    cav.set_destination(cav.vehicle.get_location(), cav.agent.end_waypoint.transform.location, clean=True, end_reset=True)

                self.mtp_controlled_vehicles.remove(vehicle_id)

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
        for vehicle_id in self.sumo_cavs_ids:
            # Initialize trajectory if this is a new vehicle
            if vehicle_id not in self.trajs:
                self.trajs[vehicle_id] = []

            # Get current vehicle position and find nearest node
            position = np.array(traci.vehicle.getPosition(vehicle_id))
            nearest_node = self._get_nearest_node(position)

            # Skip excluded regions
            if self.excluded_nodes and nearest_node in self.excluded_nodes:
                continue

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
                assert isinstance(intention, str)

            # Append current state to trajectory
            self.trajs[vehicle_id].append((rel_x, rel_y, speed, yaw_rad, yaw_deg_sumo, intention))

        # Remove trajectories of vehicles that have left the scene
        for vehicle_id in list(self.trajs):
            if vehicle_id not in self.sumo_cavs_ids:
                del self.trajs[vehicle_id]

        return self.trajs

    def get_intention_from_vehicle_id(self, vehicle_id):
        """
        Parse the vehicle id to distinguish its intention.
        """
        # TODO: Intetion должно браться из сообщения от ТС, а не id/name
        if self._is_carla_id(vehicle_id):
            from_path, to_path = "left", "down"
        else:
            from_path, to_path, *_ = vehicle_id.split("_")

        if from_path == "left":
            if to_path == "right":
                return "straight"
            elif to_path == "up":
                return "left"
            elif to_path == "down":
                return "right"

        elif from_path == "right":
            if to_path == "left":
                return "straight"
            elif to_path == "up":
                return "right"
            elif to_path == "down":
                return "left"

        elif from_path == "up":
            if to_path == "down":
                return "straight"
            elif to_path == "left":
                return "right"
            elif to_path == "right":
                return "left"

        elif from_path == "down":
            if to_path == "up":
                return "straight"
            elif to_path == "right":
                return "right"
            elif to_path == "left":
                return "left"

        raise Exception("Wrong vehicle id")

    def get_intention_by_rotation(self, rotation):
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
        rel_x = waypoint1.location.x - waypoint2[0].transform.location.x
        rel_y = waypoint1.location.y - waypoint2[0].transform.location.y
        position = np.array([rel_x, rel_y])
        return np.linalg.norm(position)

    def get_opencda_intention(self, waypoints, mid, radius=CONTROL_RADIUS):
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
            # else:
            #     print("No way, points in radius")

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

    def get_sumo_intention(self, sumo_id):
        route = traci.vehicle.getRoute(sumo_id)
        index = traci.vehicle.getRouteIndex(sumo_id)

        current_edge = route[index]
        if index + 1 < len(route):
            next_edge = route[index + 1]
        else:
            logger.debug("Last node")
            return "null"
        current_angle = traci.edge.getAngle(current_edge)
        next_angle = traci.edge.getAngle(next_edge)
        rotation = (next_angle - current_angle + 360) % 360
        return self.get_intention_by_rotation(rotation)

    def get_intention(self, vehicle_id):
        """
        Parse the vehicle id to distinguish its intention.
        """
        if self._is_carla_id(vehicle_id):
            cav = self._get_vmanager_by_vid(vehicle_id)
            cav.set_destination(cav.vehicle.get_location(), cav.agent.end_waypoint.transform.location, clean=True, end_reset=True)
            waypoints = cav.agent.get_local_planner().get_waypoint_buffer()
            curr_pos = np.array(traci.vehicle.getPosition(vehicle_id))
            nearest_node = self._get_nearest_node(curr_pos)
            control_center = nearest_node.getCoord()
            return self.get_opencda_intention(waypoints, control_center, CONTROL_RADIUS)
        else:
            return self.get_sumo_intention(vehicle_id)

    def encoding_scenario_features(self):
        features = []
        target_agent_ids = []

        for vehicle_id, trajectory in self.trajs.items():
            last_position = trajectory[-1]
            position = np.array(last_position[:2])
            distance_to_origin = np.linalg.norm(position)

            if distance_to_origin < 65:
                motion_features = np.array(last_position[:-2])
                intention_vector = self.get_intention_vector(last_position[-1])
                feature_vector = np.concatenate((motion_features, intention_vector)).reshape(1, -1)

                features.append(feature_vector)
                target_agent_ids.append(vehicle_id)

        x = np.vstack(features) if features else np.empty((0, 7))
        return x, target_agent_ids

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
        nearest_node = self._get_nearest_node(pos)
        if not self.trajs[vehicle_id] or self.trajs[vehicle_id][-1][-1] == "null":
            logging.warning("Intention isn't defined")
            return 0.0
        else:
            intention = self.trajs[vehicle_id][-1][-1]

        if self._is_carla_id(vehicle_id):
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
                    self.trajs[vehicle_id].append(
                        (previous_traj[0], previous_traj[1], previous_traj[2], previous_traj[3], previous_traj[4], intention)
                    )
            route = self.yaw_id[vehicle_id][nearest_node]
        else:
            rotation = traci.vehicle.getAngle(vehicle_id)
            if rotation < 45 or rotation > 315:
                start = "down"
            elif rotation < 135:
                start = "left"
            elif rotation > 225:
                start = "up"
            else:
                start = "right"
            end = self.get_end(start, intention)
            v = f"{start}_{end}"
            if vehicle_id not in self.yaw_id:
                self.yaw_id[vehicle_id] = {nearest_node: v}
            else:
                if nearest_node not in self.yaw_id[vehicle_id]:
                    self.yaw_id[vehicle_id] = {nearest_node: v}
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
            None
        else:
            raise NotImplementedError
        return intention_feature
