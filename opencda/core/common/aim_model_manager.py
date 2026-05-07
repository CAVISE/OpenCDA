# import carla
import traci
import torch
import logging
import numpy as np
import pickle as pkl
from scipy.spatial import distance

# from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper
from AIM import AIMModel
from AIM.models.mtp.learning.learning_src.data_scripts.data_config import config

INTENTION = "left"

logger = logging.getLogger("cavise.codriving_model_manager")


class AIMModelManager:
    def __init__(self, model: AIMModel, nodes, excluded_nodes=None):
        """
        :param model_name: model name contained in the filename
        :param pretrained: filepath to saved model state
        :param nodes: intersections present in the simulation
        :param excluded_nodes: nodes that do not use AIM

        :return: None
        """
        self.CONTROL_RADIUS = 50
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

        self.yaw_dict = self._load_yaw()
        self.yaw_id = {}

        traci.init(port=3000, host="sumo")
        self.triggered_vehicles = [
            (1, "0", "E4_-E5", "typeWE", False, INTENTION),
            (2, "1", "E6_-E7", "typeWE", False, INTENTION),
            (2, "2", "E5_-E4", "typeWE", False, INTENTION),
            (4, "3", "E6_-E4", "typeWE", False, INTENTION),
            (4, "4", "E7_-E4", "typeWE", False, INTENTION),
            (5, "5", "E7_-E4", "typeWE", False, INTENTION),
            (6, "6", "E4_-E3", "typeWE", False, INTENTION),
            (6, "7", "E7_-E6", "typeWE", False, INTENTION),
            (6, "8", "E5_-E4", "typeWE", False, INTENTION),
            (8, "9", "E5_-E3", "typeWE", False, INTENTION),
            (8, "10", "E7_-E5", "typeWE", False, INTENTION),
            (9, "11", "E6_-E5", "typeWE", False, INTENTION),
            (12, "12", "E7_-E5", "typeWE", False, INTENTION),
        ]

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

    def spawn_triggered_vehicles(self):
        """Спавним все triggered машины из rou.xml"""

        for i, (time, vid, route_id, type_id, spawned, vec_intention) in enumerate(self.triggered_vehicles):
            try:
                if abs(time - traci.simulation.getTime()) < 0.3 and not spawned:
                    traci.vehicle.add(vehID=vid, routeID=route_id, typeID=type_id, departLane="best", departSpeed="0", departPos="base")

                    traci.vehicle.setSpeedMode(vid, 0)
                    traci.vehicle.setLaneChangeMode(vid, 0)
                    traci.vehicle.setSpeed(vid, 0)

                    print(f"[SPAWN] {vid} added and fully controlled")
                    self.triggered_vehicles[i][4] = True
            except Exception as e:
                print(f"Error spawning {vid}: {e}")

    def make_trajs(self, carla_vmanagers):
        """
        Creates new trajectories based on model predictions, assigns CAVs new destinations.

        :param carla_vmanagers: carla virtual managers
        :return: None
        """
        features, target_agent_ids = self.encoding_scenario_features()
        num_agents = features.shape[0]

        if num_agents != 0:
            move_or_not, predictions = self.model.predict(features, target_agent_ids)

        live_ids = set(traci.vehicle.getIDList())
        for idx in range(num_agents):
            vehicle_id = target_agent_ids[idx]

            if vehicle_id not in live_ids:
                continue

            pos_x, pos_y = traci.vehicle.getPosition(vehicle_id)
            curr_pos = np.array([pos_x, pos_y])
            if True:
                next_loc = predictions[0][idx].detach().cpu().numpy()[0]
                new_yaw = self.get_yaw(vehicle_id, next_loc, self.yaw_dict)
                print("predict>>>>>>>", vehicle_id, next_loc, new_yaw, np.linalg.norm(next_loc - curr_pos))

                # traci.vehicle.moveToXY(vehicle_id, edgeID="", lane=0, x=next_loc[0], y=next_loc[1], angle=new_yaw, keepRoute=2)
                traci.vehicle.moveToXY(
                    vehicle_id,
                    edgeID="",  # пусто = не привязываемся к дороге
                    laneIndex=0,
                    x=next_loc[0],
                    y=next_loc[1],
                    angle=new_yaw,
                    keepRoute=2,
                )
                new_speed = np.linalg.norm(next_loc - curr_pos) * config.temporal.sample_rate
                new_speed = new_speed if new_speed > 10 else 10
                traci.vehicle.setSpeed(vehicle_id, new_speed)
            else:
                new_yaw = self.get_yaw(vehicle_id, curr_pos, self.yaw_dict)
                traci.vehicle.moveToXY(
                    vehicle_id,
                    edgeID="",  # пусто = не привязываемся к дороге
                    laneIndex=0,
                    x=curr_pos[0],
                    y=curr_pos[1],
                    angle=new_yaw,
                    keepRoute=2,
                )
                traci.vehicle.setSpeed(vehicle_id, 0)

        for vehicle_id in self.trajs.keys():
            if vehicle_id not in target_agent_ids:
                if vehicle_id in target_agent_ids:
                    continue

                if vehicle_id not in live_ids:
                    continue

                pos_x, pos_y = traci.vehicle.getPosition(vehicle_id)
                curr_pos = np.array([pos_x, pos_y])

                yaw = self.get_yaw(vehicle_id, curr_pos, self.yaw_dict)
                speed = 4
                next_loc_x = pos_x + speed * np.cos(np.deg2rad(yaw - 90))
                next_loc_y = pos_y - speed * np.sin(np.deg2rad(yaw - 90))
                next_loc = np.array([next_loc_x, next_loc_y])

                traci.vehicle.moveToXY(vehicle_id, edgeID="", lane=0, x=next_loc[0], y=next_loc[1], angle=yaw, keepRoute=2)
                new_speed = np.linalg.norm(next_loc - curr_pos) * config.temporal.sample_rate
                # print("<><><<<<<<><><<> new speed: ", new_speed)
                traci.vehicle.setSpeed(vehicle_id, new_speed)

        self.update_trajs()
        traci.simulationStep()

    def update_trajs(self):
        """
        Updates trajectory history for ALL vehicles in SUMO, while SUMO does not auto-move them
        """
        self.spawn_triggered_vehicles()

        vehicle_ids = traci.vehicle.getIDList()
        for vehicle_id in vehicle_ids:
            try:
                traci.vehicle.setSpeedMode(vehicle_id, 0)  # полностью игнорирует всё
                traci.vehicle.setLaneChangeMode(vehicle_id, 0)  # без смены полос

                traci.vehicle.setAccel(vehicle_id, 100)
                traci.vehicle.setDecel(vehicle_id, 100)
                traci.vehicle.setEmergencyDecel(vehicle_id, 100)
                traci.vehicle.setImperfection(vehicle_id, 0.0)
                traci.vehicle.setTau(vehicle_id, 0.0)
                traci.vehicle.setMinGap(vehicle_id, 0.0)

                # Важно: постоянно "подтверждаем" нашу скорость
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                traci.vehicle.setSpeed(vehicle_id, current_speed)

                position = np.array(traci.vehicle.getPosition(vehicle_id))
                speed = traci.vehicle.getSpeed(vehicle_id)
                yaw_deg_sumo = traci.vehicle.getAngle(vehicle_id)

            except traci.exceptions.TraCIException as e:
                logger.warning(f"TraCI error for {vehicle_id}: {e}")
                continue

            if vehicle_id not in self.trajs:
                self.trajs[vehicle_id] = []

            rel_x, rel_y = position

            intention = INTENTION
            for i, (time, vid, route_id, type_id, spawned, vec_intention) in enumerate(self.triggered_vehicles):
                if vehicle_id == vid:
                    intention = vec_intention
                    break

            print("adding", rel_x, rel_y, speed, yaw_deg_sumo, intention)
            self.trajs[vehicle_id].append((rel_x, rel_y, speed, yaw_deg_sumo, intention))

        active_ids = set(vehicle_ids)
        for vid in list(self.trajs.keys()):
            if vid not in active_ids:
                del self.trajs[vid]

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

    # def get_opencda_intention(self, waypoints, mid):
    #     """
    #     Gets intention by averaged rotation to pass 3 next waypoints.

    #     :param waypoints: list of waypoint 2D-coordinates
    #     :param mid: middle of the turning path 2D-coordinates
    #     :param radius: search radius for waypoints nearby
    #     :return: intention
    #     """
    #     # Too few waypoints
    #     if len(waypoints) < 2:
    #         return "null"

    #     waypoint_index = 0
    #     location = carla.Location(mid[0], mid[1], 0)
    #     rotation = carla.Rotation(0)

    #     in_sumo_transform = carla.Transform(location, rotation)
    #     mid = BridgeHelper.get_carla_transform(in_sumo_transform, carla.Vector3D(0, 0, 0))
    #     if self.get_distance(mid, waypoints[0]) > self.CONTROL_RADIUS:
    #         logger.debug("Car not int radius")
    #         return "null"
    #     while self.get_distance(mid, waypoints[waypoint_index]) > self.CONTROL_RADIUS:
    #         waypoint_index += 1
    #         if waypoint_index >= len(waypoints):
    #             logger.debug("No waypoints in radius")
    #             return "null"
    #             # raise IndexError("No waypoints in radius")

    #     first_waypoint = waypoints[waypoint_index]
    #     first_waypoint_index = waypoint_index
    #     while (self.get_distance(mid, waypoints[waypoint_index]) <= self.CONTROL_RADIUS) and waypoint_index < len(waypoints) - 1:
    #         waypoint_index += 1

    #     # average the values of several points to reduce noise
    #     mean_yaw = 0
    #     if waypoint_index - first_waypoint_index < 3 and waypoint_index > 3:
    #         logger.warning("Too few waypoints in radius")
    #     else:
    #         for i in range(3):
    #             mean_yaw += waypoints[waypoint_index - i][0].transform.rotation.yaw
    #     mean_yaw //= 3
    #     rotation = (mean_yaw - first_waypoint[0].transform.rotation.yaw + 360) % 360
    #     return self.get_intention_by_rotation(rotation)

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

        Uses live TraCI pose (x, y, speed, yaw) when the vehicle exists in SUMO so the model
        sees the same coordinates the network is in, not a stale trajs[-1] from before the last step.

        :return: x: list((motion features, intention vector)), target: agent ids list(vehicle id)
        """
        features = []
        target_agent_ids = []
        live_ids = set(traci.vehicle.getIDList())

        for vehicle_id, trajectory in self.trajs.items():
            last_position = trajectory[-1]
            try:
                if vehicle_id in live_ids:
                    px, py = traci.vehicle.getPosition(vehicle_id)
                    spd = traci.vehicle.getSpeed(vehicle_id)
                    yaw_deg = traci.vehicle.getAngle(vehicle_id)
                    motion_features = np.array([px, py, spd, yaw_deg])
                    position = np.array([px, py])
                else:
                    motion_features = np.array(last_position[:4])
                    position = np.array(last_position[:2])
            except traci.exceptions.TraCIException:
                motion_features = np.array(last_position[:4])
                position = np.array(last_position[:2])

            distance_to_origin = np.linalg.norm(position)

            if distance_to_origin < self.CONTROL_RADIUS:
                start_yaw = np.array([trajectory[0][3]])
                last_yaw = start_yaw.copy()

                if last_position[-1] == "right":
                    last_yaw += 90
                elif last_position[-1] == "left":
                    last_yaw -= 90

                last_yaw = (last_yaw + 360) % 360

                # intention_vector = self.get_intention_vector(last_position[-1])
                # feature_vector = np.concatenate((motion_features, intention_vector)).reshape(1, -1)
                feature_vector = np.concatenate((motion_features, start_yaw, last_yaw)).reshape(1, -1)

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

        intention = self.trajs[vehicle_id][0][-1]

        if 45 <= self.trajs[vehicle_id][0][3] < 135:
            start = "left"
        elif 135 <= self.trajs[vehicle_id][0][3] < 225:
            start = "up"
        elif 225 <= self.trajs[vehicle_id][0][3] < 315:
            start = "right"
        else:
            start = "down"

        end = self.get_end(start, intention)
        v = f"{start}_{end}"
        print("get yaw>>>>>>>>>", intention, vehicle_id, v)

        if vehicle_id not in self.yaw_id:
            self.yaw_id[vehicle_id] = v

        route = self.yaw_id[vehicle_id]
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
