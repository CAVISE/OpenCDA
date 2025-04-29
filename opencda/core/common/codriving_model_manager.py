

import traci
import torch
import sumolib
import logging
import importlib
import numpy as np
import pickle as pkl
from scipy.spatial import distance

from CoDriving.sripts.constants import CONTROL_RADIUS, RELEASE_RADIUS, HIDDEN_CHANNELS


logger = logging.getLogger('cavise.codriving_model_manager')

class CodrivingModelManager:
    NORMALIZED_CENTER = np.array([356.0, 356.0])

    # TODO: replace nodes
    def __init__(self, model_name, pretrained, nodes): 

        self.mtp_controlled_vehicles = set()

        self.vehicle_ids = set()
        self.time = 0
        
        self.trajs = dict()

        self.nodes = nodes
        self.node_coords = np.array([node.getCoord() for node in nodes])

        # model
        self.model_name = model_name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = self._import_model()(hidden_channels=HIDDEN_CHANNELS)
        checkpoint_dir = pretrained if len(pretrained)>0 else None

        if checkpoint_dir:
            checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint)
            self.model = self.model.to(self.device)

        self.model.eval()

        self.yaw_dict = self._load_yaw()
    
    def _load_yaw(self):
        with open("opencda/assets/yaw_dict_10m.pkl", "rb") as f:
            return pkl.load(f)

    def _get_nearest_node(self, pos):
        distances = np.linalg.norm(self.node_coords - pos, axis=1)
        return self.nodes[np.argmin(distances)]

    def _import_model(self):
        try:
            model_filename = 'CoDriving.models.' + self.model_name
            model_lib = importlib.import_module(model_filename)
            
            for name, cls in model_lib.__dict__.items():
                if name.lower() == self.model_name.lower():
                    model = cls

            return model

        except ModuleNotFoundError:
            logger.error(f'Model module {self.model_name} not found in CoDriving/models directory!')

    def make_trajs(self):
        self.vehicle_ids = traci.vehicle.getIDList()
        self.time = traci.simulation.getTime()

        self.trajs = self.update_trajs()


        x, tgt_agent_ids = self.encoding_scenario_features()
        num_tgt_agent = x.shape[0]
        if num_tgt_agent > 0:
            edge_indexs = torch.tensor([[x,y] for x in range(num_tgt_agent) for y in range(num_tgt_agent)]).T.to(self.device)
            self.transform_sumo2carla(x)
            x = torch.tensor(x).float().to(self.device)
            out = self.model(x[:,[0,1,4,5,6]], edge_indexs)
            

            for tgt in range(num_tgt_agent):
                curr_x, curr_y = traci.vehicle.getPosition(tgt_agent_ids[tgt])
                curr_pos = np.array([curr_x, curr_y])

                nearest_node = self._get_nearest_node(curr_pos)
                control_center = np.array(nearest_node.getCoord())

                dist = np.linalg.norm(curr_pos - control_center)
                
                vehicle_id = tgt_agent_ids[tgt]
                if dist < CONTROL_RADIUS:
                    if vehicle_id not in self.mtp_controlled_vehicles:
                        self.mtp_controlled_vehicles.add(vehicle_id)

                    _local_delta = out[tgt].reshape(30,2).detach().cpu().numpy()
                    local_delta = _local_delta[0].reshape(2,1)
                    last_delta = _local_delta[-1].reshape(2,1)

                    if last_delta[1,0] > 1:
                        local_delta[1,0] = max(1e-8, local_delta[1,0])
                    else:
                        local_delta[1,0] = 1e-8
                        local_delta[0,0] = 1e-10
                    
                    yaw = x[tgt, 3].detach().cpu().item()
                    rotation_back = self.rotation_matrix_back(yaw)
                    global_delta = (rotation_back @ local_delta).squeeze()
                    global_delta[1] *= -1
                    
                    pos = traci.vehicle.getPosition(vehicle_id)
                    next_x = pos[0] + global_delta[0]
                    next_y = pos[1] + global_delta[1]
                    angle = self.get_yaw(tgt_agent_ids[tgt], np.array([next_x, next_y]), self.yaw_dict)
                    try:
                        traci.vehicle.moveToXY(vehicle_id, edgeID=-1, lane=-1, x=next_x, y=next_y, angle=angle, keepRoute=2)
                    except traci.TraCIException as e:
                        logger.error(f"Failed to move vehicle {vehicle_id}: {e}")

                if dist > RELEASE_RADIUS and vehicle_id in self.mtp_controlled_vehicles:
                    self.mtp_controlled_vehicles.remove(vehicle_id)


    def update_trajs(self):
        """
        Update the dict trajs, e.g. {'left_0': [(x0, y0, speed, yaw, yaw(sumo-degree), intention(str)), (x1, y1, speed, yaw, yaw(sumo-degree), intention(str)), ...], 'left_1': [...]}
        """
        
        # add the new vehicles in the scene
        for vehicle_id in self.vehicle_ids:
            if 'carla' in vehicle_id:
                continue
            if vehicle_id not in self.trajs.keys():
                self.trajs[vehicle_id] = []

            pos = traci.vehicle.getPosition(vehicle_id)
            nearest_node = self._get_nearest_node(pos)

            speed = traci.vehicle.getSpeed(vehicle_id)

            yaw_sumo_degree = traci.vehicle.getAngle(vehicle_id)
            yaw = np.deg2rad(self.get_yaw(vehicle_id, np.array(pos), self.yaw_dict))

            norm_x, norm_y = nearest_node.getCoord()
            already_steps = len(self.trajs[vehicle_id])

            if already_steps == 0:
                intention = self.get_intention_from_vehicle_id(vehicle_id)
                self.trajs[vehicle_id].append((pos[0] - norm_x, pos[1] - norm_y, speed, yaw, yaw_sumo_degree, intention)) 
            else:
                intention = self.trajs[vehicle_id][-1][-1]
                assert isinstance(intention, str)
                self.trajs[vehicle_id].append((pos[0] - norm_x, pos[1] - norm_y, speed, yaw, yaw_sumo_degree, intention))    # TODO: normalize the time

        # remove the vehicles out of the scene
        for vehicle_id in list(self.trajs):
            if vehicle_id not in self.vehicle_ids:
                del self.trajs[vehicle_id]
        
        return self.trajs


    def get_intention_from_vehicle_id(self, vehicle_id):
        """
        Parse the vehicle id to distinguish its intention.
        """

        from_path, to_path, _ = vehicle_id.split('_')
        if from_path == 'left':
            if to_path == 'right':
                return 'straight'
            elif to_path == 'up':
                return 'left'
            elif to_path == 'down':
                return 'right'

        elif from_path == 'right':
            if to_path == 'left':
                return 'straight'
            elif to_path == 'up':
                return 'right'
            elif to_path == 'down':
                return 'left'

        elif from_path == 'up':
            if to_path == 'down':
                return 'straight'
            elif to_path == 'left':
                return 'right'
            elif to_path == 'right':
                return 'left'

        elif from_path == 'down':
            if to_path == 'up':
                return 'straight'
            elif to_path == 'right':
                return 'right'
            elif to_path == 'left':
                return 'left'

        raise Exception('Wrong vehicle id')


    def encoding_scenario_features(self):
        """
        Args:
            - trajs: e.g. {'left_0': [(x0, y0, speed, yaw, yaw', intention(str)), (x1, y1, speed, yaw, yaw', intention(str)), ...], 'left_1': [...]}
        
        Returns:
            - x: [[xs, ys, xe, ye, timestamp, left, straight, right, stop, polyline_id], ...], x.shape = [N, 10]
            - num_tgt_agent
            - num_agent
            - edge_indexs: shape = [2, edges]
            - tgt_agent_ids
        """

        x = np.empty((0, 7))
        tgt_agent_ids = []

        for vehicle_id in self.trajs.keys():

            if np.linalg.norm(self.trajs[vehicle_id][-1][:2]) < 65:
                _x = np.concatenate((np.array(self.trajs[vehicle_id][-1][:-2]), self.get_intention_vector(self.trajs[vehicle_id][-1][-1]))).reshape(1, -1) # [1, 7]
                x = np.vstack((x, _x))
                tgt_agent_ids.append(vehicle_id)

        return x, tgt_agent_ids

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
        rotation = np.array([[np.cos(-np.pi/2+yaw), -np.sin(-np.pi/2+yaw)],[np.sin(-np.pi/2+yaw), np.cos(-np.pi/2+yaw)]])
        return rotation


    @staticmethod
    def get_yaw(vehicle_id: str, pos: np.ndarray, yaw_dict: dict):
        route = '_'.join(vehicle_id.split('_')[:-1])
        if route not in yaw_dict:
            logging.warning(f"Route '{route}' not found for vehicle {vehicle_id}. Using default yaw.")
            return 0.0
        yaws = yaw_dict[route]
        dists = distance.cdist(pos.reshape(1,2), yaws[:,:-1])
        return yaws[np.argmin(dists), -1]
    
    @staticmethod
    def get_intention_vector(intention: str = 'straight')-> np.ndarray:
        """
        Return a 3-bit one-hot format intention vector.
        """

        intention_feature = np.zeros(3) 
        if intention == 'left':
            intention_feature[0] = 1
        elif intention == 'straight':
            intention_feature[1] = 1
        elif intention == 'right':
            intention_feature[2] = 1
        elif intention == 'null':
            None
        else:
            raise NotImplementedError
        return intention_feature

    
