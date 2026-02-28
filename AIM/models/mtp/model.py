import torch
import numpy as np

from AIM import AIMModel
from .GNN_mtl_gnn.GNN_mtl_gnn import GNN_mtl_gnn
from importlib.resources import files


class MTP(AIMModel):
    def __init__(self, **kwargs):
        super().__init__()
        self._models = {"GNN_mtl_gnn": GNN_mtl_gnn}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hidden_channels = kwargs.get("hidden_channels", 128)
        underling_model = kwargs.get("underling_model", "GNN_mtl_gnn")
        weight = kwargs.get("weights", "model_rot_gnn_mtl_np_sumo_0911_e3_1930.pth")
        self.model = self._models[underling_model](hidden_channels=hidden_channels)

        weights_path = files(__package__).joinpath(f"{underling_model}/weights/{weight}")
        checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)

        self.model.eval()

    def predict(self, features, target_agent_ids):
        num_agents = features.shape[0]
        # Preparing an agent graph for GNN
        edge_index = torch.tensor([[i, j] for i in range(num_agents) for j in range(num_agents)]).T.to(self.device)

        # Transform coordinates and make a model prediction
        self._transform_sumo2carla(features)
        x_tensor = torch.tensor(features).float().to(self.device)
        predictions = self.model(x_tensor[:, [0, 1, 4, 5, 6]], edge_index)
        return predictions

    @staticmethod
    def _transform_sumo2carla(states: np.ndarray):
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
