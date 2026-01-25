from typing import List
import torch
import numpy as np
import numpy.typing as npt

from AIM import AIMModel
from .constants import HIDDEN_CHANNELS
from .GNN_mtl_gnn import GNN_mtl_gnn
from importlib.resources import files


class MTP(AIMModel):
    """
    Multi-modal Trajectory Prediction (MTP) model.

    This class implements trajectory prediction functionality with support
    for coordinate system transformations between SUMO and CARLA.
    """
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GNN_mtl_gnn(hidden_channels=HIDDEN_CHANNELS)

        weights_path = files(__package__).joinpath("model_rot_gnn_mtl_np_sumo_0911_e3_1930.pth")
        checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)

        self.model.eval()

    def predict(self, features: npt.NDArray[np.float64], target_agent_ids: List[str]) -> torch.Tensor:
        """
        Predict trajectories for target agents.

        Parameters
        ----------
        features : NDArray[float64]
            Feature array of shape (n_agents, n_features) containing agent states.
        target_agent_ids : List[str]
            List of agent identifiers for which to predict trajectories.
        """
        num_agents = features.shape[0]
        # Preparing an agent graph for GNN
        edge_index = torch.tensor([[i, j] for i in range(num_agents) for j in range(num_agents)]).T.to(self.device)

        # Transform coordinates and make a model prediction
        self._transform_sumo2carla(features)
        x_tensor = torch.tensor(features).float().to(self.device)
        predictions = self.model(x_tensor[:, [0, 1, 4, 5, 6]], edge_index)
        return predictions

    @staticmethod
    def _transform_sumo2carla(states: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Transform coordinates from SUMO to CARLA coordinate system.

        Parameters
        ----------
        states : NDArray[float]
            Array of states in SUMO coordinate system with shape (n_states, n_dims).

        Returns
        -------
        NDArray[float]
            Transformed states in CARLA coordinate system with the same shape.
        """
        if states.ndim == 1:
            states[1] = -states[1]
            states[3] -= np.deg2rad(90)
        elif states.ndim == 2:
            states[:, 1] = -states[:, 1]
            states[:, 3] -= np.deg2rad(90)
        else:
            raise NotImplementedError
