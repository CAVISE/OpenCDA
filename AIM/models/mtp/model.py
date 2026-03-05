import torch
import numpy as np

from typing import Any, Dict, List
from importlib.resources import files

from AIM import AIMModel
from .GNN_mtl_gnn.GNN_mtl_gnn import GNN_mtl_gnn


class MTP(AIMModel):
    """
    Multi-agent trajectory prediction model using a GNN backend.

    Loads pretrained weights and performs trajectory prediction
    for multiple interacting agents.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the MTP model.

        Parameters
        ----------
        hidden_channels : int, optional
            Hidden dimensionality of the GNN (default: 128).
        underling_model : str, optional
            Backend model name (default: "GNN_mtl_gnn").
        weights : str, optional
            Filename of pretrained weights.
        """
        super().__init__()

        self._models: Dict[str, Any] = {
            "GNN_mtl_gnn": GNN_mtl_gnn
        }

        self.device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        hidden_channels: int = kwargs.get("hidden_channels", 128)
        underling_model: str = kwargs.get("underling_model", "GNN_mtl_gnn")
        weight: str = kwargs.get(
            "weights", "model_rot_gnn_mtl_np_sumo_0911_e3_1930.pth"
        )

        self.model: torch.nn.Module = self._models[underling_model](
            hidden_channels=hidden_channels
        )

        weights_path = files(__package__).joinpath(
            f"{underling_model}/weights/{weight}"
        )

        checkpoint = torch.load(
            weights_path, map_location=torch.device("cpu")
        )

        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        features: np.ndarray,
        target_agent_ids: List[str]
    ) -> torch.Tensor:
        """
        Predict future trajectories.

        Parameters
        ----------
        features : np.ndarray
            Agent feature matrix of shape [N, D].
        target_agent_ids : list[str]
            Target agent identifiers (currently unused).

        Returns
        -------
        torch.Tensor
            Predicted trajectories of shape [N, 60].
        """
        num_agents: int = features.shape[0]

        edge_index: torch.Tensor = torch.tensor(
            [[i, j] for i in range(num_agents) for j in range(num_agents)]
        ).T.to(self.device)

        self._transform_sumo2carla(features)

        x_tensor: torch.Tensor = (
            torch.tensor(features).float().to(self.device)
        )

        predictions: torch.Tensor = self.model(
            x_tensor[:, [0, 1, 4, 5, 6]],
            edge_index
        )

        return predictions

    @staticmethod
    def _transform_sumo2carla(states: np.ndarray) -> None:
        """
        In-place transform from SUMO to CARLA coordinates.

        Parameters
        ----------
        states : np.ndarray
            Agent state array (1D or 2D).

        Raises
        ------
        NotImplementedError
            If input dimensionality is unsupported.
        """
        if states.ndim == 1:
            states[1] = -states[1]
            states[3] -= np.deg2rad(90)
        elif states.ndim == 2:
            states[:, 1] = -states[:, 1]
            states[:, 3] -= np.deg2rad(90)
        else:
            raise NotImplementedError
