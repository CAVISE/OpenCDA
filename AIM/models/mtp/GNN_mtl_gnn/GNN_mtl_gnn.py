import torch
from torch import nn
from torch_geometric.nn import GraphConv as GNNConv
from typing import Any


class GNN_mtl_gnn(torch.nn.Module):
    """
    Graph neural network for multi-agent trajectory prediction.

    Combines MLP feature encoding with graph convolution layers
    to model agent interactions.
    """

    def __init__(self, hidden_channels: int) -> None:
        """
        Parameters
        ----------
        hidden_channels : int
            Hidden feature dimensionality.
        """
        super().__init__()

        torch.manual_seed(21)

        self.conv1: GNNConv = GNNConv(hidden_channels, hidden_channels)
        self.conv2: GNNConv = GNNConv(hidden_channels, hidden_channels)

        self.linear1: nn.Linear = nn.Linear(5, 64)
        self.linear2: nn.Linear = nn.Linear(64, hidden_channels)
        self.linear3: nn.Linear = nn.Linear(hidden_channels, hidden_channels)
        self.linear4: nn.Linear = nn.Linear(hidden_channels, hidden_channels)
        self.linear5: nn.Linear = nn.Linear(hidden_channels, 30 * 2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [N, F].
        edge_index : torch.Tensor
            Graph connectivity in COO format [2, E].

        Returns
        -------
        torch.Tensor
            Predicted trajectories [N, 60].
        """
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu() + x
        x = self.linear4(x).relu() + x
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.linear5(x)
        return x
