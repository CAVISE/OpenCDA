import torch
from torch import nn
from torch_geometric.nn import GraphConv as GNNConv


class GNN_mtl_gnn(torch.nn.Module):
    """
    Graph Neural Network for Multi-Task Learning trajectory prediction.

    This model uses graph convolutions and fully connected layers to process
    agent features and predict multi-modal trajectories based on graph structure.

    Parameters
    ----------
    hidden_channels : int
        Number of hidden channels for graph convolution and linear layers.

    Attributes
    ----------
    conv1 : GNNConv
        First graph convolution layer with hidden_channels dimensions.
    conv2 : GNNConv
        Second graph convolution layer with hidden_channels dimensions.
    linear1 : nn.Linear
        Input layer transforming 5-dimensional features to 64 dimensions.
    linear2 : nn.Linear
        Second linear layer transforming 64 dimensions to hidden_channels.
    linear3 : nn.Linear
        Third linear layer with hidden_channels input and output dimensions.
    linear4 : nn.Linear
        Fourth linear layer with hidden_channels input and output dimensions.
    linear5 : nn.Linear
        Output layer producing 60-dimensional trajectory predictions (30 timesteps × 2 coordinates).
    """
    def __init__(self, hidden_channels: int):
        super().__init__()
        torch.manual_seed(21)
        self.conv1 = GNNConv(hidden_channels, hidden_channels)
        self.conv2 = GNNConv(hidden_channels, hidden_channels)
        self.linear1 = nn.Linear(5, 64)
        self.linear2 = nn.Linear(64, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels)
        self.linear4 = nn.Linear(hidden_channels, hidden_channels)
        self.linear5 = nn.Linear(hidden_channels, 30 * 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN model.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix with shape (n_nodes, 5).
        edge_index : torch.Tensor
            Graph connectivity in COO format with shape (2, n_edges).

        Returns
        -------
        torch.Tensor
            Predicted trajectories with shape (n_nodes, 60) representing
            30 timesteps of 2D coordinates for each node.
        """
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu() + x
        x = self.linear4(x).relu() + x
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.linear5(x)
        return x  # mtl
