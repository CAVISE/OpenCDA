import torch
from torch import nn
from torch_geometric.nn import GraphConv as GNNConv

from AIM.models.mtp.learning.learning_src.data_scripts.data_config import INPUT_VECTOR_SIZE, PRED_LEN, PREDICT_VECTOR_SIZE
from AIM.aim_model import MTPModel


class GNN_mtl_gnn(MTPModel):
    """
    graph neural network model using graph convolutions
    """

    def __init__(self, hidden_channels: int) -> None:
        """
        initialize gnn model with graph convolutions

        :param hidden_channels: number of hidden channels
        """
        super().__init__()
        torch.manual_seed(21)
        self.linear1 = nn.Linear(INPUT_VECTOR_SIZE, 64)
        self.linear2 = nn.Linear(64, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels)
        self.linear4 = nn.Linear(hidden_channels, hidden_channels)
        self.conv1 = GNNConv(hidden_channels, hidden_channels)
        self.conv2 = GNNConv(hidden_channels, hidden_channels)
        self.linear5 = nn.Linear(hidden_channels, PRED_LEN * PREDICT_VECTOR_SIZE)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        forward pass through gnn model

        :param x: input node features
        :param edge_index: edge indices for graph convolutions

        :return: predicted trajectories
        """
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu() + x
        x = self.linear4(x).relu() + x
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.linear5(x)
        return x  # mtl
