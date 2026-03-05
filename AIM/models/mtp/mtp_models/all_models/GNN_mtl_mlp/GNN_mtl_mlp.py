import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from AIM.models.mtp.learning.learning_src.data_scripts.data_config import INPUT_VECTOR_SIZE, PRED_LEN, PREDICT_VECTOR_SIZE


class GNN_mtl_mlp(torch.nn.Module, PyTorchModelHubMixin):
    """
    graph neural network model using mlp layers instead of graph convolutions
    """

    def __init__(self, hidden_channels: int) -> None:
        """
        initialize gnn model with mlp layers

        :param hidden_channels: number of hidden channels
        """
        super().__init__()
        torch.manual_seed(21)
        self.linear1 = nn.Linear(INPUT_VECTOR_SIZE, 64)
        self.linear2 = nn.Linear(64, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels)
        self.linear4 = nn.Linear(hidden_channels, hidden_channels)
        self.conv1 = nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.linear5 = nn.Linear(hidden_channels, PRED_LEN * PREDICT_VECTOR_SIZE)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        forward pass through gnn model

        :param x: input node features
        :param edge_index: edge indices (not used in mlp version)

        :return: predicted trajectories
        """
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu() + x
        x = self.linear4(x).relu() + x
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.linear5(x)
        return x  # mtl
