import torch
from torch import nn
from torch_geometric.nn import GraphConv as GNNConv
from huggingface_hub import PyTorchModelHubMixin
from CoDriving.data_scripts.data_config.data_config import INPUT_VECTOR_SIZE, PRED_LEN, PREDICT_VECTOR_SIZE


class GNN_mtl_gnn(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(21)
        self.linear1 = nn.Linear(INPUT_VECTOR_SIZE, 64)
        self.linear2 = nn.Linear(64, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels)
        self.linear4 = nn.Linear(hidden_channels, hidden_channels)
        self.conv1 = GNNConv(hidden_channels, hidden_channels)
        self.conv2 = GNNConv(hidden_channels, hidden_channels)
        self.linear5 = nn.Linear(hidden_channels, PRED_LEN * PREDICT_VECTOR_SIZE)

    def forward(self, x, edge_index):
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu() + x
        x = self.linear4(x).relu() + x
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.linear5(x)
        return x  # mtl
