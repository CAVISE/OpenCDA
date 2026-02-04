import torch
from torch import nn
from torch_geometric.nn import GraphConv as GNNConv


class GNN_mtl_gnn(torch.nn.Module):
    """
    Simple GNN + MLP head for per-node MTL predictions.

    The model applies several linear layers with residual connections, then two
    `GraphConv` layers, and finally projects each node embedding to 60 outputs
    (30 * 2).

    Parameters
    ----------
    hidden_channels : int
        Hidden dimension used in the MLP (after the first two linear layers)
        and in the GNN (`GraphConv`) layers.

    Attributes
    ----------
    conv1 : torch_geometric.nn.conv.GraphConv
        First graph convolution layer
    conv2 : torch_geometric.nn.conv.GraphConv
        Second graph convolution layer
    linear1 : torch.nn.Linear
        Linear layer mapping input node
    linear2 : torch.nn.Linear
        Linear layer mapping hidden_channels
    linear3 : torch.nn.Linear
        Linear layer mapping hidden_channels
    linear4 : torch.nn.Linear
        Linear layer mapping hidden_channels
    linear5 : torch.nn.Linear
        Output layer mapping hidden_channels
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
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (N, 5), where N is the number of nodes.
        edge_index : torch.Tensor
            Edge index tensor in COO format, typically shape (2, E) and dtype
            torch.long, where E is the number of edges.

        Returns
        -------
        torch.Tensor
            Per-node output tensor of shape (N, 60).
        """
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu() + x
        x = self.linear4(x).relu() + x
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.linear5(x)
        return x  # mtl
