import torch
from torch import nn


class GNN_mtl_mlp(torch.nn.Module):
    """
    MLP-only (no message passing) per-node MTL head.

    Despite the names `conv1/conv2`, this module uses two `nn.Linear` layers
    instead of graph convolutions; the `edge_index` argument is accepted only
    to keep the same forward signature as the GNN-based variant.

    Parameters
    ----------
    hidden_channels : int
        Hidden dimension used in the MLP stack and in the final projection.

    Attributes
    ----------
    conv1 : torch.nn.Linear
        Linear layer mapping `hidden_channels` -> `hidden_channels`.
    conv2 : torch.nn.Linear
        Linear layer mapping `hidden_channels` -> `hidden_channels`.
    linear1 : torch.nn.Linear
        Linear layer mapping input node features 5 -> 64.
    linear2 : torch.nn.Linear
        Linear layer mapping 64 -> `hidden_channels`.
    linear3 : torch.nn.Linear
        Residual linear layer mapping `hidden_channels` -> `hidden_channels`.
    linear4 : torch.nn.Linear
        Residual linear layer mapping `hidden_channels` -> `hidden_channels`.
    linear5 : torch.nn.Linear
        Output projection mapping `hidden_channels` -> 60 (30 * 2).
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        torch.manual_seed(21)
        self.conv1 = nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)
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
            Unused in this implementation. Kept for API compatibility with
            graph-based models; typically has shape (2, E) and dtype `torch.long`.

        Returns
        -------
        torch.Tensor
            Per-node output tensor of shape (N, 60).
        """
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu() + x
        x = self.linear4(x).relu() + x
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.linear5(x)
        return x  # mtl
