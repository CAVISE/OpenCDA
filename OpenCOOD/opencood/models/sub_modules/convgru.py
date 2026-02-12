"""
Convolutional Gated Recurrent Unit (ConvGRU) Module.

This module implements ConvGRU cells and multi-layer ConvGRU networks for
processing spatiotemporal data using gated recurrent units with convolutional operations.
"""

from typing import List, Any, Tuple, Union, Optional
import torch
from torch import nn
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell.

    This cell implements a single GRU unit with convolutional operations
    instead of fully connected layers, suitable for processing spatial data.

    Parameters
    ----------
    input_size : tuple of int
        Height and width of input tensor as (height, width).
    input_dim : int
        Number of channels of input tensor.
    hidden_dim : int
        Number of channels of hidden state.
    kernel_size : tuple of int
        Size of the convolutional kernel as (kernel_h, kernel_w).
    bias : bool
        Whether or not to add bias in convolutions.

    Attributes
    ----------
    height : int
        Height of input tensor.
    width : int
        Width of input tensor.
    padding : tuple of int
        Padding for convolutions to maintain spatial dimensions.
    hidden_dim : int
        Number of hidden channels.
    bias : bool
        Whether bias is used.
    conv_gates : nn.Conv2d
        Convolution for computing update and reset gates.
    conv_can : nn.Conv2d
        Convolution for computing candidate hidden state.
    """

    def __init__(self, input_size: Tuple[int, int], input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], bias: bool):
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * self.hidden_dim,
            # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_can = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=self.hidden_dim,
            # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden state with zeros.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        Tensor
            Zero-initialized hidden state with shape (B, hidden_dim, H, W).
        """
        return Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))

    def forward(self, input_tensor: torch.Tensor, h_cur: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvGRU cell.

        Parameters
        ----------
        input_tensor : Tensor
            Input features with shape (B, input_dim, H, W).
        h_cur : Tensor
            Current hidden state with shape (B, hidden_dim, H, W).

        Returns
        -------
        Tensor
            Next hidden state with shape (B, hidden_dim, H, W).
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRU(nn.Module):
    """
    Multi-layer Convolutional GRU network.

    This module stacks multiple ConvGRU cells to create a deep recurrent
    network for spatiotemporal sequence processing.

    Parameters
    ----------
    input_size : tuple of int
        Height and width of input tensor as (height, width).
    input_dim : int
        Number of channels of input tensor (e.g., 256).
    hidden_dim : int or list of int
        Number of channels of hidden state for each layer (e.g., 1024 or [512, 1024]).
    kernel_size : tuple of int or list of tuple of int
        Size of the convolutional kernel for each layer.
    num_layers : int
        Number of ConvGRU layers.
    batch_first : bool, optional
        If True, input shape is (B, T, C, H, W). If False, (T, B, C, H, W).
        Default is False.
    bias : bool, optional
        Whether to add bias in convolutions. Default is True.
    return_all_layers : bool, optional
        If True, return hidden states for all layers. Otherwise, only return
        the last layer. Default is False.

    Attributes
    ----------
    height : int
        Height of input tensor.
    width : int
        Width of input tensor.
    input_dim : int
        Number of input channels.
    hidden_dim : list of int
        Number of hidden channels for each layer.
    kernel_size : list of tuple of int
        Kernel sizes for each layer.
    num_layers : int
        Number of layers.
    batch_first : bool
        Whether batch dimension is first.
    bias : bool
        Whether bias is used.
    return_all_layers : bool
        Whether to return all layer outputs.
    cell_list : nn.ModuleList
        List of ConvGRU cells for each layer.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        input_dim: int,
        hidden_dim: Union[int, List[int]],
        kernel_size: Union[Tuple[int, int], List[Tuple[int, int]]],
        num_layers: int,
        batch_first: bool = False,
        bias: bool = True,
        return_all_layers: bool = False,
    ):
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and
        # `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(
                ConvGRUCell(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor: torch.Tensor, hidden_state: Optional[List[torch.Tensor]] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through multi-layer ConvGRU.

        Parameters
        ----------
        input_tensor : Tensor
            Input sequence with shape (T, B, C, H, W) if batch_first=False,
            or (B, T, C, H, W) if batch_first=True.
        hidden_state : list of Tensor, optional
            Initial hidden states for each layer. If None, initialized to zeros.

        Returns
        -------
        layer_output_list : list of Tensor
            Output sequences from each layer (or just last layer if
            return_all_layers=False). Each tensor has shape (B, T, hidden_dim, H, W).
        last_state_list : list of list of Tensor
            Final hidden states for each layer.

        Raises
        ------
        NotImplementedError
            If hidden_state is provided (stateful mode not implemented).
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), device=input_tensor.device, dtype=input_tensor.dtype)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state
                # then compute the next hidden
                # and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],  # (b,t,c,h,w)
                    h_cur=h,
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
        """
        Initialize hidden states for all layers.

        Parameters
        ----------
        batch_size : int
            Batch size.
        device : torch.device, optional
            Device for tensors.
        dtype : torch.dtype, optional
            Data type for tensors.

        Returns
        -------
        list of Tensor
            List of initialized hidden states for each layer.
        """
        init_states = []
        for i in range(self.num_layers):
            cell = self.cell_list[i]
            assert isinstance(cell, ConvGRUCell)
            init_state = cell.init_hidden(batch_size)
            if device is not None:
                init_state = init_state.to(device)
            if dtype is not None:
                init_state = init_state.to(dtype=dtype)
            init_states.append(init_state)
        return init_states

    @staticmethod
    def _extend_for_multilayer(param: Union[Any, List[Any]], num_layers: int) -> List[Any]:
        """
        Extend parameter to list of length num_layers if not already a list.

        Parameters
        ----------
        param : Any or list of Any
            Parameter to extend.
        num_layers : int
            Number of layers.

        Returns
        -------
        list of Any
            List of parameters with length num_layers.
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
