"""
VoxelNet implementation for 3D object detection.

This module implements the VoxelNet architecture for processing point cloud data
through voxel-based feature extraction and region proposal networks.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.utils.common_utils import torch_tensor_to_numpy


class Conv2d(nn.Module):
    """
    2D Convolution with optional batch normalization and ReLU activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    k : int
        Kernel size.
    s : int
        Stride.
    p : int
        Padding.
    activation : bool, optional
        Whether to apply ReLU activation, by default True.
    batch_norm : bool, optional
        Whether to apply batch normalization, by default True.

    Attributes
    ----------
    conv : nn.Conv2d
        2D convolution layer.
    bn : nn.BatchNorm2d or None
        Batch normalization layer if enabled, None otherwise.
    activation : bool
        Flag indicating whether to apply activation.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int, s: int, p: int, activation: bool = True, batch_norm: bool = True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Conv2d layer.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (N, C, H, W).

        Returns
        -------
        Tensor
            Output tensor with shape (N, out_channels, H', W').
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class Conv3d(nn.Module):
    """
    3D Convolution with batch normalization and ReLU activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    k : int or tuple
        Kernel size.
    s : int or tuple
        Stride.
    p : int or tuple
        Padding.
    batch_norm : bool, optional
        Whether to apply batch normalization, by default True.

    Attributes
    ----------
    conv : nn.Conv3d
        3D convolution layer.
    bn : nn.BatchNorm3d or None
        Batch normalization layer if enabled, None otherwise.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int | tuple[int, int, int],
        s: int | tuple[int, int, int],
        p: int | tuple[int, int, int],
        batch_norm: bool = True,
    ):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Conv3d layer.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (N, C, D, H, W).

        Returns
        -------
        Tensor
            Output tensor with shape (N, out_channels, D', H', W').
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


class FCN(nn.Module):
    """
    Fully Connected Network layer.

    Parameters
    ----------
    cin : int
        Number of input channels.
    cout : int
        Number of output channels.

    Attributes
    ----------
    cout : int
        Number of output channels.
    linear : nn.Linear
        Linear transformation layer.
    bn : nn.BatchNorm1d
        Batch normalization layer.
    """

    def __init__(self, cin: int, cout: int):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FCN layer.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (KK, T, cin).

        Returns
        -------
        Tensor
            Output tensor with shape (KK, T, cout).
        """
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk * t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)


class VFE(nn.Module):
    """
    Voxel Feature Encoding layer.

    Parameters
    ----------
    cin : int
        Number of input channels.
    cout : int
        Number of output channels (must be even).
    T : int
        Maximum number of points per voxel.

    Attributes
    ----------
    units : int
        Half of the output channels.
    fcn : FCN
        Fully connected network layer.
    T : int
        Maximum number of points per voxel.
    """

    def __init__(self, cin: int, cout: int, T: int):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)
        self.T = T

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of VFE layer.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (KK, T, cin).
        mask : Tensor
            Binary mask with shape (KK, T).

        Returns
        -------
        Tensor
            Output tensor with shape (KK, T, cout).
        """
        # point-wise feature
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = torch.max(pwf, 1)[0].unsqueeze(1).repeat(1, self.T, 1)
        # point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf


class SVFE(nn.Module):
    """
    Stacked Voxel Feature Encoding.

    Parameters
    ----------
    T : int
        Maximum number of points per voxel.

    Attributes
    ----------
    vfe_1 : VFE
        First voxel feature encoding layer.
    vfe_2 : VFE
        Second voxel feature encoding layer.
    fcn : FCN
        Fully connected network layer.
    """

    def __init__(self, T: int):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7, 32, T)
        self.vfe_2 = VFE(32, 128, T)
        self.fcn = FCN(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SVFE.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (KK, T, 7).

        Returns
        -------
        Tensor
            Output tensor with shape (KK, 128).
        """
        mask = torch.ne(torch.max(x, 2)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x, 1)[0]
        return x


class CML(nn.Module):
    """
    Convolutional Middle Layer for processing 3D voxel features.

    Attributes
    ----------
    conv3d_1 : Conv3d
        First 3D convolution layer.
    conv3d_2 : Conv3d
        Second 3D convolution layer.
    conv3d_3 : Conv3d
        Third 3D convolution layer.
    """

    def __init__(self) -> None:
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CML.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (N, 64, D, H, W).

        Returns
        -------
        Tensor
            Output tensor with shape (N, 64, D', H, W)
        """
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x


class RPN(nn.Module):
    """
    Region Proposal Network for 3D object detection.

    Parameters
    ----------
    anchor_num : int, optional
        Number of anchor boxes per position, by default 2.

    Attributes
    ----------
    anchor_num : int
        Number of anchor boxes per position.
    block_1 : nn.Sequential
        First block of convolutional layers.
    block_2 : nn.Sequential
        Second block of convolutional layers.
    block_3 : nn.Sequential
        Third block of convolutional layers.
    deconv_1 : nn.Sequential
        First deconvolution block.
    deconv_2 : nn.Sequential
        Second deconvolution block.
    deconv_3 : nn.Sequential
        Third deconvolution block.
    score_head : Conv2d
        Classification head for anchor scores.
    reg_head : Conv2d
        Regression head for bounding box parameters.
    """

    def __init__(self, anchor_num: int = 2):
        super(RPN, self).__init__()
        self.anchor_num = anchor_num

        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0), nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0), nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0), nn.BatchNorm2d(256))

        self.score_head = Conv2d(768, self.anchor_num, 1, 1, 0, activation=False, batch_norm=False)
        self.reg_head = Conv2d(768, 7 * self.anchor_num, 1, 1, 0, activation=False, batch_norm=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of RPN.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (N, 128, H, W).

        Returns
        -------
        tuple of Tensor
            Probability score map with shape (N, anchor_num, H', W') and
            regression map with shape (N, 7*anchor_num, H', W').
        """
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        x = torch.cat((x_0, x_1, x_2), 1)
        return self.score_head(x), self.reg_head(x)


class VoxelNet(nn.Module):
    """
    VoxelNet architecture for 3D object detection.

    Parameters
    ----------
    args : dict[str, Any]
        Configuration dictionary containing model hyperparameters.

    Attributes
    ----------
    svfe : PillarVFE
        Pillar-based voxel feature extraction module.
    cml : CML
        Convolutional middle layer.
    rpn : RPN
        Region proposal network.
    N : int
        Batch size.
    D : int
        Depth dimension of voxel grid.
    H : int
        Height dimension of voxel grid.
    W : int
        Width dimension of voxel grid.
    T : int
        Maximum number of points per voxel.
    anchor_num : int
        Number of anchor boxes per position.
    """

    def __init__(self, args: Dict[str, Any]):
        super(VoxelNet, self).__init__()
        self.svfe = PillarVFE(args["pillar_vfe"], num_point_features=4, voxel_size=args["voxel_size"], point_cloud_range=args["lidar_range"])

        # self.svfe = SVFE(args['T'])
        self.cml = CML()
        self.rpn = RPN(args["anchor_num"])

        self.N = args["N"]
        self.D = args["D"]
        self.H = args["H"]
        self.W = args["W"]
        self.T = args["T"]
        self.anchor_num = args["anchor_num"]

    def voxel_indexing(self, sparse_features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Convert sparse voxel features to dense representation.

        Parameters
        ----------
        sparse_features : Tensor
            Sparse voxel features.
        coords : Tensor
            Voxel coordinates.

        Returns
        -------
        Tensor
            Dense feature volume.
        """
        dim = sparse_features.shape[-1]

        dense_feature = Variable(torch.zeros(dim, self.N, self.D, self.H, self.W).cuda())

        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = sparse_features.transpose(0, 1)

        return dense_feature.transpose(0, 1)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of VoxelNet.

        Parameters
        ----------
        data_dict : dict of str to Any
            Input data dictionary containing processed lidar data.

        Returns
        -------
        dict of str to Tensor
            Output dictionary with keys:
            - 'psm': Probability score map with shape (N, anchor_num, H, W).
            - 'rm': Regression map with shape (N, 7*anchor_num, H, W).
        """
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]

        batch_dict = {"voxel_features": voxel_features, "voxel_coords": voxel_coords, "voxel_num_points": voxel_num_points}

        # feature learning network
        vwfs = self.svfe(batch_dict)["pillar_features"]

        voxel_coords = torch_tensor_to_numpy(voxel_coords)
        vwfs = self.voxel_indexing(vwfs, voxel_coords)

        # convolutional middle network
        vwfs = self.cml(vwfs)

        # region proposal network

        # merge the depth and feature dim into one, output probability score
        # map and regression map
        psm, rm = self.rpn(vwfs.view(self.N, -1, self.H, self.W))

        output_dict = {"psm": psm, "rm": rm}

        return output_dict
