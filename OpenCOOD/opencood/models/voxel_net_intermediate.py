"""
VoxelNet with intermediate feature fusion for 3D object detection.

This module implements a VoxelNet variant that performs intermediate fusion
of features from multiple agents using attention mechanisms.
"""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from opencood.models.voxel_net import RPN, CML
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.auto_encoder import AutoEncoder


class Conv2d(nn.Module):
    """
    A 2D convolutional layer with optional batch normalization and ReLU activation.

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
        Whether to use batch normalization, by default True.
    bias : bool, optional
        Whether to add bias to the convolution, by default True.

    Attributes
    ----------
    conv : nn.Conv2d
        2D convolution layer.
    bn : nn.BatchNorm2d or None
        Batch normalization layer if enabled, None otherwise.
    activation : bool
        Flag indicating whether to apply activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        s: int,
        p: int,
        activation: bool = True,
        batch_norm: bool = True,
        bias: bool = True
    ):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=bias)
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
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor after convolution, batch norm, and activation.
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class NaiveFusion(nn.Module):
    """
    A simple fusion module that concatenates multiple feature maps and applies convolutions.

    This module processes concatenated features from multiple agents through
    a series of convolutional layers to produce fused representations.

    Attributes
    ----------
    conv1 : Conv2d
        First convolutional layer reducing concatenated features.
    conv2 : Conv2d
        Second convolutional layer producing final fused features.
    """

    def __init__(self):
        super(NaiveFusion, self).__init__()
        self.conv1 = Conv2d(128 * 5, 256, 3, 1, 1, batch_norm=False, bias=False)
        self.conv2 = Conv2d(256, 128, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of NaiveFusion.

        Parameters
        ----------
        x : Tensor
            Concatenated feature tensor from multiple agents.

        Returns
        -------
        Tensor
            Fused feature tensor.
        """
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class VoxelNetIntermediate(nn.Module):
    """
    VoxelNet with intermediate fusion for cooperative 3D object detection.

    This model extends VoxelNet to support multi-agent cooperative perception
    by fusing intermediate features from multiple vehicles using attention mechanisms.

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
    fusion_net : AttFusion
        Attention-based fusion network for multi-agent features.
    rpn : RPN
        Region proposal network.
    N : int
        Batch size or total number of samples.
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
    compression : bool
        Flag indicating whether compression is enabled.
    compression_layer : AutoEncoder, optional
        Autoencoder for feature compression if compression is enabled.
    """

    def __init__(self, args: Dict[str, Any]):
        super(VoxelNetIntermediate, self).__init__()
        self.svfe = PillarVFE(args["pillar_vfe"], num_point_features=4, voxel_size=args["voxel_size"], point_cloud_range=args["lidar_range"])
        self.cml = CML()
        self.fusion_net = AttFusion(128)
        self.rpn = RPN(args["anchor_num"])

        self.N = args["N"]
        self.D = args["D"]
        self.H = args["H"]
        self.W = args["W"]
        self.T = args["T"]
        self.anchor_num = args["anchor_num"]

        self.compression = False
        if "compression" in args and args["compression"] > 0:
            self.compression = True
            self.compression_layer = AutoEncoder(128, args["compression"])

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

    def regroup(self, dense_feature: torch.Tensor, record_len: list) -> torch.Tensor:
        """
        Regroup the data based on the record_len.

        Parameters
        ----------
        dense_feature : torch.Tensor
            Dense features of shape (N, C, H, W).
        record_len : list
            List of sample lengths [sample1_len, sample2_len, ...].

        Returns
        -------
        torch.Tensor
            Regrouped features of shape (B, 5C, H, W).
        """
        cum_sum_len = list(np.cumsum(record_len))
        split_features = torch.tensor_split(dense_feature, cum_sum_len[:-1])
        regroup_features = []

        for split_feature in split_features:
            # M, C, H, W
            feature_shape = split_feature.shape

            # the maximum M is 5 as most 5 cavs
            padding_len = 5 - feature_shape[0]
            padding_tensor = torch.zeros(padding_len, feature_shape[1], feature_shape[2], feature_shape[3])
            padding_tensor = padding_tensor.to(split_feature.device)

            split_feature = torch.cat([split_feature, padding_tensor], dim=0)

            # 1, 5C, H, W
            split_feature = split_feature.view(-1, feature_shape[2], feature_shape[3]).unsqueeze(0)
            regroup_features.append(split_feature)

        # B, 5C, H, W
        regroup_features = torch.cat(regroup_features, dim=0)

        return regroup_features

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VoxelNetIntermediate model.

        Parameters
        ----------
        data_dict : dict of str to Any
            Input data dictionary containing:
            - 'processed_lidar': Dictionary with voxel features, coordinates, and point counts.
            - 'record_len': Tensor indicating number of agents per batch sample.

        Returns
        -------
        dict of str to Tensor
            Output dictionary with keys:
            - 'psm': Probability score map with shape (B, anchor_num, H, W).
            - 'rm': Regression map with shape (B, 7*anchor_num, H, W).
        """
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]

        batch_dict = {"voxel_features": voxel_features, "voxel_coords": voxel_coords, "voxel_num_points": voxel_num_points}

        record_len_tmp = record_len.cpu() if voxel_coords.is_cuda else record_len

        record_len_tmp = list(record_len_tmp.numpy())

        self.N = sum(record_len_tmp)

        # feature learning network
        vwfs = self.svfe(batch_dict)["pillar_features"]

        voxel_coords = torch_tensor_to_numpy(voxel_coords)
        vwfs = self.voxel_indexing(vwfs, voxel_coords)

        # convolutional middle network
        vwfs = self.cml(vwfs)
        # convert from 3d to 2d N C H W
        vmfs = vwfs.view(self.N, -1, self.H, self.W)

        # compression layer
        if self.compression:
            vmfs = self.compression_layer(vmfs)

        # information naive fusion
        vmfs_fusion = self.fusion_net(vmfs, record_len)

        # region proposal network
        # merge the depth and feature dim into one, output probability score
        # map and regression map
        psm, rm = self.rpn(vmfs_fusion)

        output_dict = {"psm": psm, "rm": rm}

        return output_dict
