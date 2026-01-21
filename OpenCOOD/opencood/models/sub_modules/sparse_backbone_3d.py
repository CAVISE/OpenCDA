"""
Sparse 3D Convolutional Backbone for Voxel-based 3D Detection.

This module implements a sparse 3D CNN backbone with 8x downsampling
for efficient processing of voxelized point cloud data.
"""

from functools import partial

from typing import Dict, List, Tuple, Optional, Union, Any
import torch.nn as nn

try:  # spconv1
    from spconv import SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor
except ImportError:  # spconv2
    from spconv.pytorch import SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor


def post_act_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int, int]],
    indice_key: Optional[str] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    conv_type: str = "subm",
    norm_fn: Optional[callable] = None,
) -> SparseSequential:
    """
    Create sparse convolution block with post-activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int
        Convolution kernel size.
    indice_key : str, optional
        Index key for sparse convolution.
    stride : int or tuple of int, optional
        Convolution stride. Default is 1.
    padding : int or tuple of int, optional
        Convolution padding. Default is 0.
    conv_type : str, optional
        Type of convolution ('subm', 'spconv', 'inverseconv'). Default is 'subm'.
    norm_fn : callable, optional
        Normalization function (e.g., BatchNorm1d).

    Returns
    -------
    SparseSequential
        Sequential sparse convolution block with normalization and ReLU.

    Raises
    ------
    NotImplementedError
        If conv_type is not one of 'subm', 'spconv', or 'inverseconv'.
    """
    if conv_type == "subm":
        conv = SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == "spconv":
        conv = SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == "inverseconv":
        conv = SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class VoxelBackBone8x(nn.Module):
    """
    Sparse 3D CNN backbone with 8x downsampling.

    This backbone processes voxelized point cloud data using sparse 3D convolutions,
    producing multi-scale features for 3D object detection.

    Parameters
    ----------
    model_cfg : dict of str to Any
        Model configuration dictionary containing:
        - 'num_features_out': Output feature dimension (optional).
    input_channels : int
        Number of input feature channels from voxel feature encoder.
    grid_size : list of int
        Voxel grid size [X, Y, Z].
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    model_cfg : dict
        Model configuration.
    sparse_shape : list of int
        Sparse tensor spatial shape [Z, Y, X, 1, 0, 0].
    conv_input : SparseSequential
        Input convolution block.
    conv1 : SparseSequential
        First convolution stage (stride 1).
    conv2 : SparseSequential
        Second convolution stage (stride 2, 2x downsample).
    conv3 : SparseSequential
        Third convolution stage (stride 2, 4x downsample).
    conv4 : SparseSequential
        Fourth convolution stage (stride 2, 8x downsample).
    conv_out : SparseSequential
        Output convolution for detection head.
    num_point_features : int
        Output feature dimension.
    backbone_channels : dict
        Channel dimensions for each convolution stage.
    """

    def __init__(self, model_cfg: Dict[str, Any], input_channels: int, grid_size: List[int], **kwargs) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = SparseSequential(
            SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key="subm1"),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key="subm1"),
        )

        self.conv2 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key="spconv2", conv_type="spconv"),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key="subm2"),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key="subm2"),
        )

        self.conv3 = SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key="spconv3", conv_type="spconv"),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm3"),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm3"),
        )

        self.conv4 = SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key="spconv4", conv_type="spconv"),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm4"),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key="subm4"),
        )

        last_pad = 0
        if "num_features_out" in self.model_cfg:
            self.num_point_features = self.model_cfg["num_features_out"]
        else:
            self.num_point_features = 128
        self.conv_out = SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            SparseConv3d(64, self.num_point_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key="spconv_down2"),
            norm_fn(self.num_point_features),
            nn.ReLU(),
        )

        self.backbone_channels = {"x_conv1": 16, "x_conv2": 32, "x_conv3": 64, "x_conv4": 64}  # noqa: DC05

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass through sparse 3D backbone.

        Parameters
        ----------
        batch_dict : dict of str to Any
            Batch dictionary containing:
            - 'batch_size': Batch size (int).
            - 'voxel_features': Voxel features with shape (N_voxels, C).
            - 'voxel_coords': Voxel coordinates with shape (N_voxels, 4)
              in format [batch_idx, z_idx, y_idx, x_idx].

        Returns
        -------
        dict of str to Any
            Updated batch dictionary with:
            - 'encoded_spconv_tensor': Final sparse tensor for detection head.
            - 'encoded_spconv_tensor_stride': Downsampling stride (8).
            - 'multi_scale_3d_features': Dictionary of multi-scale sparse features.
            - 'multi_scale_3d_strides': Dictionary of strides for each scale.
        """
        voxel_features, voxel_coords = batch_dict["voxel_features"], batch_dict["voxel_coords"]
        batch_size = batch_dict["batch_size"]
        input_sp_tensor = SparseConvTensor(
            features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({"encoded_spconv_tensor": out, "encoded_spconv_tensor_stride": 8})
        batch_dict.update(
            {
                "multi_scale_3d_features": {
                    "x_conv1": x_conv1,
                    "x_conv2": x_conv2,
                    "x_conv3": x_conv3,
                    "x_conv4": x_conv4,
                }
            }
        )
        batch_dict.update(
            {
                "multi_scale_3d_strides": {
                    "x_conv1": 1,
                    "x_conv2": 2,
                    "x_conv3": 4,
                    "x_conv4": 8,
                }
            }
        )

        return batch_dict
