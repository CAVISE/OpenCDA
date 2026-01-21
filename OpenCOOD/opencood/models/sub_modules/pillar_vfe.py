"""
Pillar Voxel Feature Encoder (VFE).

This module implements the Pillar Feature Network for encoding point features
within voxels/pillars for 3D object detection. Credits to OpenPCDet.
"""

from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayer(nn.Module):
    """
    Pillar Feature Network Layer.

    This layer processes point features within each pillar by applying linear
    transformation, batch normalization, ReLU activation, and max pooling.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    use_norm : bool, optional
        Whether to use batch normalization. Default is True.
    last_layer : bool, optional
        Whether this is the last layer in the network. Default is False.

    Attributes
    ----------
    last_vfe : bool
        Whether this is the last VFE layer.
    use_norm : bool
        Whether batch normalization is used.
    linear : nn.Linear
        Linear transformation layer.
    norm : nn.BatchNorm1d, optional
        Batch normalization layer.
    part : int
        Batch partition size for large inputs.
    """

    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True, last_layer: bool = False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PFN layer.

        Parameters
        ----------
        inputs : Tensor
            Input point features with shape (N_pillars, N_points, C).

        Returns
        -------
        Tensor
            If last_layer: Max-pooled features.
            Otherwise: Concatenated features.
        """
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part : (num_part + 1) * self.part]) for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False  # noqa: DC05
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True  # noqa: DC05
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    """
    Pillar Voxel Feature Encoder.

    Encodes raw point features within pillars/voxels into learned feature
    representations using a series of PFN layers.

    Parameters
    ----------
    model_cfg : dict of str to Any
        Model configuration dictionary containing:
        - 'use_norm': Whether to use batch normalization.
        - 'with_distance': Whether to include point distance features.
        - 'use_absolute_xyz': Whether to use absolute XYZ coordinates.
        - 'num_filters': List of output channel dimensions for PFN layers.
    num_point_features : int
        Number of raw point features (e.g., 4 for [x, y, z, intensity]).
    voxel_size : list of float
        Voxel size [voxel_x, voxel_y, voxel_z].
    point_cloud_range : list of float
        Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].

    Attributes
    ----------
    model_cfg : dict
        Model configuration.
    use_norm : bool
        Whether to use batch normalization.
    with_distance : bool
        Whether to include distance features.
    use_absolute_xyz : bool
        Whether to use absolute XYZ coordinates.
    num_filters : list of int
        Output channel dimensions for each PFN layer.
    pfn_layers : nn.ModuleList
        List of PFN layers.
    voxel_x : float
        Voxel size along X axis.
    voxel_y : float
        Voxel size along Y axis.
    voxel_z : float
        Voxel size along Z axis.
    x_offset : float
        X coordinate offset to voxel center.
    y_offset : float
        Y coordinate offset to voxel center.
    z_offset : float
        Z coordinate offset to voxel center.
    """

    def __init__(self, model_cfg: Dict[str, Any], num_point_features: int, voxel_size: List[float], point_cloud_range: List[float]):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg["use_norm"]
        self.with_distance = self.model_cfg["with_distance"]

        self.use_absolute_xyz = self.model_cfg["use_absolute_xyz"]
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg["num_filters"]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2)))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    @staticmethod
    def get_paddings_indicator(actual_num: torch.Tensor, max_num: int, axis: int = 0) -> torch.Tensor:
        """
        Generate padding mask for variable-length sequences.

        Parameters
        ----------
        actual_num : Tensor
            Actual number of valid elements with shape (N,).
        max_num : int
            Maximum number of elements.
        axis : int, optional
            Axis along which to generate mask. Default is 0.

        Returns
        -------
        Tensor
            Boolean mask indicating valid elements with shape (N, max_num).
        """
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode pillar features from raw point features.

        Parameters
        ----------
        batch_dict : dict of str to Tensor
            Batch dictionary containing:
            - 'voxel_features': Point features with shape (N_pillars, N_points, C).
            - 'voxel_num_points': Number of points per pillar with shape (N_pillars,).
            - 'voxel_coords': Pillar coordinates with shape (N_pillars, 4)
              in format [batch_idx, z_idx, y_idx, x_idx].

        Returns
        -------
        dict of str to Tensor
            Updated batch dictionary with:
            - 'pillar_features': Encoded pillar features with shape (N_pillars, C_out).
        """
        voxel_features, voxel_num_points, coords = batch_dict["voxel_features"], batch_dict["voxel_num_points"], batch_dict["voxel_coords"]
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict["pillar_features"] = features
        return batch_dict
