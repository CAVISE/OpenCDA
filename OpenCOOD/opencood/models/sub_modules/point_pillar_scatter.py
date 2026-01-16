"""
PointPillar Scatter Module for BEV Feature Map Generation.

This module scatters pillar features into a dense 2D Bird's Eye View (BEV)
spatial feature map for downstream processing.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    """
    Scatter pillar features to dense BEV spatial feature map.

    This module converts sparse pillar features with their coordinates
    into a dense 2D spatial representation suitable for convolutional processing.

    Parameters
    ----------
    model_cfg : dict of str to Any
        Model configuration dictionary containing:
        - 'num_features': Number of pillar feature channels.
        - 'grid_size': Voxel grid size [X, Y, Z].

    Attributes
    ----------
    model_cfg : dict
        Model configuration.
    num_bev_features : int
        Number of feature channels in output BEV map.
    nx : int
        Grid size along X axis.
    ny : int
        Grid size along Y axis.
    nz : int
        Grid size along Z axis.
    """
        
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg["num_features"]
        self.nx, self.ny, self.nz = model_cfg["grid_size"]
        assert self.nz == 1

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Scatter pillar features into dense BEV spatial feature map.

        Parameters
        ----------
        batch_dict : dict of str to Tensor
            Batch dictionary containing:
            - 'pillar_features': Pillar features with shape (N_pillars, C).
            - 'voxel_coords': Pillar coordinates with shape (N_pillars, 4)
              in format [batch_idx, z_idx, y_idx, x_idx].

        Returns
        -------
        dict of str to Tensor
            Updated batch dictionary with:
            - 'spatial_features': Dense BEV features with shape (B, C, H, W)
              where H=ny and W=nx.
        """
        pillar_features, coords = batch_dict["pillar_features"], batch_dict["voxel_coords"]
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features, self.nz * self.nx * self.ny, dtype=pillar_features.dtype, device=pillar_features.device
            )

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]

            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)

            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict["spatial_features"] = batch_spatial_features

        return batch_dict
