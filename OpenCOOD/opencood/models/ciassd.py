"""
CIA-SSD model for collaborative 3D object detection.

This module implements CIA-SSD (Collaborative Image-Aware Single Shot Multibox Detector)
for multi-agent cooperative 3D object detection using sparse voxel features.
"""

from torch import nn
import numpy as np
from typing import Dict, Any
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head


class CIASSD(nn.Module):
    """
    CIA-SSD model for collaborative 3D object detection.

    This model implements CIA-SSD (Collaborative Image-Aware Single Shot Multibox
    Detector) architecture combining sparse 3D convolutions, height compression,
    and spatial-semantic feature aggregation for efficient 3D object detection.

    Parameters
    ----------
    args : dict of str to Any
        Model configuration dictionary containing:
        - 'lidar_range': Detection range [x_min, y_min, z_min, x_max, y_max, z_max].
        - 'voxel_size': Voxel dimensions [vx, vy, vz] in meters.
        - 'mean_vfe': VFE configuration with 'num_point_features'.
        - 'spconv': Sparse convolution backbone config with 'num_features_in'.
        - 'map2bev': Height compression configuration.
        - 'ssfa': SSFA (Spatial-Semantic Feature Aggregation) module configuration.
        - 'head': Detection head configuration.

    Attributes
    ----------
    vfe : MeanVFE
        Mean voxel feature encoder module.
    spconv_block : VoxelBackBone8x
        3D sparse convolutional backbone with 8x downsampling.
    map_to_bev : HeightCompression
        Height compression module to convert 3D features to BEV representation.
    ssfa : SSFA
        Spatial-Semantic Feature Aggregation module.
    head : Head
        Detection head for predicting bounding boxes and classifications.
    """
    
    def __init__(self, args):
        super(CIASSD, self).__init__()
        lidar_range = np.array(args["lidar_range"])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) / np.array(args["voxel_size"])).astype(np.int64)
        self.vfe = MeanVFE(args["mean_vfe"], args["mean_vfe"]["num_point_features"])
        self.spconv_block = VoxelBackBone8x(args["spconv"], input_channels=args["spconv"]["num_features_in"], grid_size=grid_size)
        self.map_to_bev = HeightCompression(args["map2bev"])
        self.ssfa = SSFA(args["ssfa"])
        self.head = Head(**args["head"])

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass through CIA-SSD network.

        Parameters
        ----------
        batch_dict : dict of str to Any
            Input batch dictionary containing:
            - 'object_bbx_center': Object bounding box centers for determining batch size.
            - 'voxel_features': Voxel features from point cloud.
            - 'voxel_coords': Voxel coordinates.
            - 'voxel_num_points': Number of points per voxel.

        Returns
        -------
        dict of str to torch.Tensor
            Output dictionary with keys:
            - 'preds_dict_stage1': Stage 1 predictions containing bounding boxes
              and classification scores.
            - All keys from input batch_dict are preserved.
        """
        batch_dict["batch_size"] = batch_dict["object_bbx_center"].shape[0]
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        out = self.ssfa(batch_dict["processed_lidar"]["spatial_features"])
        out = self.head(out)
        batch_dict["preds_dict_stage1"] = out

        return batch_dict


if __name__ == "__main__":
    model = SSFA(None)
    print(model)
