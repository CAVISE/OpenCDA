"""
FPV-RCNN model for two-stage 3D object detection.

This module implements FPV-RCNN (Frustum Point-Voxel R-CNN), a two-stage detector
combining voxel-based and point-based representations for accurate 3D object detection.
"""

from torch import nn
import numpy as np
from typing import Dict, Any

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head
from opencood.models.sub_modules.vsa import VoxelSetAbstraction
from opencood.models.sub_modules.roi_head import RoIHead
from opencood.models.sub_modules.matcher import Matcher
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import FpvrcnnPostprocessor


class FPVRCNN(nn.Module):
    """
    FPVRCNN (Frustum Point-Voxel R-CNN) model for 3D object detection.

    Parameters
    ----------
    args : dict
        Model configuration containing:
            - lidar_range : list of float
                Detection range [x_min, y_min, z_min, x_max, y_max, z_max].
            - voxel_size : list of float
                Voxel dimensions [vx, vy, vz].
            - mean_vfe : dict
                VFE configuration.
            - spconv : dict
                Sparse convolution backbone config.
            - map2bev : dict
                Height compression config.
            - ssfa : dict
                SSFA module config.
            - head : dict
                Stage 1 detection head config.
            - post_processer : dict
                Postprocessor config (NMS, score threshold).
            - vsa : dict
                Voxel Set Abstraction config.
            - matcher : dict
                Proposal-GT matching config.
            - roi_head : dict
                Stage 2 RoI head config.
            - activate_stage2 : bool
                Whether to train/use stage 2 refinement.

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
        Stage 1 detection head for initial proposals.
    post_processor : FpvrcnnPostprocessor
        Post-processor for NMS and filtering of stage 1 predictions.
    vsa : VoxelSetAbstraction
        Voxel Set Abstraction module for extracting point features from proposals.
    matcher : Matcher
        Module for matching proposals with ground truth boxes.
    roi_head : RoIHead
        Stage 2 RoI head for refining initial proposals.
    train_stage2 : bool
        Flag indicating whether stage 2 refinement is active.
    """

    def __init__(self, args: Dict[str, Any]):
        super(FPVRCNN, self).__init__()
        lidar_range = np.array(args["lidar_range"])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) / np.array(args["voxel_size"])).astype(np.int64)
        self.vfe = MeanVFE(args["mean_vfe"], args["mean_vfe"]["num_point_features"])
        self.spconv_block = VoxelBackBone8x(args["spconv"], input_channels=args["spconv"]["num_features_in"], grid_size=grid_size)
        self.map_to_bev = HeightCompression(args["map2bev"])
        self.ssfa = SSFA(args["ssfa"])
        self.head = Head(**args["head"])
        self.post_processor = FpvrcnnPostprocessor(args["post_processer"], train=True)
        self.vsa = VoxelSetAbstraction(args["vsa"], args["voxel_size"], args["lidar_range"], num_bev_features=128, num_rawpoint_features=3)
        self.matcher = Matcher(args["matcher"], args["lidar_range"])
        self.roi_head = RoIHead(args["roi_head"])
        self.train_stage2 = args["activate_stage2"]

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass through FPV-RCNN two-stage detector.

        Parameters
        ----------
        batch_dict : dict of str to Any
            Input batch dictionary containing:
            - 'processed_lidar': Dictionary with 'voxel_features', 'voxel_coords',
              and 'voxel_num_points'.
            - 'record_len': Tensor indicating number of samples per batch.
            - 'object_bbx_center': Ground truth bounding box centers (training only).

        Returns
        -------
        dict of str to torch.Tensor
            Output dictionary with keys:
            - 'preds_dict_stage1': Stage 1 predictions with bounding boxes and scores.
            - 'det_boxes': List of detected bounding boxes after NMS.
            - 'det_scores': List of detection scores.
            - 'preds_dict_stage2': Stage 2 refined predictions (if stage 2 is active).
            - Additional intermediate features from VSA and matcher modules.
        """
        voxel_features = batch_dict["processed_lidar"]["voxel_features"]
        voxel_coords = batch_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = batch_dict["processed_lidar"]["voxel_num_points"]

        # save memory
        batch_dict.pop("processed_lidar")
        batch_dict.update(
            {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": voxel_num_points,
                "batch_size": int(batch_dict["record_len"].sum()),
            }
        )

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)

        out = self.ssfa(batch_dict["spatial_features"])
        batch_dict["preds_dict_stage1"] = self.head(out)

        data_dict, output_dict = {}, {}
        data_dict["ego"], output_dict["ego"] = batch_dict, batch_dict

        pred_box3d_list, scores_list = self.post_processor.post_process(data_dict, output_dict, stage1=True)
        batch_dict["det_boxes"] = pred_box3d_list
        batch_dict["det_scores"] = scores_list

        if pred_box3d_list is not None and self.train_stage2:
            batch_dict = self.vsa(batch_dict)
            batch_dict = self.matcher(batch_dict)
            batch_dict = self.roi_head(batch_dict)

        return batch_dict


if __name__ == "__main__":
    model = SSFA(None)
    print(model)
