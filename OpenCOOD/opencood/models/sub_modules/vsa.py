"""
Voxel Set Abstraction module for 3D object detection.

This module implements voxel-based set abstraction for processing multi-scale
3D features in cooperative perception systems.
"""

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from opencood.pcdet_utils.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from opencood.pcdet_utils.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from opencood.utils import common_utils


def bilinear_interpolate_torch(im: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Perform bilinear interpolation on 2D feature map.

    Parameters
    ----------
    im : Tensor
        Feature map with shape (H, W, C).
    x : Tensor
        X coordinates with shape (N,).
    y : Tensor
        Y coordinates with shape (N,).

    Returns
    -------
    Tensor
        Interpolated features with shape (N, C).
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class VoxelSetAbstraction(nn.Module):
    """
    Voxel Set Abstraction module for multi-scale 3D feature aggregation.

    This module performs hierarchical feature aggregation from multi-scale voxel
    features, BEV features, and raw points using PointNet++ set abstraction layers.

    Parameters
    ----------
    model_cfg : dict
        Model configuration dictionary containing:
        - 'features_source': List of feature sources ['bev', 'raw_points', 'x_conv1', etc.].
        - 'sa_layer': Set abstraction layer configuration per source.
        - 'num_keypoints': Number of keypoints to sample.
        - 'num_out_features': Output feature dimension.
        - 'point_source': Source for sampling keypoints ('raw_points' or 'voxel_centers').
        - 'enlarge_selection_boxes': Whether to enlarge detection boxes for selection.
    voxel_size : list of float
        Voxel size [vx, vy, vz].
    point_cloud_range : list of float
        Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
    num_bev_features : int, optional
        Number of BEV feature channels.
    num_rawpoint_features : int, optional
        Number of raw point feature channels.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    model_cfg : dict
        Model configuration.
    voxel_size : list of float
        Voxel size for discretization.
    point_cloud_range : list of float
        Valid point cloud range.
    SA_layers : nn.ModuleList
        List of set abstraction layers for multi-scale features.
    SA_layer_names : list of str
        Names of set abstraction layers.
    downsample_times_map : dict
        Mapping from layer name to downsample factor.
    SA_rawpoints : StackSAModuleMSG, optional
        Set abstraction layer for raw points if used.
    vsa_point_feature_fusion : nn.Sequential
        Feature fusion module combining multi-scale features.
    num_point_features : int
        Output feature dimension.
    num_point_features_before_fusion : int
        Feature dimension before fusion.
    """

    def __init__(
        self,
        model_cfg: Dict[str, any],
        voxel_size: List[float],
        point_cloud_range: List[float],
        num_bev_features: Optional[int] = None,
        num_rawpoint_features: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg["sa_layer"]

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg["features_source"]:
            if src_name in ["bev", "raw_points"]:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name]["downsample_factor"]
            mlps = copy.copy(SA_cfg[src_name]["mlps"])
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name]["pool_radius"],
                nsamples=SA_cfg[src_name]["n_sample"],
                mlps=mlps,
                use_xyz=True,
                pool_method="max_pool",
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if "bev" in self.model_cfg["features_source"]:
            c_bev = num_bev_features
            c_in += c_bev

        if "raw_points" in self.model_cfg["features_source"]:
            mlps = copy.copy(SA_cfg["raw_points"]["mlps"])
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]

            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg["raw_points"]["pool_radius"], nsamples=SA_cfg["raw_points"]["n_sample"], mlps=mlps, use_xyz=True, pool_method="max_pool"
            )
            c_in += sum([x[-1] for x in mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg["num_out_features"], bias=False),
            nn.BatchNorm1d(self.model_cfg["num_out_features"]),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg["num_out_features"]
        self.num_point_features_before_fusion = c_in  # noqa: DC05

    def interpolate_from_bev_features(self, keypoints: torch.Tensor, bev_features: torch.Tensor, batch_size: int, bev_stride: int) -> torch.Tensor:
        """
        Interpolate BEV features at keypoint locations.

        Parameters
        ----------
        keypoints : Tensor
            Keypoint coordinates with shape (B, N, 3).
        bev_features : Tensor
            BEV features with shape (B, C, H, W).
        batch_size : int
            Batch size.
        bev_stride : int
            Stride of BEV feature map.

        Returns
        -------
        Tensor
            Interpolated features with shape (B, N, C).
        """
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_sampled_points(self, batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Sample keypoints using Furthest Point Sampling (FPS).

        Parameters
        ----------
        batch_dict : dict of str to Tensor
            Batch dictionary containing:
            - 'batch_size': Batch size.
            - 'origin_lidar' or 'voxel_coords': Source points.

        Returns
        -------
        Tensor
            Sampled keypoints with shape (B, num_keypoints, 4).
        """
        batch_size = batch_dict["batch_size"]
        if self.model_cfg["point_source"] == "raw_points":
            src_points = batch_dict["origin_lidar"][:, 1:]
            batch_indices = batch_dict["origin_lidar"][:, 0].long()
        elif self.model_cfg["point_source"] == "voxel_centers":
            src_points = common_utils.get_voxel_centers(
                batch_dict["voxel_coords"][:, 1:4], downsample_times=1, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict["voxel_coords"][:, 0].long()
        else:
            raise NotImplementedError

        keypoints_batch = torch.randn((batch_size, self.model_cfg["num_keypoints"], 4), device=src_points.device)
        keypoints_batch[..., 0] = keypoints_batch[..., 0] * 140
        keypoints_batch[..., 1] = keypoints_batch[..., 0] * 40
        # points with height flag 10 are padding/invalid, for later filtering
        keypoints_batch[..., 2] = 10.0
        for bs_idx in range(batch_size):
            bs_mask = batch_indices == bs_idx
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            # sample points with FPS
            # some cropped pcd may have very few points, select various number
            # of points to ensure similar sample density
            # 50000 is approximately the number of points in one full pcd
            num_kpts = int(self.model_cfg["num_keypoints"] * sampled_points.shape[1] / 50000) + 1
            num_kpts = min(num_kpts, self.model_cfg["num_keypoints"])
            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(sampled_points[:, :, 0:3].contiguous(), num_kpts).long()

            if sampled_points.shape[1] < num_kpts:
                empty_num = num_kpts - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            keypoints_batch[bs_idx, : len(keypoints[0]), :] = keypoints

        # keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints_batch

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-scale feature aggregation.

        Parameters
        ----------
        batch_dict : dict of str to Tensor
            Batch dictionary containing:
            - 'batch_size': Number of samples.
            - 'origin_lidar' or 'voxel_coords': Source points.
            - 'spatial_features': BEV features (optional).
            - 'spatial_features_stride': BEV stride (optional).
            - 'multi_scale_3d_features': Multi-scale voxel features.
            - 'det_boxes': Detection boxes for keypoint filtering (optional).

        Returns
        -------
        dict of str to Tensor
            Updated batch dictionary with:
            - 'point_features': List of point features per sample.
            - 'point_coords': List of point coordinates per sample.
            - 'point_features_before_fusion': Features before fusion.
        """
        keypoints = self.get_sampled_points(batch_dict)  # BxNx4
        kpt_mask1 = torch.logical_and(keypoints[..., 2] > -2.8, keypoints[..., 2] < 1.0)
        kpt_mask2 = None
        # Only select the points that are in the predicted bounding boxes
        if "det_boxes" in batch_dict:
            dets_list = batch_dict["det_boxes"]
            max_len = max([len(dets) for dets in dets_list])
            boxes = torch.zeros((len(dets_list), max_len, 7), dtype=dets_list[0].dtype, device=dets_list[0].device)
            for i, dets in enumerate(dets_list):
                if len(dets) == 0:
                    continue
                cur_dets = dets.clone()
                if self.model_cfg["enlarge_selection_boxes"]:
                    cur_dets[:, 3:6] += 0.5
                boxes[i, : len(dets)] = cur_dets
            # mask out some keypoints to spare the GPU storage
            kpt_mask2 = points_in_boxes_gpu(keypoints[..., :3], boxes) >= 0

        kpt_mask = torch.logical_and(kpt_mask1, kpt_mask2) if kpt_mask2 is not None else kpt_mask1
        # Ensure there are more than 2 points are selected to satisfy the
        # condition of batch norm in the FC layers of feature fusion module
        if (kpt_mask).sum() < 2:
            kpt_mask[0, torch.randint(0, 1024, (2,))] = True

        point_features_list = []
        if "bev" in self.model_cfg["features_source"]:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints[..., :3], batch_dict["spatial_features"], batch_dict["batch_size"], bev_stride=batch_dict["spatial_features_stride"]
            )
            point_features_list.append(point_bev_features[kpt_mask])

        batch_size, _, _ = keypoints.shape

        new_xyz = keypoints[kpt_mask]
        new_xyz_batch_cnt = torch.tensor([(mask).sum() for mask in kpt_mask], device=new_xyz.device).int()

        if "raw_points" in self.model_cfg["features_source"]:
            raw_points = batch_dict["origin_lidar"]
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            indices = raw_points[:, 0].long()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (indices == bs_idx).sum()
            point_features = None

            _, pooled_features = self.SA_rawpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz[:, :3].contiguous(),
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
            )
            point_features_list.append(pooled_features)

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict["multi_scale_3d_features"][src_name].indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range,
            )
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            _, pooled_features = self.SA_layers[k](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz[:, :3].contiguous(),
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=batch_dict["multi_scale_3d_features"][src_name].features.contiguous(),
            )

            point_features_list.append(pooled_features)

        point_features = torch.cat(point_features_list, dim=1)

        batch_dict["point_features_before_fusion"] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        cur_idx = 0
        batch_dict["point_features"] = []
        batch_dict["point_coords"] = []
        for num in new_xyz_batch_cnt:
            batch_dict["point_features"].append(point_features[cur_idx : cur_idx + num])
            batch_dict["point_coords"].append(new_xyz[cur_idx : cur_idx + num])
            cur_idx += num

        return batch_dict
