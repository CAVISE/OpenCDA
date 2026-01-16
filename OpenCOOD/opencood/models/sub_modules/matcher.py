"""
Bounding Box Matcher and Fusion Module.

This module implements clustering and fusion of predicted bounding boxes
from multiple agents based on IoU overlap and confidence scores.
"""

from typing import Dict, List, Tuple, Any
import torch
from torch import nn, Tensor

from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

pi = 3.141592653


def limit_period(
    val: Tensor,
    offset: float = 0.5,
    period: float = 2 * pi
) -> Tensor:
    """
    Limit angles to a specific period range.

    Parameters
    ----------
    val : Tensor
        Input angle values.
    offset : float, optional
        Offset for period wrapping. Default is 0.5.
    period : float, optional
        Period length (e.g., 2*pi for full rotation). Default is 2*pi.

    Returns
    -------
    Tensor
        Angle values wrapped to the period range.
    """
    return val - torch.floor(val / period + offset) * period


class Matcher(nn.Module):
    """
    Bounding box matcher and fusion module for multi-agent detection.

    This module clusters predicted bounding boxes based on IoU overlap
    and fuses boxes within each cluster using score-weighted averaging.
    Implements Algorithm 1: BBox matching with scores.

    Parameters
    ----------
    cfg : dict of str to Any
        Configuration dictionary.
    pc_range : list of float
        Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].

    Attributes
    ----------
    pc_range : list of float
        Point cloud range.
    """

    def __init__(self, cfg: Dict[str, Any], pc_range: List[float]) -> None:
        super(Matcher, self).__init__()
        self.pc_range = pc_range

    @torch.no_grad()
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform box clustering and fusion.

        Parameters
        ----------
        data_dict : dict of str to Any
            Data dictionary containing:
            - 'det_boxes': List of detected boxes from all agents.
            - 'det_scores': List of detection scores from all agents.
            - 'record_len': Number of agents per scene.

        Returns
        -------
        dict of str to Any
            Updated data dictionary with:
            - 'boxes_fused': List of fused boxes per scene.
            - 'scores_fused': List of fused scores per scene.
        """
        clusters, scores = self.clustering(data_dict)
        data_dict["boxes_fused"], data_dict["scores_fused"] = self.cluster_fusion(clusters, scores)
        self.merge_keypoints(data_dict)
        return data_dict

    def clustering(
        self,
        data_dict: Dict[str, Any]
    ) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """
        Cluster predicted boxes based on IoU overlap.

        Boxes with IoU > 0.1 are assigned to the same cluster, representing
        detections of the same object from different agents.

        Parameters
        ----------
        data_dict : dict of str to Any
            Data dictionary containing detection results.

        Returns
        -------
        clusters_batch : list of list of Tensor
            Clustered boxes for each scene. Each cluster contains boxes
            that likely correspond to the same object.
        scores_batch : list of list of Tensor
            Corresponding scores for each cluster.
        """
        clusters_batch = []
        scores_batch = []
        record_len = [int(length) for length in data_dict["record_len"]]
        for i, length in enumerate(record_len):
            cur_boxes_list = data_dict["det_boxes"][sum(record_len[:i]) : sum(record_len[:i]) + length]
            cur_scores_list = data_dict["det_scores"][sum(record_len[:i]) : sum(record_len[:i]) + length]
            cur_boxes_list = [b for b in cur_boxes_list if len(b) > 0]
            cur_scores_list = [s for s in cur_scores_list if len(s) > 0]
            if len(cur_scores_list) == 0:
                clusters_batch.append([torch.Tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.57]).to(torch.device("cuda:0")).view(1, 7)])
                scores_batch.append([torch.Tensor([0.01]).to(torch.device("cuda:0")).view(-1)])
                continue

            pred_boxes_cat = torch.cat(cur_boxes_list, dim=0)
            pred_boxes_cat[:, -1] = limit_period(pred_boxes_cat[:, -1])
            pred_scores_cat = torch.cat(cur_scores_list, dim=0)

            ious = boxes_iou3d_gpu(pred_boxes_cat, pred_boxes_cat)
            cluster_indices = torch.zeros(len(ious)).int()  # gt assignments of preds
            cur_cluster_id = 1
            while torch.any(cluster_indices == 0):
                cur_idx = torch.where(cluster_indices == 0)[0][0]  # find the idx of the first pred which is not assigned yet
                cluster_indices[torch.where(ious[cur_idx] > 0.1)[0]] = cur_cluster_id
                cur_cluster_id += 1
            clusters = []
            scores = []
            for j in range(1, cur_cluster_id):
                clusters.append(pred_boxes_cat[cluster_indices == j])
                scores.append(pred_scores_cat[cluster_indices == j])
            clusters_batch.append(clusters)
            scores_batch.append(scores)

        return clusters_batch, scores_batch

    def cluster_fusion(
        self,
        clusters: List[List[Tensor]],
        scores: List[List[Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Fuse boxes within each cluster using score-weighted averaging.

        Parameters
        ----------
        clusters : list of list of Tensor
            Clustered boxes for each scene.
        scores : list of list of Tensor
            Corresponding detection scores.

        Returns
        -------
        boxes_fused : list of Tensor
            Fused boxes for each scene with shape (N_clusters, 7).
        scores_fused : list of Tensor
            Fused confidence scores for each scene with shape (N_clusters,).
        """
        boxes_fused = []
        scores_fused = []
        for cl, sl in zip(clusters, scores):
            for c, s in zip(cl, sl):
                # reverse direction for non-dominant direction of boxes
                dirs = c[:, -1]
                max_score_idx = torch.argmax(s)
                dirs_diff = torch.abs(dirs - dirs[max_score_idx].item())
                lt_pi = (dirs_diff > pi).int()
                dirs_diff = dirs_diff * (1 - lt_pi) + (2 * pi - dirs_diff) * lt_pi
                score_lt_half_pi = s[dirs_diff > pi / 2].sum()  # larger than
                score_set_half_pi = s[dirs_diff <= pi / 2].sum()  # small equal than
                # select larger scored direction as final direction
                if score_lt_half_pi <= score_set_half_pi:
                    dirs[dirs_diff > pi / 2] += pi
                else:
                    dirs[dirs_diff <= pi / 2] += pi
                dirs = limit_period(dirs)
                s_normalized = s / s.sum()
                sint = torch.sin(dirs) * s_normalized
                cost = torch.cos(dirs) * s_normalized
                theta = torch.atan2(sint.sum(), cost.sum()).view(
                    1,
                )
                center_dim = c[:, :-1] * s_normalized[:, None]
                boxes_fused.append(torch.cat([center_dim.sum(dim=0), theta]))
                s_sorted = torch.sort(s, descending=True).values
                s_fused = 0
                for i, ss in enumerate(s_sorted):
                    s_fused += ss ** (i + 1)
                s_fused = torch.tensor([min(s_fused, 1.0)], device=s.device)
                scores_fused.append(s_fused)

        assert len(boxes_fused) > 0
        boxes_fused = torch.stack(boxes_fused, dim=0)
        len_records = [len(c) for c in clusters]
        boxes_fused = [boxes_fused[sum(len_records[:i]) : sum(len_records[:i]) + length] for i, length in enumerate(len_records)]
        scores_fused = torch.stack(scores_fused, dim=0)
        scores_fused = [scores_fused[sum(len_records[:i]) : sum(len_records[:i]) + length] for i, length in enumerate(len_records)]

        return boxes_fused, scores_fused

    def merge_keypoints(self, data_dict: Dict[str, Any]) -> None:
        """
       Merge keypoint features and coordinates across samples.
        
        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing:
            
            - point_features : list
                List of point features.
            - point_coords : list
                List of point coordinates.
            - record_len : list of int
                List of integers indicating number of points per sample.
        
        Notes
        -----
        Modifies data_dict in-place to update:
        """
        # merge keypoints
        kpts_feat_out = []
        kpts_coor_out = []
        keypoints_features = data_dict["point_features"]
        keypoints_coords = data_dict["point_coords"]
        idx = 0
        for length in data_dict["record_len"]:
            kpts_coor_out.append(torch.cat(keypoints_coords[idx : length + idx], dim=0))
            kpts_feat_out.append(torch.cat(keypoints_features[idx : length + idx], dim=0))
            idx += length
        data_dict["point_features"] = kpts_feat_out
        data_dict["point_coords"] = kpts_coor_out
