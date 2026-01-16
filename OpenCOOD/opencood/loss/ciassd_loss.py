"""
Loss functions for CIASSD (Collaborative Intermediate Aggregation Single Shot Detector).

This module implements the loss function for the CIASSD architecture.
"""

import torch
import torch.nn as nn

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import aligned_boxes_iou3d_gpu
from typing import Dict, Any, Optional, Tuple
from torch import Tensor


class CiassdLoss(nn.Module):
    def __init__(self, args: Dict[str, Any]):
        super(CiassdLoss, self).__init__()
        self.pos_cls_weight = args["pos_cls_weight"]
        self.encode_rad_error_by_sin = args["encode_rad_error_by_sin"]
        self.cls = args["cls"]
        self.reg = args["reg"]
        self.iou = args["iou"]
        self.dir = args["dir"]
        self.loss_dict = {}
        ##
        self.num_cls = 2
        self.box_codesize = 7

    def forward(self, output_dict: Dict[str, Any], label_dict: Dict[str, Any]) -> Tensor:
        """
        Compute total loss from model predictions and ground truth labels.

        Parameters
        ----------
        output_dict : dict
            Model predictions containing:
                - preds_dict_stage1 : dict
                    - cls_preds : Tensor (B, C, H, W)
                        Classification predictions.
                    - box_preds : Tensor (B, 7, H, W)
                        Bounding box regression predictions.
                    - dir_cls_preds : Tensor (B, 2, H, W)
                        Direction classification predictions.
                    - iou_preds : Tensor (B, 1, H, W)
                        IoU predictions.
                - anchor_box : Tensor
                    Anchor boxes.
                - record_len : Tensor, optional
                    Number of CAVs per sample (for batch size calculation).
        label_dict : dict
            Ground truth labels containing:
                - stage1 : dict
                    - pos_equal_one : Tensor
                        Positive sample mask.
                    - neg_equal_one : Tensor
                        Negative sample mask.
                    - targets : Tensor
                        Regression targets.

        Returns
        -------
        loss : Tensor
            Total loss (scalar).
        """
        preds_dict = output_dict["preds_dict_stage1"]
        target_dict = label_dict["stage1"]
        if "record_len" in output_dict:
            batch_size = int(output_dict["record_len"].sum())
        else:
            batch_size = output_dict["batch_size"]

        # ########
        # pred = torch.sigmoid(preds_dict['cls_preds'][0]).sum(dim=0).cpu().detach().numpy()
        # tagt_pos = target_dict['pos_equal_one'][0].sum(dim=-1).cpu().detach().numpy()
        # tagt_neg = target_dict['neg_equal_one'][0].sum(dim=-1).cpu().detach().numpy()
        # import matplotlib.pyplot as plt
        #
        # fig = plt.figure(figsize=(18, 8))
        # ax1 = fig.add_subplot(3, 1, 1)
        # ax1.imshow(pred, cmap='cool')
        # ax2 = fig.add_subplot(3, 1, 2)
        # ax2.imshow(tagt_pos, cmap='cool')
        # ax3 = fig.add_subplot(3, 1, 3)
        # ax3.imshow(tagt_neg, cmap='cool')
        # plt.show()
        # plt.close()
        # #########

        cls_labls = target_dict["pos_equal_one"].view(batch_size, -1, self.num_cls - 1)
        positives = cls_labls > 0
        negatives = target_dict["neg_equal_one"].view(batch_size, -1, self.num_cls - 1) > 0
        cared = torch.logical_or(positives, negatives)
        cls_labls = cls_labls * cared.type_as(cls_labls)
        # num_normalizer = cared.sum(1, keepdim=True)
        pos_normalizer = positives.sum(1, keepdim=True).float()

        # cls loss
        cls_preds = preds_dict["cls_preds"].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_cls - 1)
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
        cls_loss_reduced = cls_loss.sum() * self.cls["weight"] / batch_size

        # reg loss
        reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = preds_dict["box_preds"].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.box_codesize)
        reg_targets = target_dict["targets"].view(batch_size, -1, self.box_codesize)
        if self.encode_rad_error_by_sin:
            reg_preds, reg_targets = add_sin_difference(reg_preds, reg_targets)
        reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg["sigma"])
        reg_loss_reduced = reg_loss.sum() * self.reg["weight"] / batch_size

        # dir loss
        dir_targets = get_direction_target(reg_targets, output_dict["anchor_box"])
        dir_logits = preds_dict["dir_cls_preds"].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        dir_weights = (cls_labls > 0).type_as(dir_logits).view(batch_size, -1)
        dir_weights /= torch.clamp(dir_weights.sum(-1, keepdim=True), min=1.0)  # [8, 70400], averaged in sample.
        dir_loss = softmax_cross_entropy_with_logits(dir_logits.view(-1, self.num_cls), dir_targets.view(-1, self.num_cls))
        dir_loss = dir_loss.view(dir_weights.shape) * dir_weights
        dir_loss_reduced = dir_loss.sum() * self.dir["weight"] / batch_size

        # iou loss
        iou_preds = preds_dict["iou_preds"].permute(0, 2, 3, 1).contiguous()
        pos_pred_mask = reg_weights.squeeze(dim=-1) > 0  # (4, 70400)
        iou_pos_preds = iou_preds.view(batch_size, -1)[pos_pred_mask]
        boxes3d_pred = VoxelPostprocessor.delta_to_boxes3d(
            preds_dict["box_preds"].permute(0, 2, 3, 1).contiguous().detach(), output_dict["anchor_box"], False
        )[pos_pred_mask]
        boxes3d_tgt = VoxelPostprocessor.delta_to_boxes3d(target_dict["targets"], output_dict["anchor_box"], False)[pos_pred_mask]
        iou_weights = reg_weights[pos_pred_mask].view(-1)
        iou_pos_targets = (
            aligned_boxes_iou3d_gpu(boxes3d_pred.float()[:, [0, 1, 2, 5, 4, 3, 6]], boxes3d_tgt.float()[:, [0, 1, 2, 5, 4, 3, 6]]).detach().squeeze()
        )
        iou_pos_targets = 2 * iou_pos_targets.view(-1) - 1
        iou_loss = weighted_smooth_l1_loss(iou_pos_preds, iou_pos_targets, weights=iou_weights, sigma=self.iou["sigma"])

        iou_loss_reduced = iou_loss.sum() * self.iou["weight"] / batch_size

        loss = cls_loss_reduced + reg_loss_reduced + dir_loss_reduced + iou_loss_reduced

        self.loss_dict.update(
            {
                "total_loss": loss,
                "cls_loss": cls_loss_reduced,
                "reg_loss": reg_loss_reduced,
                "dir_loss": dir_loss_reduced,
                "iou_loss": iou_loss_reduced,
            }
        )

        return loss

    def logging(self, 
               epoch: int, 
               batch_id: int, 
               batch_len: int, 
               writer, 
               pbar: Optional[Any] = None) -> None:
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict["total_loss"]
        reg_loss = self.loss_dict["reg_loss"]
        cls_loss = self.loss_dict["cls_loss"]
        dir_loss = self.loss_dict["dir_loss"]
        iou_loss = self.loss_dict["iou_loss"]
        if (batch_id + 1) % 10 == 0:
            print(
                "[epoch %d][%d/%d], || Loss: %.4f || Cls: %.4f"
                " || Loc: %.4f || Dir: %.4f || Iou: %.4f"
                % (epoch, batch_id + 1, batch_len, total_loss.item(), cls_loss.item(), reg_loss.item(), dir_loss.item(), iou_loss.item())
            )

        writer.add_scalar("Regression_loss", reg_loss.item(), epoch * batch_len + batch_id)
        writer.add_scalar("Confidence_loss", cls_loss.item(), epoch * batch_len + batch_id)
        writer.add_scalar("Direction_loss", dir_loss.item(), epoch * batch_len + batch_id)
        writer.add_scalar("Iou_loss", iou_loss.item(), epoch * batch_len + batch_id)


def add_sin_difference(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Encode rotation angle error using sine-cosine difference.
    
    Converts rotation angle regression to sin(pred)*cos(gt) and cos(pred)*sin(gt)
    for better gradient flow near angle boundaries (e.g., 0° and 360°).

    Parameters
    ----------
    boxes1 : Tensor
        Predicted bounding boxes (..., 7) where last dim is rotation angle.
    boxes2 : Tensor
        Ground truth bounding boxes (..., 7) where last dim is rotation angle.

    Returns
    -------
    res_boxes1 : Tensor
        Predicted boxes with angle encoded as sin(pred)*cos(gt).
    res_boxes2 : Tensor
        GT boxes with angle encoded as cos(pred)*sin(gt).
    """
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])  # ry -> sin(pred_ry)*cos(gt_ry)
    rad_gt_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])  # ry -> cos(pred_ry)*sin(gt_ry)
    res_boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    res_boxes2 = torch.cat([boxes2[..., :-1], rad_gt_encoding], dim=-1)
    return res_boxes1, res_boxes2


def get_direction_target(reg_targets: Tensor, 
                        anchors: Tensor, 
                        one_hot: bool = True, 
                        dir_offset: float = 0.0) -> Tensor:
    """
    Generate targets for bounding box direction classification.

    Parameters
    ----------
    anchors: torch.Tensor
        shape as (H*W*2, 7) or (H, W, 2, 7)
    reg_targets: torch.Tensor
        shape as (B, H*W*2, 7)

    Returns
    -------
    dir_cls_targets : torch.Tensor
        [batch_size, w*h*num_anchor_per_pos, 2]
    """
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]  # [4, 70400]
    dir_cls_targets = ((rot_gt - dir_offset) > 0).long()  # [4, 70400]
    if one_hot:
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets


def one_hot_f(tensor: Tensor, 
              depth: int, 
              dim: int = -1, 
              on_value: float = 1.0, 
              dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Convert integer tensor to one-hot encoding.

    Parameters
    ----------
    tensor : Tensor
        Input tensor with integer class indices.
    depth : int
        Number of classes (one-hot vector length).
    dim : int, optional
        Dimension along which to add one-hot encoding. Default is -1.
    on_value : float, optional
        Value for the "on" position. Default is 1.0.
    dtype : torch.dtype, optional
        Output tensor dtype. Default is torch.float32.

    Returns
    -------
    tensor_onehot : Tensor
        One-hot encoded tensor with shape (*tensor.shape, depth).
    """
    tensor_onehot = torch.zeros(*list(tensor.shape), depth, dtype=dtype, device=tensor.device)  # [4, 70400, 2]
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)  # [4, 70400, 2]
    return tensor_onehot


def sigmoid_focal_loss(preds: Tensor, 
                       targets: Tensor, 
                       weights: Optional[Tensor] = None, 
                       **kwargs) -> Tensor:
    """
    Compute focal loss with sigmoid activation.
    
    Focal loss down-weights easy examples and focuses on hard negatives,
    reducing class imbalance impact.

    Parameters
    ----------
    preds : Tensor
        Raw logits (before sigmoid).
    targets : Tensor
        Binary targets (0 or 1).
    weights : Tensor, optional
        Per-sample weights.
    **kwargs
        Must contain 'gamma' (focusing parameter) and 'alpha' (balancing factor).

    Returns
    -------
    loss : Tensor
        Focal loss per sample.
    """
    assert "gamma" in kwargs and "alpha" in kwargs
    # sigmoid cross entropy with logits
    # more details: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    per_entry_cross_ent = torch.clamp(preds, min=0) - preds * targets.type_as(preds)
    per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(preds)))
    # focal loss
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, kwargs["gamma"])
    alpha_weight_factor = targets * kwargs["alpha"] + (1 - targets) * (1 - kwargs["alpha"])

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss


def softmax_cross_entropy_with_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Compute cross-entropy loss with softmax.

    Parameters
    ----------
    logits : Tensor
        Raw logits (..., num_classes).
    labels : Tensor
        One-hot encoded labels (..., num_classes).

    Returns
    -------
    loss : Tensor
        Cross-entropy loss per sample.
    """
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


def weighted_smooth_l1_loss(preds: Tensor, 
                            targets: Tensor, 
                            sigma: float = 3.0, 
                            weights: Optional[Tensor] = None) -> Tensor:
    """
    Compute weighted smooth L1 loss (Huber loss variant).
    
    Smooth L1 is less sensitive to outliers than L2 loss, with quadratic
    behavior near zero and linear behavior for large errors.

    Parameters
    ----------
    preds : Tensor
        Predicted values.
    targets : Tensor
        Ground truth values.
    sigma : float, optional
        Smoothness parameter. Higher sigma = more like L2 loss. Default is 3.0.
    weights : Tensor, optional
        Per-sample weights.

    Returns
    -------
    loss : Tensor
        Smooth L1 loss per sample.
    """
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma**2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + (abs_diff - 0.5 / (sigma**2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        loss *= weights
    return loss
