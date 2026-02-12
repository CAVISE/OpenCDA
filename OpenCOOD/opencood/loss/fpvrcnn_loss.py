"""
Loss functions for FPVRCNN (Frustum Point Voxel R-CNN) 3D object detection.

This module implements the loss function for the FPVRCNN architecture.
"""

import torch
from torch import nn, Tensor
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
from opencood.loss.ciassd_loss import CiassdLoss, weighted_smooth_l1_loss


class FpvrcnnLoss(nn.Module):
    """
    FPVRCNN (Frustum Point Voxel R-CNN) loss function.

    This loss combines the first stage detection loss (Ciassd) with the second
    stage refinement loss for 3D object detection.

    Parameters
    ----------
    args : dict
        Configuration dictionary containing:
        - stage1 : dict
            Configuration for first stage (Ciassd) loss
        - stage2 : dict
            Configuration for second stage loss with keys:
            - cls : dict
                Classification loss config
            - reg : dict
                Regression loss config
            - iou : dict
                IoU loss config

    Attributes
    ----------
    ciassd_loss : CiassdLoss
        First stage detection loss module.
    cls : dict
        Dictionary containing classification loss configuration.
    reg : dict
        Dictionary containing regression loss configuration.
    iou : dict
        Dictionary containing IoU loss configuration.
    loss_dict : dict
        Dictionary to store all loss components.
    """

    def __init__(self, args: Dict):
        super(FpvrcnnLoss, self).__init__()
        self.ciassd_loss = CiassdLoss(args["stage1"])
        self.cls: Dict[str, float] = args["stage2"]["cls"]
        self.reg: Dict[str, float] = args["stage2"]["reg"]
        self.iou: Dict[str, float] = args["stage2"]["iou"]
        self.loss_dict: Dict[str, Tensor] = {}

    def forward(self, output_dict: Dict[str, Any], label_dict: Dict[str, Any]) -> Tensor:
        """
        Forward pass for FPVRCNN loss computation.

        Parameters
        ----------
        output_dict : dict
            Dictionary containing model outputs with keys:
            - preds_dict_stage1 : dict
                First stage predictions
            - fvprcnn_out : dict
                Second stage predictions with keys:
                - rcnn_cls : torch.Tensor
                    Classification logits
                - rcnn_iou : torch.Tensor
                    IoU predictions
                - rcnn_reg : torch.Tensor
                    Regression predictions
            - rcnn_label_dict : dict
                Dictionary with target values for second stage
            - record_len : torch.Tensor, optional
                Tensor with sequence lengths for batch processing
        label_dict : dict
            Dictionary containing ground truth labels.

        Returns
        -------
        torch.Tensor
            Total loss value (scalar tensor).
        """
        ciassd_loss = self.ciassd_loss(output_dict, label_dict)

        # only update ciassd if no bbox is detected in the first stage
        if "fpvrcnn_out" not in output_dict:
            self.loss_dict = {
                "loss": ciassd_loss,
            }
            return ciassd_loss

        # rcnn out
        rcnn_cls = output_dict["fpvrcnn_out"]["rcnn_cls"].view(1, -1, 1)
        rcnn_iou = output_dict["fpvrcnn_out"]["rcnn_iou"].view(1, -1, 1)
        rcnn_reg = output_dict["fpvrcnn_out"]["rcnn_reg"].view(1, -1, 7)

        tgt_cls = output_dict["rcnn_label_dict"]["cls_tgt"].view(1, -1, 1)
        tgt_iou = output_dict["rcnn_label_dict"]["iou_tgt"].view(1, -1, 1)
        tgt_reg = output_dict["rcnn_label_dict"]["reg_tgt"].view(1, -1, 7)

        _ = tgt_cls.sum()  # pos_norm
        # cls loss
        loss_cls = weighted_sigmoid_binary_cross_entropy(rcnn_cls, tgt_cls)

        # iou loss
        # TODO: also count the negative samples
        loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou, weights=tgt_cls).mean()

        # regression loss
        # Target resampling : Generate a weights mask to force the regressor concentrate on low iou predictions
        # sample 50% with iou>0.7 and 50% < 0.7
        weights = torch.ones(tgt_iou.shape, device=tgt_iou.device)
        weights[tgt_cls == 0] = 0
        neg = torch.logical_and(tgt_iou < 0.7, tgt_cls != 0)
        pos = torch.logical_and(tgt_iou >= 0.7, tgt_cls != 0)
        num_neg = int(neg.sum(dim=1))
        num_pos = int(pos.sum(dim=1))
        num_pos_smps = max(num_neg, 2)
        pos_indices = torch.where(pos)[1]
        not_selsected = torch.randperm(num_pos)[: num_pos - num_pos_smps]
        # not_selsected_indices = pos_indices[not_selsected]
        weights[:, pos_indices[not_selsected]] = 0
        norm = torch.clamp(weights.sum(), min=1.0)
        loss_reg = weighted_smooth_l1_loss(rcnn_reg, tgt_reg, weights=weights / norm).sum()

        loss_cls_reduced = loss_cls * self.cls["weight"]
        loss_iou_reduced = loss_iou * self.iou["weight"]
        loss_reg_reduced = loss_reg * self.reg["weight"]

        # if torch.isnan(loss_reg_reduced):
        #     print('debug')

        rcnn_loss = loss_cls_reduced + loss_iou_reduced + loss_reg_reduced
        loss = rcnn_loss + ciassd_loss

        self.loss_dict.update(
            {
                "loss": loss,
                "rcnn_loss": rcnn_loss,
                "cls_loss": loss_cls_reduced,
                "iou_loss": loss_iou_reduced,
                "reg_loss": loss_reg_reduced,
            }
        )

        return loss

    def logging(self, epoch: int, batch_id: int, batch_len: int, writer: SummaryWriter, pbar: Optional[Any] = None) -> None:
        """
        Log training metrics and losses to console and TensorBoard.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        batch_id : int
            Current batch index within the epoch.
        batch_len : int
            Total number of batches in the dataset.
        writer : SummaryWriter
            TensorBoard SummaryWriter for logging.
        pbar : Any, optional
            Optional progress bar for display. If None, prints to console.
            Default is None.
        """
        ciassd_loss_dict = self.ciassd_loss.loss_dict
        ciassd_total_loss = ciassd_loss_dict["total_loss"]
        reg_loss = ciassd_loss_dict["reg_loss"]
        cls_loss = ciassd_loss_dict["cls_loss"]
        dir_loss = ciassd_loss_dict["dir_loss"]
        iou_loss = ciassd_loss_dict["iou_loss"]

        if (batch_id + 1) % 10 == 0:
            str_to_print = "[epoch %d][%d/%d], || Loss: %.4f || Ciassd: %.4f || Cls1: %.4f || Loc1: %.4f || Dir1: %.4f || Iou1: %.4f" % (
                epoch,
                batch_id + 1,
                batch_len,
                self.loss_dict["loss"],
                ciassd_total_loss.item(),
                cls_loss.item(),
                reg_loss.item(),
                dir_loss.item(),
                iou_loss.item(),
            )
            if "rcnn_loss" in self.loss_dict:
                str_to_print += " || Rcnn: %.4f || Cls2: %.4f || Loc2: %.4f || Iou2: %.4f" % (
                    self.loss_dict["rcnn_loss"],
                    self.loss_dict["cls_loss"].item(),
                    self.loss_dict["reg_loss"].item(),
                    self.loss_dict["iou_loss"].item(),
                )
            print(str_to_print)

        writer.add_scalar("Ciassd_regression_loss", reg_loss.item(), epoch * batch_len + batch_id)
        writer.add_scalar("Ciassd_Confidence_loss", cls_loss.item(), epoch * batch_len + batch_id)
        writer.add_scalar("Ciassd_Direction_loss", dir_loss.item(), epoch * batch_len + batch_id)
        writer.add_scalar("Ciassd_Iou_loss", iou_loss.item(), epoch * batch_len + batch_id)
        writer.add_scalar("Ciassd_loss", ciassd_total_loss.item(), epoch * batch_len + batch_id)
        if "rcnn_loss" in self.loss_dict:
            writer.add_scalar("Rcnn_regression_loss", self.loss_dict["reg_loss"].item(), epoch * batch_len + batch_id)
            writer.add_scalar("Rcnn_Confidence_loss", self.loss_dict["cls_loss"].item(), epoch * batch_len + batch_id)
            writer.add_scalar("Rcnn_Iou_loss", self.loss_dict["iou_loss"].item(), epoch * batch_len + batch_id)
            writer.add_scalar("Rcnn_loss", self.loss_dict["rcnn_loss"].item(), epoch * batch_len + batch_id)
            writer.add_scalar("Total_loss", self.loss_dict["loss"].item(), epoch * batch_len + batch_id)


def weighted_sigmoid_binary_cross_entropy(
    preds: Tensor, tgts: Tensor, weights: Optional[Tensor] = None, class_indices: Optional[torch.LongTensor] = None
) -> Tensor:
    """
    Compute weighted binary cross entropy with logits.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted logits with shape (batch_size, num_anchors, num_classes).
    tgts : torch.Tensor
        Target values with same shape as preds.
    weights : torch.Tensor, optional
        Optional weight tensor for each prediction. Default is None.
    class_indices : torch.LongTensor, optional
        Optional tensor of class indices to apply weights to. Default is None.

    Returns
    -------
    torch.Tensor
        Computed binary cross entropy loss.
    """
    if weights is not None:
        weights = weights.unsqueeze(-1)
    if class_indices is not None:
        if weights is None:
            weights = torch.ones_like(preds)
        weights = weights * indices_to_dense_vector(class_indices, preds.shape[2]).view(1, 1, -1).type_as(preds)
    per_entry_cross_ent = nn.functional.binary_cross_entropy_with_logits(preds, tgts, weight=weights, reduction="none")
    return per_entry_cross_ent


def indices_to_dense_vector(
    indices: torch.LongTensor, size: int, indices_value: float = 1.0, default_value: float = 0, dtype: torch.dtype = torch.float32
) -> Tensor:
    """
    Creates a dense vector with specified indices set to a given value and the rest to default.

    This is a PyTorch implementation that creates a dense vector where only the specified
    indices have the value `indices_value` and all others have `default_value`.

    Parameters
    ----------
    indices : torch.LongTensor
        1D tensor containing integer indices to set to `indices_value`.
    size : int
        Size of the output tensor.
    indices_value : float, optional
        Value to set at the specified indices. Default is 1.0.
    default_value : float, optional
        Value to set for all other indices. Default is 0.
    dtype : torch.dtype, optional
        Data type of the output tensor. Default is torch.float32.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (size,) with values set at specified indices.

    Examples
    --------
    >>> indices = torch.tensor([1, 3, 5])
    >>> indices_to_dense_vector(indices, size=6)
    tensor([0., 1., 0., 1., 0., 1.])
    """
    dense = torch.zeros(size, dtype=dtype, device=indices.device).fill_(default_value)
    dense[indices] = indices_value
    return dense
