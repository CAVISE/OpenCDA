"""
Evaluation utilities for object detection metrics.

This module provides functions for calculating average precision (AP), 
true positives (TP), false positives (FP), and other evaluation metrics 
for object detection tasks.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from opencood.hypes_yaml import yaml_utils
from opencood.utils import common_utils

logger = logging.getLogger("cavise.OpenCOOD.opencood.utils.eval_utils")


def voc_ap(rec: List[float], prec: List[float]) -> Tuple[float, List[float], List[float]]:
    """
    VOC 2010 Average Precision.

    Parameters
    ----------
    rec : list of float
        Recall values.
    prec : list of float
        Precision values.

    Returns
    -------
    ap : float
        Average precision value.
    mrec : list of float
        Modified recall values with boundaries.
    mpre : list of float
        Modified precision values with boundaries.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


def caluclate_tp_fp(
    det_boxes: Optional[torch.Tensor],
    det_score: Optional[torch.Tensor],
    gt_boxes: torch.Tensor,
    result_stat: Dict[float, Dict[str, List[Any]]],
    iou_thresh: float
) -> None:
    """
    Calculate true positive and false positive numbers for current frames.

    Parameters
    ----------
    det_boxes : torch.Tensor or None
        Detection bounding boxes with shape (N, 8, 3) or (N, 4, 2).
    det_score : torch.Tensor or None
        Confidence scores for each predicted bounding box.
    gt_boxes : torch.Tensor
        Ground truth bounding boxes.
    result_stat : dict
        Dictionary containing fp, tp and gt statistics for different IoU thresholds.
    iou_thresh : float
        IoU threshold for matching predictions to ground truth.

    Returns
    -------
    None
        Updates result_stat in-place.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend]  # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]["score"] += det_score.tolist()

    result_stat[iou_thresh]["fp"] += fp
    result_stat[iou_thresh]["tp"] += tp
    result_stat[iou_thresh]["gt"] += gt


def calculate_ap(
    result_stat: Dict[float, Dict[str, List[Any]]],
    iou: float,
    global_sort_detections: bool
) -> Tuple[float, List[float], List[float]]:
    """
    Calculate average precision and recall values.

    Parameters
    ----------
    result_stat : dict
        Dictionary containing fp, tp and gt statistics.
    iou : float
        IoU threshold value.
    global_sort_detections : bool
        Whether to sort detection results globally by confidence score.

    Returns
    -------
    ap : float
        Average precision value.
    mrec : list of float
        Modified recall values.
    mprec : list of float
        Modified precision values.
    """
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5["fp"])
        tp = np.array(iou_5["tp"])
        score = np.array(iou_5["score"])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()
    else:
        fp = iou_5["fp"]
        tp = iou_5["tp"]
        assert len(fp) == len(tp)

    gt_total = iou_5["gt"]

    if gt_total == 0:
        logger.warning("Variable gt_total is 0")
        return 0.0, [0], [0]

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(
    result_stat: Dict[float, Dict[str, List[Any]]],
    save_path: str,
    global_sort_detections: bool
) -> None:
    """
    Evaluate and save final detection results.

    Parameters
    ----------
    result_stat : dict
        Dictionary containing evaluation statistics for different IoU thresholds.
    save_path : str
        Directory path to save the evaluation results.
    global_sort_detections : bool
        Whether to sort detections globally by confidence score.

    Returns
    -------
    None
        Saves evaluation results to a YAML file and logs AP metrics.
    """
    dump_dict = {}
    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30, global_sort_detections)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50, global_sort_detections)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70, global_sort_detections)

    dump_dict.update(
        {
            "ap30": ap_30,
            "ap_50": ap_50,
            "ap_70": ap_70,
            "mpre_50": mpre_50,
            "mrec_50": mrec_50,
            "mpre_70": mpre_70,
            "mrec_70": mrec_70,
        }
    )

    output_file = "eval.yaml" if not global_sort_detections else "eval_global_sort.yaml"
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

    logger.info(f"The Average Precision at IOU 0.3 is {ap_30:.3f}")
    logger.info(f"The Average Precision at IOU 0.5 is {ap_50:.3f}")
    logger.info(f"The Average Precision at IOU 0.7 is {ap_70:.3f}")
