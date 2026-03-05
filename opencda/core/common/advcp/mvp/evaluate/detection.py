from typing import Any, Dict

import numpy as np

from mvp.tools.iou import iou3d


def iou3d_batch(gt_bboxes: np.ndarray, pred_bboxes: np.ndarray) -> np.ndarray:
    iou = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for gt_idx, gt_bbox in enumerate(gt_bboxes):
        for pred_idx, pred_bbox in enumerate(pred_bboxes):
            iou[gt_idx, pred_idx] = iou3d(gt_bbox, pred_bbox)
    return iou


def evaluate_single_vehicle(gt_bboxes: np.ndarray, pred_bboxes: np.ndarray, iou_thres: float = 0.7) -> Dict[str, Any]:
    iou = iou3d_batch(gt_bboxes, pred_bboxes)
    report: Dict[str, Any] = {"iou": iou, "gt": {"bboxes": gt_bboxes}, "pred": {"bboxes": pred_bboxes}}

    iou_mask = (iou >= iou_thres).astype(np.uint8)
    P, PP = gt_bboxes.shape[0], pred_bboxes.shape[0]
    TP = int(np.sum(np.max(iou, axis=0) >= iou_thres))
    FP = PP - TP
    FN = P - TP

    TP_bbox_mask = np.max(iou, axis=1) >= iou_thres
    FP_bbox_mask = np.max(iou, axis=0) < iou_thres
    FN_bbox_mask = np.max(iou, axis=1) < iou_thres

    report["metrics"] = {
        "P": P,
        "PP": PP,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "iou_mask": iou_mask,
        "TP_bbox_mask": TP_bbox_mask,
        "FP_bbox_mask": FP_bbox_mask,
        "FN_bbox_mask": FN_bbox_mask,
    }

    return report
