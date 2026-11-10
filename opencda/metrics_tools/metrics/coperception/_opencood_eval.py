"""Shared helpers for metrics derived from OpenCOOD's `result_stat` accumulation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np

ResultStat: TypeAlias = dict[float, dict[str, Any]]


def load_eval_utils() -> Any:
    from opencood.utils import eval_utils

    return eval_utils


def create_empty_result_stat(iou_thresholds: Sequence[float]) -> ResultStat:
    return {float(iou): {"tp": [], "fp": [], "gt": 0, "score": []} for iou in iou_thresholds}


def snapshot_result_stat(result_stat: ResultStat) -> ResultStat:
    """Return a copy with fresh lists, since eval_utils mutates tp/fp in place."""
    return {
        iou: {
            "tp": list(stat["tp"]),
            "fp": list(stat["fp"]),
            "gt": stat["gt"],
            "score": list(stat["score"]),
        }
        for iou, stat in result_stat.items()
    }


def accumulate_tp_fp(
    result_stat: ResultStat,
    iou_thresholds: Sequence[float],
    context: Mapping[str, Any],
    metric_name: str,
) -> bool:
    gt_box_tensor = context.get("gt_box_tensor")
    if gt_box_tensor is None:
        return False

    pred_box_tensor = context.get("pred_box_tensor")
    pred_score = context.get("pred_score")
    eval_utils = load_eval_utils()
    for iou in iou_thresholds:
        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat,
            iou,
        )
    return True


def calculate_threshold_metrics(
    result_stat: ResultStat,
    iou: float,
    confidence_threshold: float = 0.5,
) -> dict[str, int | float | None]:
    stat = result_stat[iou]

    scores = np.asarray(stat["score"])
    tp_flags = np.asarray(stat["tp"])
    fp_flags = np.asarray(stat["fp"])

    selected = scores >= confidence_threshold

    tp = int(tp_flags[selected].sum())
    fp = int(fp_flags[selected].sum())
    fn = int(stat["gt"]) - tp

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / stat["gt"] if stat["gt"] else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def mean_curve_value(curve_values: Sequence[float]) -> float:
    """Return a scalar summary for a VOC AP precision/recall curve."""
    values = list(curve_values)
    if len(values) >= 2:
        values = values[1:-1]
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def iou_series_name(prefix: str, iou: float) -> str:
    return f"{prefix}_iou_{str(iou).replace('.', '_')}"
