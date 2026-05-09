"""Shared helpers for metrics derived from OpenCOOD's `result_stat` accumulation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeAlias

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
) -> None:
    gt_box_tensor = context.get("gt_box_tensor")
    if gt_box_tensor is None:
        raise ValueError(f"{metric_name} metric requires 'gt_box_tensor' in the update context.")

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


def iou_series_name(prefix: str, iou: float) -> str:
    return f"{prefix}_iou_{str(iou).replace('.', '_')}"
