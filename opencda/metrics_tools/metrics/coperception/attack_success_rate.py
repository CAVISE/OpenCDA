"""Attack success rate metric for cooperative perception attacks."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Mapping

import numpy as np
import numpy.typing as npt

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.metrics.coperception.attack_success_rate")


def _load_common_utils() -> Any:
    from opencood.utils import common_utils

    return common_utils


def _find_gt_in_removal_zone(
    gt_np: npt.NDArray,
    removal_np: npt.NDArray,
    common_utils: Any,
) -> list[int]:
    """Return indices of GT boxes whose XY centroid lies inside any removal zone polygon."""
    gt_polygons = list(common_utils.convert_format(gt_np))
    removal_polygons = list(common_utils.convert_format(removal_np))
    indices = []
    for i, gt_poly in enumerate(gt_polygons):
        centroid = gt_poly.centroid
        for removal_poly in removal_polygons:
            if removal_poly.contains(centroid):
                indices.append(i)
                break
    return indices


def _deduplicate_boxes(boxes_np: npt.NDArray, iou_threshold: float = 0.9) -> npt.NDArray:
    """Remove near-duplicate boxes (IoU >= iou_threshold), keeping the first occurrence."""
    if len(boxes_np) <= 1:
        return boxes_np
    try:
        common_utils = _load_common_utils()
        polygons = list(common_utils.convert_format(boxes_np))
    except Exception:
        return boxes_np
    keep: list[int] = []
    for i, poly in enumerate(polygons):
        for kept_idx in keep:
            union_area = poly.union(polygons[kept_idx]).area
            if union_area > 0 and poly.intersection(polygons[kept_idx]).area / union_area >= iou_threshold:
                break
        else:
            keep.append(i)
    return boxes_np[keep]


def _has_iou_match(
    target_polygon: Any,
    pred_polygons: list[Any],
    common_utils: Any,
    iou_threshold: float,
) -> bool:
    if not pred_polygons:
        return False
    ious = common_utils.compute_iou(target_polygon, pred_polygons)
    return len(ious) > 0 and float(np.max(ious)) >= iou_threshold


class AttackSuccessRateMetric(BaseMetric):
    """Collect ASR for spoofing and removal attacks."""

    metric_name = "attack_success_rate"
    _SERIES_NAMES: ClassVar[tuple[str, str]] = ("asr_removal", "asr_spoofing")

    def __init__(self, warmup_steps: int = 0, iou_threshold: float = 0.3):
        super().__init__(warmup_steps=warmup_steps)
        self.iou_threshold = float(iou_threshold)
        self._samples: dict[str, list[MetricSample]] = {name: [] for name in self._SERIES_NAMES}

    def _process_context(self, context: Mapping[str, Any]) -> None:
        visualization_context = context.get("visualization_context")
        mode = self._normalize_mode(self._get_context_value(visualization_context, "mode"))

        if mode == "removal":
            result = self._compute_removal_result(context, visualization_context)
            series_name = "asr_removal"
        elif mode == "spoofing":
            result = self._compute_spoofing_result(context, visualization_context)
            series_name = "asr_spoofing"
        else:
            return

        if result is None:
            return

        matched_count, target_count = result
        success_rate = matched_count / target_count if mode == "spoofing" else (target_count - matched_count) / target_count
        self._samples[series_name].append(self._make_sample(success_rate))
        self._log_asr(mode=mode, success_rate=success_rate, matched_count=matched_count, target_count=target_count)

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return tuple(MetricSeries(name=name, samples=tuple(self._samples[name])) for name in self._SERIES_NAMES)

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Attack Success Rate",
            series_names=cls._SERIES_NAMES,
            summary_specs=(
                MetricSummarySpec(series_name="asr_removal", display_name="Removal ASR"),
                MetricSummarySpec(series_name="asr_spoofing", display_name="Spoofing ASR"),
            ),
        )

    def _compute_removal_result(
        self,
        context: Mapping[str, Any],
        visualization_context: Any,
    ) -> tuple[int, int] | None:
        """
        Targets = GT objects whose centroid lies inside the removal zone.
        Matched = still detected (IoU with any pred >= threshold).
        ASR = (target_count - matched_count) / target_count.
        """
        removal_np = self._to_box_array(self._get_context_value(visualization_context, "removed_box_tensor"))
        gt_np = self._to_box_array(context.get("gt_box_tensor"))
        if removal_np is None or gt_np is None:
            return None

        try:
            common_utils = _load_common_utils()
            target_indices = _find_gt_in_removal_zone(gt_np, removal_np, common_utils)
        except Exception as error:
            logger.debug("Unable to find GT targets in removal zone: %s", error)
            return None

        target_count = len(target_indices)
        if target_count == 0:
            return None

        pred_np = self._to_box_array(context.get("pred_box_tensor"))
        if pred_np is None:
            return 0, target_count

        try:
            target_polygons = list(common_utils.convert_format(gt_np[target_indices]))
            pred_polygons = list(common_utils.convert_format(pred_np))
        except Exception as error:
            logger.debug("Unable to convert removal boxes to polygons: %s", error)
            return 0, target_count

        matched_count = sum(1 for tp in target_polygons if _has_iou_match(tp, pred_polygons, common_utils, self.iou_threshold))
        return matched_count, target_count

    def _compute_spoofing_result(
        self,
        context: Mapping[str, Any],
        visualization_context: Any,
    ) -> tuple[int, int] | None:
        """
        Targets = unique configured spoof boxes (deduplicated fake_box_tensor).
        Matched = detected in predictions (IoU with any pred >= threshold).
        ASR = matched_count / target_count.
        """
        fake_np = self._to_box_array(self._get_context_value(visualization_context, "fake_box_tensor"))
        if fake_np is None:
            return None

        unique_np = _deduplicate_boxes(fake_np)

        try:
            common_utils = _load_common_utils()
            target_polygons = list(common_utils.convert_format(unique_np))
        except Exception as error:
            logger.debug("Unable to convert spoof boxes to polygons: %s", error)
            return None

        target_count = len(target_polygons)
        if target_count == 0:
            return None

        pred_np = self._to_box_array(context.get("pred_box_tensor"))
        if pred_np is None:
            return 0, target_count

        try:
            pred_polygons = list(common_utils.convert_format(pred_np))
        except Exception as error:
            logger.debug("Unable to convert pred boxes to polygons: %s", error)
            return 0, target_count

        matched_count = sum(1 for tp in target_polygons if _has_iou_match(tp, pred_polygons, common_utils, self.iou_threshold))
        return matched_count, target_count

    @staticmethod
    def _log_asr(*, mode: str, success_rate: float, matched_count: int, target_count: int) -> None:
        logger.info(
            "AdvCP %s ASR value=%.3f matched_targets=%s/%s",
            mode,
            success_rate,
            matched_count,
            target_count,
        )

    @staticmethod
    def _get_context_value(context: Any, key: str) -> Any:
        if context is None:
            return None
        if isinstance(context, Mapping):
            return context.get(key)
        return getattr(context, key, None)

    @staticmethod
    def _normalize_mode(mode: Any) -> str:
        return "" if mode is None else str(mode).strip().lower()

    @staticmethod
    def _to_box_array(value: Any) -> npt.NDArray[np.float32] | None:
        if value is None:
            return None

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()

        try:
            boxes = np.asarray(value)
        except (TypeError, ValueError):
            return None

        if boxes.ndim == 2:
            boxes = boxes[np.newaxis, ...]

        if boxes.ndim < 3 or boxes.shape[0] == 0 or boxes.shape[-1] < 2:
            return None
        if not np.issubdtype(boxes.dtype, np.number):
            return None
        return boxes.astype(np.float32, copy=False)
