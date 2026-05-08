"""Model confidence on attacker target boxes for AdvCP attacks."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Mapping

import numpy as np
import numpy.typing as npt

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.metrics.attacker_target_confidence")


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


def _best_score_for_target(
    target_polygon: Any,
    pred_polygons: list[Any],
    pred_scores_np: npt.NDArray,
    common_utils: Any,
    iou_threshold: float,
) -> float:
    """Return the highest pred score for a detection that overlaps target by >= iou_threshold, or 0."""
    ious = common_utils.compute_iou(target_polygon, pred_polygons)
    if len(ious) == 0:
        return 0.0
    ious_np = np.asarray(ious, dtype=np.float32)
    best_idx = int(np.argmax(ious_np))
    if float(ious_np[best_idx]) < iou_threshold:
        return 0.0
    return float(pred_scores_np[best_idx])


class AttackerTargetConfidenceMetric(BaseMetric):
    """Mean model confidence on per-tick attacker target boxes (removal and spoofing)."""

    metric_name = "attacker_target_confidence"
    _SERIES_NAMES: ClassVar[tuple[str, str]] = ("confidence_removal", "confidence_spoofing")

    def __init__(self, warmup_steps: int = 0, iou_threshold: float = 0.3):
        super().__init__(warmup_steps=warmup_steps)
        self.iou_threshold = float(iou_threshold)
        self._samples: dict[str, list[MetricSample]] = {name: [] for name in self._SERIES_NAMES}

    def _process_context(self, context: Mapping[str, Any]) -> None:
        visualization_context = context.get("visualization_context")
        mode = self._normalize_mode(self._get_context_value(visualization_context, "mode"))

        if mode == "removal":
            per_target = self._collect_removal_confidence(context, visualization_context)
            series_name = "confidence_removal"
        elif mode == "spoofing":
            per_target = self._collect_spoofing_confidence(context, visualization_context)
            series_name = "confidence_spoofing"
        else:
            return

        if per_target is None:
            return

        mean_confidence = float(np.mean(per_target)) if per_target else 0.0
        self._samples[series_name].append(self._make_sample(mean_confidence))
        self._log_confidence(mode=mode, mean_confidence=mean_confidence, target_count=len(per_target))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return tuple(MetricSeries(name=name, samples=tuple(self._samples[name])) for name in self._SERIES_NAMES)

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Attacker Target Confidence",
            series_names=cls._SERIES_NAMES,
            summary_specs=(
                MetricSummarySpec(series_name="confidence_removal", display_name="Removal Target Confidence"),
                MetricSummarySpec(series_name="confidence_spoofing", display_name="Spoofing Target Confidence"),
            ),
        )

    def _collect_removal_confidence(
        self,
        context: Mapping[str, Any],
        visualization_context: Any,
    ) -> list[float] | None:
        """
        Targets = GT objects whose centroid lies inside the removal zone.
        Confidence for each = best pred score with IoU >= threshold, else 0.
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

        if not target_indices:
            return None

        pred_scores_np = self._to_score_array(context.get("pred_score"))
        pred_np = self._to_box_array(context.get("pred_box_tensor"))
        if pred_np is None or pred_scores_np is None or pred_np.shape[0] != pred_scores_np.shape[0]:
            return [0.0] * len(target_indices)

        try:
            target_polygons = list(common_utils.convert_format(gt_np[target_indices]))
            pred_polygons = list(common_utils.convert_format(pred_np))
        except Exception as error:
            logger.debug("Unable to convert removal boxes to polygons: %s", error)
            return [0.0] * len(target_indices)

        return [_best_score_for_target(tp, pred_polygons, pred_scores_np, common_utils, self.iou_threshold) for tp in target_polygons]

    def _collect_spoofing_confidence(
        self,
        context: Mapping[str, Any],
        visualization_context: Any,
    ) -> list[float] | None:
        """
        Targets = unique configured spoof boxes (deduplicated fake_box_tensor).
        Confidence for each = best pred score with IoU >= threshold, else 0.
        """
        fake_np = self._to_box_array(self._get_context_value(visualization_context, "fake_box_tensor"))
        if fake_np is None:
            return None

        unique_np = _deduplicate_boxes(fake_np)

        pred_scores_np = self._to_score_array(context.get("pred_score"))
        pred_np = self._to_box_array(context.get("pred_box_tensor"))

        try:
            common_utils = _load_common_utils()
            target_polygons = list(common_utils.convert_format(unique_np))
        except Exception as error:
            logger.debug("Unable to convert spoof boxes to polygons: %s", error)
            return None

        if not target_polygons:
            return None

        if pred_np is None or pred_scores_np is None or pred_np.shape[0] != pred_scores_np.shape[0]:
            return [0.0] * len(target_polygons)

        try:
            pred_polygons = list(common_utils.convert_format(pred_np))
        except Exception as error:
            logger.debug("Unable to convert pred boxes to polygons: %s", error)
            return [0.0] * len(target_polygons)

        return [_best_score_for_target(tp, pred_polygons, pred_scores_np, common_utils, self.iou_threshold) for tp in target_polygons]

    @staticmethod
    def _log_confidence(*, mode: str, mean_confidence: float, target_count: int) -> None:
        logger.info(
            "AdvCP %s target confidence value=%.3f targets=%s",
            mode,
            mean_confidence,
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

    @staticmethod
    def _to_score_array(value: Any) -> npt.NDArray[np.float32] | None:
        if value is None:
            return None

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()

        try:
            scores = np.asarray(value)
        except (TypeError, ValueError):
            return None

        if scores.ndim == 0:
            scores = scores[np.newaxis]
        if scores.ndim != 1 or scores.shape[0] == 0:
            return None
        if not np.issubdtype(scores.dtype, np.number):
            return None
        return scores.astype(np.float32, copy=False)
