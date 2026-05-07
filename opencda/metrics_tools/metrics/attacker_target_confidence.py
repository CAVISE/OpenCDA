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


class AttackerTargetConfidenceMetric(BaseMetric):
    """Mean model confidence on per-tick attacker target boxes (removal and spoofing)."""

    metric_name = "attacker_target_confidence"
    _SPEC_BY_MODE: ClassVar[dict[str, tuple[str, str]]] = {
        "removal": ("confidence_removal", "removed_box_tensor"),
        "spoofing": ("confidence_spoofing", "fake_box_tensor"),
    }

    def __init__(self, warmup_steps: int = 0, iou_threshold: float = 0.3):
        super().__init__(warmup_steps=warmup_steps)
        self.iou_threshold = float(iou_threshold)
        self._samples: dict[str, list[MetricSample]] = {series_name: [] for series_name, _ in self._SPEC_BY_MODE.values()}

    def _process_context(self, context: Mapping[str, Any]) -> None:
        visualization_context = context.get("visualization_context")
        mode = self._normalize_mode(self._get_context_value(visualization_context, "mode"))
        spec = self._SPEC_BY_MODE.get(mode)
        if spec is None:
            return
        series_name, target_key = spec

        target_boxes = self._get_context_value(visualization_context, target_key)
        target_boxes_np = self._to_box_array(target_boxes)
        if target_boxes_np is None:
            return

        per_target_confidence = self._collect_per_target_confidence(
            context.get("pred_box_tensor"),
            context.get("pred_score"),
            target_boxes_np,
            mode,
        )
        if per_target_confidence is None:
            return

        mean_confidence = float(np.mean(per_target_confidence)) if len(per_target_confidence) > 0 else 0.0
        self._samples[series_name].append(self._make_sample(mean_confidence))
        self._log_confidence(
            mode=mode,
            mean_confidence=mean_confidence,
            target_count=len(per_target_confidence),
        )

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return tuple(
            MetricSeries(
                name=series_name,
                samples=tuple(self._samples[series_name]),
            )
            for series_name, _ in self._SPEC_BY_MODE.values()
        )

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Attacker Target Confidence",
            series_names=tuple(series_name for series_name, _ in cls._SPEC_BY_MODE.values()),
            summary_specs=(
                MetricSummarySpec(
                    series_name="confidence_removal",
                    display_name="Removal Target Confidence",
                ),
                MetricSummarySpec(
                    series_name="confidence_spoofing",
                    display_name="Spoofing Target Confidence",
                ),
            ),
        )

    def _collect_per_target_confidence(
        self,
        pred_box_tensor: Any,
        pred_score: Any,
        target_boxes_np: npt.NDArray[np.float32],
        mode: str,
    ) -> list[float] | None:
        try:
            common_utils = _load_common_utils()
            target_polygon_list = list(common_utils.convert_format(target_boxes_np))
        except Exception as error:
            logger.debug("Unable to convert target boxes to polygons: %s", error)
            return None

        target_count = len(target_polygon_list)
        if target_count == 0:
            return None

        pred_scores_np = self._to_score_array(pred_score)
        pred_boxes_np = self._to_box_array(pred_box_tensor)
        if pred_boxes_np is None or pred_scores_np is None or pred_boxes_np.shape[0] != pred_scores_np.shape[0]:
            return [0.0] * target_count

        try:
            pred_polygon_list = list(common_utils.convert_format(pred_boxes_np))
        except Exception as error:
            logger.debug("Unable to convert predicted boxes to polygons: %s", error)
            return [0.0] * target_count

        return [
            self._confidence_for_target(target_polygon, pred_polygon_list, pred_scores_np, common_utils, mode)
            for target_polygon in target_polygon_list
        ]

    def _confidence_for_target(
        self,
        target_polygon: Any,
        pred_polygon_list: list[Any],
        pred_scores_np: npt.NDArray[np.float32],
        common_utils: Any,
        mode: str,
    ) -> float:
        if mode == "removal":
            covers_index = self._first_detection_index_inside_target(target_polygon, pred_polygon_list)
            if covers_index is not None:
                return float(pred_scores_np[covers_index])

        ious = common_utils.compute_iou(target_polygon, pred_polygon_list)
        if len(ious) == 0:
            return 0.0
        ious_np = np.asarray(ious, dtype=np.float32)
        best_index = int(np.argmax(ious_np))
        if float(ious_np[best_index]) < self.iou_threshold:
            return 0.0
        return float(pred_scores_np[best_index])

    @staticmethod
    def _first_detection_index_inside_target(target_polygon: Any, pred_polygon_list: list[Any]) -> int | None:
        for pred_index, pred_polygon in enumerate(pred_polygon_list):
            try:
                if target_polygon.covers(pred_polygon.centroid):
                    return pred_index
            except AttributeError:
                return None
        return None

    @staticmethod
    def _log_confidence(
        *,
        mode: str,
        mean_confidence: float,
        target_count: int,
    ) -> None:
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
