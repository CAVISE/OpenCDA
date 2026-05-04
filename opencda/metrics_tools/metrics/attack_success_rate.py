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

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.metrics.attack_success_rate")


def _load_common_utils() -> Any:
    from opencood.utils import common_utils

    return common_utils


class AttackSuccessRateMetric(BaseMetric):
    """Collect ASR for spoofing and removal attacks."""

    metric_name = "attack_success_rate"
    _SERIES_BY_MODE: ClassVar[dict[str, str]] = {
        "remove": "asr_remove",
        "removal": "asr_remove",
        "spoof": "asr_spoof",
        "spoofing": "asr_spoof",
    }
    _TARGET_KEY_BY_MODE: ClassVar[dict[str, str]] = {
        "remove": "removed_box_tensor",
        "removal": "removed_box_tensor",
        "spoof": "fake_box_tensor",
        "spoofing": "fake_box_tensor",
    }

    def __init__(self, warmup_steps: int = 0, iou_threshold: float = 0.3):
        super().__init__(warmup_steps=warmup_steps)
        self.iou_threshold = float(iou_threshold)
        self._samples: dict[str, list[MetricSample]] = {
            "asr_remove": [],
            "asr_spoof": [],
        }

    def _process_context(self, context: Mapping[str, Any]) -> None:
        visualization_context = context.get("visualization_context")
        mode = self._normalize_mode(self._get_context_value(visualization_context, "mode"))
        if mode not in self._SERIES_BY_MODE:
            return

        target_boxes = self._get_context_value(visualization_context, self._TARGET_KEY_BY_MODE[mode])
        target_boxes_np = self._to_box_array(target_boxes)
        if target_boxes_np is None:
            return

        match_result = self._match_targets(context.get("pred_box_tensor"), target_boxes_np, mode)
        if match_result is None:
            return

        matched_target_count, target_count = match_result
        success_rate = matched_target_count / target_count if mode in {"spoof", "spoofing"} else (target_count - matched_target_count) / target_count
        self._samples[self._SERIES_BY_MODE[mode]].append(self._make_sample(success_rate))
        self._log_asr(
            mode=mode,
            success_rate=success_rate,
            matched_target_count=matched_target_count,
            target_count=target_count,
        )

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return tuple(
            MetricSeries(
                name=series_name,
                samples=tuple(self._samples[series_name]),
            )
            for series_name in ("asr_remove", "asr_spoof")
        )

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Attack Success Rate",
            series_names=("asr_remove", "asr_spoof"),
            summary_specs=(
                MetricSummarySpec(
                    series_name="asr_remove",
                    display_name="Removal ASR",
                ),
                MetricSummarySpec(
                    series_name="asr_spoof",
                    display_name="Spoofing ASR",
                ),
            ),
        )

    def _match_targets(self, pred_box_tensor: Any, target_boxes_np: npt.NDArray[np.float32], mode: str) -> tuple[int, int] | None:
        try:
            common_utils = _load_common_utils()
            target_polygon_list = list(common_utils.convert_format(target_boxes_np))
        except Exception as error:
            logger.debug("Unable to convert ASR target boxes to polygons: %s", error)
            return None

        target_count = len(target_polygon_list)
        if target_count == 0:
            return None

        pred_boxes_np = self._to_box_array(pred_box_tensor)
        if pred_boxes_np is None:
            return 0, target_count

        try:
            pred_polygon_list = list(common_utils.convert_format(pred_boxes_np))
        except Exception as error:
            logger.debug("Unable to convert ASR predicted boxes to polygons: %s", error)
            return 0, target_count

        matched_target_count = 0
        for target_polygon in target_polygon_list:
            if self._target_has_matching_detection(target_polygon, pred_polygon_list, common_utils, mode):
                matched_target_count += 1
        return matched_target_count, target_count

    def _target_has_matching_detection(self, target_polygon: Any, pred_polygon_list: list[Any], common_utils: Any, mode: str) -> bool:
        if mode in {"remove", "removal"} and self._has_detection_center_inside_target(target_polygon, pred_polygon_list):
            return True

        ious = common_utils.compute_iou(target_polygon, pred_polygon_list)
        return len(ious) > 0 and float(np.max(ious)) >= self.iou_threshold

    @staticmethod
    def _log_asr(
        *,
        mode: str,
        success_rate: float,
        matched_target_count: int,
        target_count: int,
    ) -> None:
        logger.info(
            "AdvCP %s ASR value=%.3f matched_targets=%s/%s",
            mode,
            success_rate,
            matched_target_count,
            target_count,
        )

    @staticmethod
    def _has_detection_center_inside_target(target_polygon: Any, pred_polygon_list: list[Any]) -> bool:
        for pred_polygon in pred_polygon_list:
            try:
                if target_polygon.covers(pred_polygon.centroid):
                    return True
            except AttributeError:
                return False
        return False

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
