"""Average precision at IoU metric for cooperative perception."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, ClassVar, ItemsView, Mapping, Sequence

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.metrics.ap_at_iou")


def _load_eval_utils() -> Any:
    from opencood.utils import eval_utils

    return eval_utils


@dataclass
class IoUResultStat:
    tp: list[Any]
    fp: list[Any]
    gt: int
    score: list[Any]

    @classmethod
    def create_empty(cls) -> "IoUResultStat":
        return cls(tp=[], fp=[], gt=0, score=[])

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": list(self.tp),
            "fp": list(self.fp),
            "gt": self.gt,
            "score": list(self.score),
        }

    def merge_from(self, other: "IoUResultStat") -> None:
        self.gt += other.gt
        self.tp += other.tp
        self.fp += other.fp
        self.score += other.score


@dataclass
class EvaluationResultStat:
    by_iou: dict[float, IoUResultStat]

    IOU_THRESHOLDS: ClassVar[tuple[float, ...]] = (0.3, 0.5, 0.7)

    @classmethod
    def create_empty(cls, iou_thresholds: Sequence[float] | None = None) -> "EvaluationResultStat":
        thresholds = tuple(cls.IOU_THRESHOLDS if iou_thresholds is None else iou_thresholds)
        return cls({float(iou): IoUResultStat.create_empty() for iou in thresholds})

    def __getitem__(self, iou: float) -> IoUResultStat:
        return self.by_iou[iou]

    def items(self) -> ItemsView[float, IoUResultStat]:
        return self.by_iou.items()

    def as_dict(self) -> dict[float, dict[str, Any]]:
        return {iou: stat.as_dict() for iou, stat in self.by_iou.items()}

    def merge_from(self, other: "EvaluationResultStat") -> None:
        for iou, stat in other.items():
            self.by_iou.setdefault(iou, IoUResultStat.create_empty()).merge_from(stat)


class APAtIoUMetric(BaseMetric):
    """Collect detection stats and report AP at configured IoU thresholds."""

    metric_name = "ap_at_iou"
    iou_thresholds: ClassVar[tuple[float, ...]] = EvaluationResultStat.IOU_THRESHOLDS

    def __init__(
        self,
        warmup_steps: int = 0,
        global_sort_detections: bool = True,
    ):
        super().__init__(warmup_steps=warmup_steps)
        self.global_sort_detections = global_sort_detections
        self.result_stat = EvaluationResultStat.create_empty(self.iou_thresholds)

    def _process_context(self, context: Mapping[str, Any]) -> None:
        gt_box_tensor = context.get("gt_box_tensor")
        if gt_box_tensor is None:
            raise ValueError("AP at IoU metric requires 'gt_box_tensor' in the update context.")

        pred_box_tensor = context.get("pred_box_tensor")
        pred_score = context.get("pred_score")
        eval_utils = _load_eval_utils()

        for iou in self.iou_thresholds:
            eval_utils.caluclate_tp_fp(
                pred_box_tensor,
                pred_score,
                gt_box_tensor,
                self.result_stat,
                iou,
            )
        self._log_ap_at_iou()

    def get_raw(self) -> tuple[MetricSeries, ...]:
        if self.steps_count <= self.warmup_steps:
            return tuple(MetricSeries(name=self._series_name(iou), samples=()) for iou in self.iou_thresholds)

        return tuple(
            MetricSeries(
                name=self._series_name(iou),
                samples=(
                    MetricSample(
                        tick=self.steps_count,
                        value=self.calculate_ap(iou),
                    ),
                ),
            )
            for iou in self.iou_thresholds
        )

    def calculate_ap(self, iou: float, global_sort_detections: bool | None = None) -> float:
        eval_utils = _load_eval_utils()
        ap, _, _ = eval_utils.calculate_ap(
            self.result_stat.as_dict(),
            iou,
            self.global_sort_detections if global_sort_detections is None else global_sort_detections,
        )
        return float(ap)

    def save_eval_results(self, save_path: str, global_sort_detections: bool | None = None) -> None:
        eval_utils = _load_eval_utils()
        eval_utils.eval_final_results(
            self.result_stat.as_dict(),
            save_path,
            self.global_sort_detections if global_sort_detections is None else global_sort_detections,
        )

    def _log_ap_at_iou(self) -> None:
        ap_parts = []
        for iou in self.iou_thresholds:
            ap_parts.append(f"AP@IoU {iou:.1f}={self.calculate_ap(iou):.3f}")
        logger.info("Cooperative perception %s", ", ".join(ap_parts))

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Average Precision at IoU",
            series_names=tuple(cls._series_name(iou) for iou in cls.iou_thresholds),
            summary_specs=tuple(
                MetricSummarySpec(
                    series_name=cls._series_name(iou),
                    display_name=f"AP at IoU {iou:.1f}",
                )
                for iou in cls.iou_thresholds
            ),
        )

    @staticmethod
    def _series_name(iou: float) -> str:
        return f"ap_iou_{str(iou).replace('.', '_')}"
