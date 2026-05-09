"""Mean recall (precision-recall curve recall axis) metric at configured IoU thresholds."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Mapping, Sequence

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.metrics._opencood_eval import (
    ResultStat,
    accumulate_tp_fp,
    create_empty_result_stat,
    iou_series_name,
    load_eval_utils,
    snapshot_result_stat,
)
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.metrics.mean_recall_at_iou")


class MeanRecallAtIoUMetric(BaseMetric):
    """Report the mean-recall axis of the precision-recall curve at configured IoU thresholds."""

    metric_name = "mean_recall_at_iou"
    iou_thresholds: ClassVar[tuple[float, ...]] = (0.3, 0.5, 0.7)

    def __init__(
        self,
        warmup_steps: int = 0,
        global_sort_detections: bool = True,
    ):
        super().__init__(warmup_steps=warmup_steps)
        self.global_sort_detections = global_sort_detections
        self.result_stat: ResultStat = create_empty_result_stat(self.iou_thresholds)

    def _process_context(self, context: Mapping[str, Any]) -> None:
        accumulate_tp_fp(self.result_stat, self.iou_thresholds, context, self.metric_name)

    def get_raw(self) -> tuple[MetricSeries, ...]:
        if self.steps_count <= self.warmup_steps:
            return tuple(MetricSeries(name=self._series_name(iou), samples=()) for iou in self.iou_thresholds)

        return tuple(
            MetricSeries(
                name=self._series_name(iou),
                samples=tuple(MetricSample(tick=index, value=float(value)) for index, value in enumerate(self._calculate_mrec(iou))),
            )
            for iou in self.iou_thresholds
        )

    def _calculate_mrec(self, iou: float) -> Sequence[float]:
        eval_utils = load_eval_utils()
        _, mrec, _ = eval_utils.calculate_ap(
            snapshot_result_stat(self.result_stat),
            iou,
            self.global_sort_detections,
        )
        return mrec

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Mean Recall at IoU",
            series_names=tuple(cls._series_name(iou) for iou in cls.iou_thresholds),
            summary_specs=tuple(
                MetricSummarySpec(
                    series_name=cls._series_name(iou),
                    display_name=f"Mean Recall at IoU {iou:.1f}",
                )
                for iou in cls.iou_thresholds
            ),
        )

    @staticmethod
    def _series_name(iou: float) -> str:
        return iou_series_name("mrec", iou)
