"""Mean precision (precision-recall curve precision axis) metric at configured IoU thresholds."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Mapping, Sequence

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.metrics.coperception._opencood_eval import (
    ResultStat,
    accumulate_tp_fp,
    create_empty_result_stat,
    iou_series_name,
    load_eval_utils,
    mean_curve_value,
    snapshot_result_stat,
)
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.metrics.coperception.mean_precision_at_iou")


class MeanPrecisionAtIoUMetric(BaseMetric):
    """Report the mean-precision axis of the precision-recall curve at configured IoU thresholds."""

    metric_name = "mean_precision_at_iou"
    iou_thresholds: ClassVar[tuple[float, ...]] = (0.3, 0.5, 0.7)

    def __init__(
        self,
        warmup_steps: int = 0,
        global_sort_detections: bool = True,
    ):
        super().__init__(warmup_steps=warmup_steps)
        self.global_sort_detections = global_sort_detections
        self.result_stat: ResultStat = create_empty_result_stat(self.iou_thresholds)
        self._samples: dict[str, list[MetricSample]] = {self._series_name(iou): [] for iou in self.iou_thresholds}

    def _process_context(self, context: Mapping[str, Any]) -> None:
        if not accumulate_tp_fp(self.result_stat, self.iou_thresholds, context, self.metric_name):
            return
        for iou in self.iou_thresholds:
            self._samples[self._series_name(iou)].append(self._make_sample(self.calculate_mean_precision(iou)))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return tuple(
            MetricSeries(
                name=self._series_name(iou),
                samples=tuple(self._samples[self._series_name(iou)]),
            )
            for iou in self.iou_thresholds
        )

    def calculate_mean_precision(self, iou: float) -> float:
        return mean_curve_value(self._calculate_mpre(iou))

    def _calculate_mpre(self, iou: float) -> Sequence[float]:
        eval_utils = load_eval_utils()
        _, _, mpre = eval_utils.calculate_ap(
            snapshot_result_stat(self.result_stat),
            iou,
            self.global_sort_detections,
        )
        return mpre

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Mean Precision at IoU",
            series_names=tuple(cls._series_name(iou) for iou in cls.iou_thresholds),
            summary_specs=tuple(
                MetricSummarySpec(
                    series_name=cls._series_name(iou),
                    display_name=f"Mean Precision at IoU {iou:.1f}",
                )
                for iou in cls.iou_thresholds
            ),
        )

    @staticmethod
    def _series_name(iou: float) -> str:
        return iou_series_name("mpre", iou)
