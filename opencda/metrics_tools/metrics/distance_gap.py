"""Platooning distance-gap metric implementation."""

from typing import Mapping

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec


class DistanceGapMetric(BaseMetric):
    """Collect platooning distance-gap samples."""

    metric_name = "distance_gap"

    def __init__(self, warmup_steps: int = 100):
        super().__init__(warmup_steps=warmup_steps)
        self._samples = []

    def _process_context(self, context: Mapping[str, object]) -> None:
        distance_gap = float(context.get("distance_gap", 100.0))
        self._samples.append(self._make_sample(distance_gap))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return (MetricSeries(name="distance_gap", samples=tuple(self._samples)),)

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Distance Gap",
            series_names=("distance_gap",),
            summary_specs=(
                MetricSummarySpec(
                    series_name="distance_gap",
                    cutoff=100.0,
                ),
            ),
        )
