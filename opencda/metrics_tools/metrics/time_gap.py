"""Platooning time-gap metric implementation."""

from typing import Mapping, Any

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec
from opencda.metrics_tools.metric_sample import MetricSample


class TimeGapMetric(BaseMetric):  # noqa DC03
    """Collect platooning time-gap samples."""

    metric_name = "time_gap"

    def __init__(self, warmup_steps: int = 100):
        super().__init__(warmup_steps=warmup_steps)
        self._samples: list[MetricSample] = []

    def _process_context(self, context: Mapping[str, Any]) -> None:
        time_gap = float(context.get("time_gap", 100.0))
        self._samples.append(self._make_sample(time_gap))

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Time Gap",
            series_names=("time_gap",),
            summary_specs=(
                MetricSummarySpec(
                    series_name="time_gap",
                    cutoff=100.0,
                ),
            ),
        )
