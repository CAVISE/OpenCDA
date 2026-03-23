from typing import Mapping

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec


class SpeedMetric(BaseMetric):
    """Metric for ego speed."""

    metric_name = "speed"
    required_capabilities = ("ego_speed",)

    def __init__(self, warmup_steps: int = 100, dt: float = 0.05):
        super().__init__(warmup_steps=warmup_steps, sample_interval=dt)
        self._speed_samples: list[MetricSample] = []

    @property
    def speed_list(self) -> list[float]:
        return [sample.value for sample in self._speed_samples]

    def _process_context(self, context: Mapping[str, object]) -> None:
        ego_speed = float(context.get("ego_speed", 0.0))
        self._speed_samples.append(self._make_sample(ego_speed / 3.6))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return (MetricSeries(name="speed", samples=tuple(self._speed_samples)),)

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Speed",
            series_names=("speed",),
            summary_specs=(MetricSummarySpec(series_name="speed"),),
        )
