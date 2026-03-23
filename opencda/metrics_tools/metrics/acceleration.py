from typing import Mapping

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec


class AccelerationMetric(BaseMetric):
    """Metric for ego acceleration."""

    metric_name = "acceleration"

    def __init__(self, warmup_steps: int = 100, dt: float = 0.05):
        super().__init__(warmup_steps=warmup_steps, sample_interval=dt)
        self.dt = dt
        self._acceleration_samples: list[MetricSample] = []
        self._previous_speed: float | None = None

    @property
    def acceleration_list(self) -> list[float]:
        return [sample.value for sample in self._acceleration_samples]

    def _process_context(self, context: Mapping[str, object]) -> None:
        ego_speed = float(context.get("ego_speed", 0.0)) / 3.6

        if self._previous_speed is None:
            acceleration_value = 0.0
        else:
            acceleration_value = (ego_speed - self._previous_speed) / self.dt

        self._previous_speed = ego_speed
        self._acceleration_samples.append(self._make_sample(acceleration_value))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return (
            MetricSeries(
                name="acceleration",
                samples=tuple(self._acceleration_samples),
            ),
        )

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Acceleration",
            series_names=("acceleration",),
            summary_specs=(MetricSummarySpec(series_name="acceleration"),),
        )
