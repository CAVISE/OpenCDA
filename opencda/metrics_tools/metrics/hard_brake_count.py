"""Hard-brake event counter metric."""

from typing import Any, Mapping

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec


class HardBrakeCountMetric(BaseMetric):  # noqa DC03
    """Count hard-braking episodes from ego speed samples."""

    metric_name = "hard_brake_count"

    def __init__(
        self,
        warmup_steps: int = 100,
        dt: float = 0.05,
        deceleration_threshold: float = -3.0,
        reset_threshold: float = -1.0,
    ):
        super().__init__(warmup_steps=warmup_steps, sample_interval=dt)
        self.dt = dt
        self.deceleration_threshold = deceleration_threshold
        self.reset_threshold = reset_threshold
        self._previous_speed: float | None = None
        self._in_hard_brake = False
        self._count = 0
        self._samples: list[MetricSample] = []

    def _process_context(self, context: Mapping[str, Any]) -> None:
        ego_speed = float(context.get("ego_speed", 0.0)) / 3.6

        if self._previous_speed is None:
            acceleration = 0.0
        else:
            acceleration = (ego_speed - self._previous_speed) / self.dt

        self._previous_speed = ego_speed

        if acceleration <= self.deceleration_threshold and not self._in_hard_brake:
            self._count += 1
            self._in_hard_brake = True
        elif acceleration >= self.reset_threshold:
            self._in_hard_brake = False

        self._samples.append(self._make_sample(self._count))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return (MetricSeries(name="hard_brake_count", samples=tuple(self._samples)),)

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Hard Brake Count",
            series_names=("hard_brake_count",),
            summary_specs=(MetricSummarySpec(series_name="hard_brake_count", display_name="Hard Brake Count"),),
        )
