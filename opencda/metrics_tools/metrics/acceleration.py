"""Acceleration metric implementation."""

from typing import Mapping, Any

from opencda.metrics_tools.base_metric import BaseMetric


class AccelerationMetric(BaseMetric):  # noqa DC03
    """Metric for ego acceleration."""

    metric_name = "acceleration"

    def __init__(self, warmup_steps: int = 100, dt: float = 0.05):
        super().__init__(warmup_steps=warmup_steps, sample_interval=dt)
        self.dt = dt
        self._previous_speed: float | None = None

    @property  # noqa DC08
    def acceleration_list(self) -> list[float]:
        return [sample.value for sample in self._samples]

    def _process_context(self, context: Mapping[str, Any]) -> None:
        ego_speed = float(context.get("ego_speed", 0.0)) / 3.6

        if self._previous_speed is None:
            acceleration_value = 0.0
        else:
            acceleration_value = (ego_speed - self._previous_speed) / self.dt

        self._previous_speed = ego_speed
        self._record_sample(acceleration_value)
