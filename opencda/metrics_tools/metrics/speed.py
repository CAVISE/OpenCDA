"""Speed metric implementation."""

from typing import Mapping, Any

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.metric_sample import MetricSample


class SpeedMetric(BaseMetric):  # noqa DC03
    """Metric for ego speed."""

    metric_name = "speed"

    def __init__(self, warmup_steps: int = 100, dt: float = 0.05):
        super().__init__(warmup_steps=warmup_steps, sample_interval=dt)
        self._samples: list[MetricSample] = []

    @property  # noqa DC08
    def speed_list(self) -> list[float]:
        return [sample.value for sample in self._samples]

    def _process_context(self, context: Mapping[str, Any]) -> None:
        ego_speed = float(context.get("ego_speed", 0.0))
        self._samples.append(self._make_sample(ego_speed / 3.6))
