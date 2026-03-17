import numpy as np
from typing import Mapping

from opencda.core.plan.metrics.base_metric import BaseMetric
from opencda.core.plan.metrics.metric_sample import MetricSample
from opencda.core.plan.report_models import MetricSeries


class DynamicsMetric(BaseMetric):
    """
    Metric for Speed and Acceleration.

    Parameters
    ----------
    warmup_steps : int
        The number of steps to ignore at the beginning.
    dt : float
        Time step interval for acceleration calculation.
    """

    metric_name = "dynamics"
    module = "planning"

    def __init__(self, warmup_steps: int = 100, dt: float = 0.05):
        super().__init__(warmup_steps=warmup_steps, sample_interval=dt)
        self.dt = dt
        self._speed_samples: list[MetricSample] = []
        self._acc_samples: list[MetricSample] = []

    @property
    def speed_list(self) -> list[float]:
        return [sample.value for sample in self._speed_samples]

    @property
    def acc_list(self) -> list[float]:
        return [sample.value for sample in self._acc_samples]

    def _process_context(self, context: Mapping[str, object]) -> None:
        ego_speed = float(context.get("ego_speed", 0.0))
        speed_sample = self._make_sample(ego_speed / 3.6)
        self._speed_samples.append(speed_sample)

        if len(self._speed_samples) <= 1:
            acc_value = 0.0
        else:
            previous_speed = self._speed_samples[-2].value
            acc_value = (speed_sample.value - previous_speed) / self.dt

        self._acc_samples.append(
            MetricSample(
                tick=speed_sample.tick,
                value=acc_value,
            )
        )

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return (
            MetricSeries(name="speed", samples=tuple(self._speed_samples)),
            MetricSeries(name="acceleration", samples=tuple(self._acc_samples)),
        )

    def speed_statistics(self) -> tuple[float, float]:
        speed_array = np.array(self.speed_list, dtype=float)
        return float(np.mean(speed_array)), float(np.std(speed_array))

    def acceleration_statistics(self) -> tuple[float, float]:
        acc_array = np.array(self.acc_list, dtype=float)
        return float(np.mean(acc_array)), float(np.std(acc_array))
