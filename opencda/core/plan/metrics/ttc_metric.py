import numpy as np
from typing import Mapping

from opencda.core.plan.metrics.base_metric import BaseMetric
from opencda.core.plan.metrics.metric_sample import MetricSample
from opencda.core.plan.report_models import MetricSeries


class TtcMetric(BaseMetric):
    """
    Metric for Time To Collision (TTC).

    Parameters
    ----------
    warmup_steps : int
        The number of steps to ignore at the beginning.
    """

    metric_name = "ttc"
    module = "planning"

    def __init__(self, warmup_steps: int = 100):
        super().__init__(warmup_steps=warmup_steps)
        self._ttc_samples: list[MetricSample] = []

    @property
    def ttc_list(self) -> list[float]:
        return [sample.value for sample in self._ttc_samples]

    def _process_context(self, context: Mapping[str, object]) -> None:
        ttc = float(context.get("ttc", 1000.0))
        self._ttc_samples.append(self._make_sample(ttc))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return (MetricSeries(name="ttc", samples=tuple(self._ttc_samples)),)

    def valid_statistics(self, cutoff: float = 1000.0) -> tuple[float | None, float | None]:
        ttc_array = np.array(self.ttc_list, dtype=float)
        ttc_array = ttc_array[ttc_array < cutoff]
        if len(ttc_array) == 0:
            return None, None
        return float(np.mean(ttc_array)), float(np.std(ttc_array))
