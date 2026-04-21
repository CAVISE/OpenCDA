"""Time-to-collision metric implementation."""

import numpy as np
from typing import Mapping, Any

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec


class TtcMetric(BaseMetric):  # noqa DC03
    """
    !Should be rewrited to be more generic for any scalar metric.!
    Metric for Time To Collision (TTC).

    Parameters
    ----------
    warmup_steps : int
        The number of steps to ignore at the beginning.
    """

    metric_name = "ttc"

    def __init__(self, warmup_steps: int = 100):
        super().__init__(warmup_steps=warmup_steps)
        self._ttc_samples: list[MetricSample] = []

    @property
    def ttc_list(self) -> list[float]:
        return [sample.value for sample in self._ttc_samples]

    def _process_context(self, context: Mapping[str, Any]) -> None:
        ttc = float(context.get("ttc", 1000.0))
        self._ttc_samples.append(self._make_sample(ttc))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return (MetricSeries(name="ttc", samples=tuple(self._ttc_samples)),)

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Time To Collision",
            series_names=("ttc",),
            summary_specs=(MetricSummarySpec(series_name="ttc", cutoff=1000.0),),
        )

    def valid_statistics(self, cutoff: float = 1000.0) -> tuple[float | None, float | None]:  # noqa DC04
        ttc_array = np.array(self.ttc_list, dtype=float)
        ttc_array = ttc_array[ttc_array < cutoff]
        if len(ttc_array) == 0:
            return None, None
        return float(np.mean(ttc_array)), float(np.std(ttc_array))
