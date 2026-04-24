"""Intersection crossing time"""

from typing import Mapping, Any

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec
from opencda.metrics_tools.metric_sample import MetricSample

import time


class CrossingTimeMetric(BaseMetric):
    """Metric for AIM"""

    metric_name = "crossing_time"

    def __init__(self, warmup_steps: int = 100):
        super().__init__(warmup_steps=warmup_steps)
        self._samples: list[MetricSample] = []
        self.intersection_enter_time: dict[Any, float] = {}
        self.at_intersection: dict[Any, bool] = {}

    def _process_context(self, context: Mapping[str, Any]) -> None:
        inst: list[dict] = context.get("at_intersection", [{"id": 0, "crossing": False}])
        for el in inst:
            if "id" not in el or "crossing" not in el:
                continue
            cav_id = el["id"]
            cur_at_intersection = el["crossing"]
            prev_on_intersection = self.at_intersection.get(cav_id, False)
            if cur_at_intersection and not prev_on_intersection:
                self.intersection_enter_time[cav_id] = time.time()
            elif not cur_at_intersection and prev_on_intersection:
                self._samples.append(self._make_sample(time.time() - self.intersection_enter_time[cav_id]))
            self.at_intersection[cav_id] = cur_at_intersection

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return (MetricSeries(name="crossing_time", samples=tuple(self._samples)),)

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Crossing Time",
            series_names=("crossing_time",),
            summary_specs=(
                MetricSummarySpec(
                    series_name="crossing_time",
                    cutoff=100.0,
                ),
            ),
        )