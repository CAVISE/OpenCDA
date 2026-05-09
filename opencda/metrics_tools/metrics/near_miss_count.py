"""Scenario-level near-miss counter metric."""

from itertools import combinations
from math import inf, sqrt
from typing import Any, Mapping

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec


class NearMissCountMetric(BaseMetric):
    """Count pairwise near-miss episodes between CAVs."""

    metric_name = "near_miss_count"

    def __init__(self, warmup_steps: int = 0, distance_threshold: float = 2.0):
        super().__init__(warmup_steps=warmup_steps)
        self.distance_threshold = distance_threshold
        self._active_pairs: set[tuple[str, str]] = set()
        self._count = 0
        self._count_samples: list[MetricSample] = []
        self._min_distance_samples: list[MetricSample] = []

    def _process_context(self, context: Mapping[str, Any]) -> None:
        vehicles = tuple(vehicle for vehicle in context.get("vehicles", ()) if isinstance(vehicle, Mapping))
        active_pairs: set[tuple[str, str]] = set()
        min_distance = inf

        for first, second in combinations(vehicles, 2):
            first_id = str(first.get("node_id", ""))
            second_id = str(second.get("node_id", ""))
            if not first_id or not second_id:
                continue
            distance = self._distance(first, second)
            min_distance = min(min_distance, distance)

            if distance <= self.distance_threshold:
                pair = (first_id, second_id) if first_id <= second_id else (second_id, first_id)
                active_pairs.add(pair)

        new_pairs = active_pairs - self._active_pairs
        self._count += len(new_pairs)
        self._active_pairs = active_pairs

        self._count_samples.append(self._make_sample(self._count))
        if min_distance < inf:
            self._min_distance_samples.append(self._make_sample(min_distance))

    @staticmethod
    def _distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
        dx = float(first.get("x", 0.0)) - float(second.get("x", 0.0))
        dy = float(first.get("y", 0.0)) - float(second.get("y", 0.0))
        dz = float(first.get("z", 0.0)) - float(second.get("z", 0.0))
        return sqrt(dx * dx + dy * dy + dz * dz)

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return (
            MetricSeries(name="near_miss_count", samples=tuple(self._count_samples)),
            MetricSeries(name="min_distance_between_cavs", samples=tuple(self._min_distance_samples)),
        )

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Near Miss Count",
            series_names=("near_miss_count", "min_distance_between_cavs"),
            summary_specs=(
                MetricSummarySpec(series_name="near_miss_count", display_name="Near Miss Count"),
                MetricSummarySpec(series_name="min_distance_between_cavs", display_name="Min Distance Between CAVs"),
            ),
        )
