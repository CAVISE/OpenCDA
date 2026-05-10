"""Scenario-level collision counter metric."""

from typing import Any, Mapping

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.metric_sample import MetricSample


class CollisionCountMetric(BaseMetric):  # noqa DC03
    """Count vehicles that reported at least one collision."""

    metric_name = "collision_count"

    def __init__(self, warmup_steps: int = 0):
        super().__init__(warmup_steps=warmup_steps)
        self._collided_node_ids: set[str] = set()
        self._count = 0
        self._samples: list[MetricSample] = []

    def _process_context(self, context: Mapping[str, Any]) -> None:
        vehicles = context.get("vehicles", ())
        current_collisions = {
            str(vehicle.get("node_id", "")) for vehicle in vehicles if isinstance(vehicle, Mapping) and bool(vehicle.get("collided", False))
        }
        current_collisions.discard("")

        new_collisions = current_collisions - self._collided_node_ids
        self._count += len(new_collisions)
        self._collided_node_ids.update(current_collisions)
        self._samples.append(self._make_sample(self._count))
