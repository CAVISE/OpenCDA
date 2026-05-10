"""Scenario-level identity-conflict counter metric."""

from collections import defaultdict
from typing import Any, Mapping

from opencda.core.application.behavior.transport_message import BROADCAST_OWNER_ID
from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec


class IdentityConflictCountMetric(BaseMetric):  # noqa DC03
    """Count episodes where one claimed identity is emitted by multiple physical nodes."""

    metric_name = "identity_conflict_count"

    def __init__(self, warmup_steps: int = 0):
        super().__init__(warmup_steps=warmup_steps)
        self._active_conflicts: set[str] = set()
        self._count = 0
        self._samples: list[MetricSample] = []

    def _process_context(self, context: Mapping[str, Any]) -> None:
        claims_by_identity: dict[str, set[str]] = defaultdict(set)
        for claim in context.get("identity_claims", ()):
            if not isinstance(claim, Mapping):
                continue

            producer_node_id = str(claim.get("producer_node_id", ""))
            claimed_node_id = str(claim.get("claimed_node_id", ""))
            if not producer_node_id or not claimed_node_id or claimed_node_id == BROADCAST_OWNER_ID:
                continue

            claims_by_identity[claimed_node_id].add(producer_node_id)

        current_conflicts = {claimed_node_id for claimed_node_id, producers in claims_by_identity.items() if len(producers) > 1}
        new_conflicts = current_conflicts - self._active_conflicts

        self._count += len(new_conflicts)
        self._active_conflicts = current_conflicts
        self._samples.append(self._make_sample(self._count))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return (MetricSeries(name="identity_conflict_count", samples=tuple(self._samples)),)

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Identity Conflict Count",
            series_names=("identity_conflict_count",),
            summary_specs=(MetricSummarySpec(series_name="identity_conflict_count", display_name="Identity Conflict Count"),),
        )
