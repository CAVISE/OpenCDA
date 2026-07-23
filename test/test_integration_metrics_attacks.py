"""Integration tests for metrics_tools and adversary_framework.

Covers:
- INT-01: MetricCollector -> MetricCollection -> UniversalReportBuilder for speed
- INT-02: Full sniffer + replayer attack cycle with mock services  [BLOCKED: needs carla/torch]
- INT-03: Loading real YAML config and creating Attack via AttackSpec.from_dict  [BLOCKED: needs carla/torch + YAML fix]
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock


from opencda.scenario_testing.types import NodeSnapshot, SimulationSnapshot
from opencda.metrics_tools.metric_collector import MetricCollector
from opencda.metrics_tools.report_builder import UniversalReportBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    vehicle_nodes: tuple[NodeSnapshot, ...] | None = None,
    rsu_nodes: tuple[NodeSnapshot, ...] | None = None,
) -> SimulationSnapshot:
    return SimulationSnapshot(
        tick=1,
        vehicle_nodes=vehicle_nodes or (),
        rsu_nodes=rsu_nodes or (),
    )


def _make_node(
    node_id: str = "v1",
    node_type: str = "vehicle",
    service_states: dict[str, Any] | None = None,
) -> NodeSnapshot:
    return NodeSnapshot(
        node_id=node_id,
        node_type=node_type,
        service_states=service_states or {},
    )


def _make_mock_service(
    service_type: str = "test_service",
    capability_bindings: dict[Any, Any] | None = None,
) -> Any:
    """Create a mock BehaviorService-like object."""
    service = MagicMock()
    service.service_type = service_type
    service.capability_bindings = capability_bindings or {}
    return service


# ---------------------------------------------------------------------------
# INT-01: MetricCollector -> MetricCollection -> Report
# ---------------------------------------------------------------------------


class TestIntegrationMetrics:
    """Integration test for the full metrics pipeline."""

    def test_int_01_metric_collector_to_report(self):
        """MetricCollector -> MetricCollection -> UniversalReportBuilder.build_entity_report
        for speed metric produces correct summary and series.
        """
        collector = MetricCollector(
            module="localization",
            entity_id="vehicle_0",
            metric_configs={"speed": {"warmup_steps": 2, "dt": 0.05}},
        )

        # Feed data: first 2 calls are warmup, then data is collected
        for i in range(10):
            collector.update({"ego_speed": 72.0})  # 72 km/h = 20 m/s

        raw = collector.get_raw()
        assert raw.module == "localization"
        assert raw.entity_id == "vehicle_0"

        # Build report
        builder = UniversalReportBuilder()
        report = builder.build_entity_report(raw)

        assert report.info.module == "localization"
        assert report.info.entity_id == "vehicle_0"
        assert "speed" in report.info.active_metrics

        # Should have one metric report (speed)
        assert len(report.metrics) == 1
        metric_report = report.metrics[0]
        assert metric_report.metric_name == "speed"

        # Summary should have correct values
        assert len(metric_report.summary) == 1
        summary = metric_report.summary[0]
        assert summary.count == 8  # 10 - 2 warmup
        assert math.isclose(summary.mean, 20.0, rel_tol=1e-6)  # 72/3.6 = 20 m/s

        # Series should have the speed data
        assert len(metric_report.series) == 1
        assert metric_report.series[0].name == "speed"
        assert len(metric_report.series[0].samples) == 8
