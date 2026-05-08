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

# --- BLOCKED IMPORTS (require carla/torch/traci via transitive chain) ---
# from opencda.core.attack.adversary_framework.models import (
#     AttackSpec,
#     AttackStageResult,
#     RuntimeStatus,
#     Status,
# )
# from opencda.core.attack.adversary_framework.attack import Attack
# from opencda.core.attack.adversary_framework.attack_manager import AttackManager
# from opencda.core.attack.adversary_framework.stages.sniffer.stage import SnifferStage
# from opencda.core.attack.adversary_framework.stages.response_replayer.stage import (
#     ResponseReplayerStage,
# )
# from opencda.core.application.behavior.capability import Capability


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


# ---------------------------------------------------------------------------
# INT-02: Full sniffer + replayer attack cycle  [BLOCKED: needs carla/torch]
# ---------------------------------------------------------------------------


# class TestIntegrationAttack:
#     """Integration test for the full sniffer + replayer attack cycle."""

#     def test_int_02_sniffer_replayer_full_cycle(self):
#         """Full attack cycle: sniffer collects output, replayer reproduces previous."""
#         from opencda.core.attack.adversary_framework.models import (
#             ConditionSpec,
#             StageSpec,
#             TargetSpec,
#             TriggerSourceSpec,
#         )

#         # Create sniffer and replayer services
#         def sniffer_observe():
#             return {"perception": "current_frame"}

#         sniffer_observe.__name__ = "response_observe"

#         sniffer_service = MagicMock()
#         sniffer_service.service_type = "aim_client"
#         sniffer_service.capability_bindings = {
#             Capability.RESPONSE_OBSERVE: sniffer_observe,
#         }
#         sniffer_service.response_observe = sniffer_observe

#         def replayer_submit():
#             return {"control": "current_control"}

#         replayer_submit.__name__ = "response_submit"

#         replayer_service = MagicMock()
#         replayer_service.service_type = "aim_server"
#         replayer_service.capability_bindings = {
#             Capability.RESPONSE_SUBMIT: replayer_submit,
#         }
#         replayer_service.response_submit = replayer_submit

#         # Build attack spec with sniffer + replayer stages
#         spec = AttackSpec(
#             name="sniff_replay_attack",
#             targets=TargetSpec(
#                 kind="service_state_field",
#                 source=TriggerSourceSpec(
#                     kind="snapshot",
#                     node_type="vehicle",
#                     service_type="aim_client",
#                     field="tracked_ids",
#                 ),
#                 resolve_to_node_type="vehicle",
#                 resolve_to_service_name="aim_client",
#             ),
#             start_trigger=ConditionSpec(
#                 source=TriggerSourceSpec(
#                     kind="snapshot",
#                     service_type="aim_client",
#                     field="start_flag",
#                 ),
#                 verb="eq",
#                 value=True,
#             ),
#             stages=(
#                 StageSpec(
#                     id="sniff_stage",
#                     type="sniffer",
#                 ),
#             ),
#         )

#         sniffer_stage = SnifferStage()
#         attack = Attack(spec=spec, stages=(sniffer_stage,))
#         manager = AttackManager()

#         # Snapshot with target
#         snapshot = _make_snapshot(
#             vehicle_nodes=(
#                 _make_node(
#                     node_id="v1",
#                     service_states={
#                         "aim_client": {"tracked_ids": "v1", "start_flag": True},
#                     },
#                 ),
#             ),
#         )

#         service_resolver = lambda node_id, svc_type: sniffer_service

#         # Tick 1: Start attack and run sniffer
#         results = manager.evaluate((attack,), snapshot, service_resolver=service_resolver)

#         # Sniffer should have installed interceptors
#         assert len(sniffer_stage._restore_callbacks) > 0
#         assert len(sniffer_stage.observed_outputs) == 0  # No calls yet

#         # Simulate the service being called (interceptor captures output)
#         output = sniffer_service.response_observe()
#         assert len(sniffer_stage.observed_outputs) == 1
#         assert sniffer_stage.observed_outputs[0].output == {"perception": "current_frame"}

#         # Cleanup
#         sniffer_stage.deactivate()


# ---------------------------------------------------------------------------
# INT-03: Load real YAML config  [BLOCKED: needs carla/torch + YAML fix]
# ---------------------------------------------------------------------------


# class TestIntegrationConfig:
#     """Integration test for loading real YAML configs."""

#     def test_int_03_load_yaml_create_attack(self):
#         """Loading real YAML config (aim_client_response_sniffer) and creating
#         Attack via AttackSpec.from_dict.
#         """
#         config_path = Path(__file__).resolve().parents[1] / (
#             "opencda/core/attack/adversary_framework/"
#             "attacks/aim_client_response_sniffer/config.yaml"
#         )
#         assert config_path.exists(), f"Config file not found: {config_path}"

#         with open(config_path) as f:
#             config = yaml.safe_load(f)

#         attack_data = config["attack"]
#         spec = AttackSpec.from_dict(attack_data)

#         assert spec.name == "aim_client_response_sniffer"
#         assert spec.start_trigger is not None
#         assert spec.stop_trigger is not None
#         assert spec.targets is not None
#         assert len(spec.stages) == 1
#         assert spec.stages[0].type == "sniffer"
#         assert spec.stages[0].stage_start_trigger is not None

#         # Verify nested conditions are properly parsed
#         assert spec.requirements is not None
#         assert len(spec.requirements.all) == 2

#         # Create the Attack object
#         attack = Attack.from_spec(spec)
#         assert attack.attack_name == "aim_client_response_sniffer"
#         assert len(attack.stage_runtimes) == 1
#         assert isinstance(attack.stage_runtimes[0].stage, SnifferStage)

#     # --- ALSO NEEDS YAML FIX: service_name -> service_type in config.yaml ---
#     def test_int_03_load_server_replay_config(self):
#         """Loading aim_server_response_replay config.

#         The real YAML uses 'service_name' instead of 'service_type' in some
#         fields, which causes a KeyError in TargetSpec.from_dict. We verify
#         that the config is loadable after normalizing those keys.
#         """
#         config_path = Path(__file__).resolve().parents[1] / (
#             "opencda/core/attack/adversary_framework/"
#             "attacks/aim_server_response_replay/config.yaml"
#         )
#         assert config_path.exists(), f"Config file not found: {config_path}"

#         with open(config_path) as f:
#             config = yaml.safe_load(f)

#         attack_data = config["attack"]

#         # The YAML uses 'service_name' instead of 'service_type'.
#         # Normalize so it is compatible with TriggerSourceSpec / TargetSpec.
#         for key in ("source", "start_trigger", "stop_trigger"):
#             node = attack_data.get(key)
#             if isinstance(node, dict) and "service_name" in node:
#                 node["service_type"] = node.pop("service_name")

#         if attack_data.get("stages"):
#             for stage in attack_data["stages"]:
#                 for cond_key in ("requirements", "stage_start_trigger", "stage_stop_trigger"):
#                     cond = stage.get(cond_key)
#                     if isinstance(cond, dict):
#                         src = cond.get("source")
#                         if isinstance(src, dict) and "service_name" in src:
#                             src["service_type"] = src.pop("service_name")

#         targets = attack_data.get("targets")
#         if isinstance(targets, dict):
#             src = targets.get("source")
#             if isinstance(src, dict) and "service_name" in src:
#                 src["service_type"] = src.pop("service_name")
#             resolve_to = targets.get("resolve_to", {})
#             if "service_name" in resolve_to:
#                 resolve_to["service_type"] = resolve_to.pop("service_name")

#         spec = AttackSpec.from_dict(attack_data)
#         assert spec.name == "aim_server_response_replay"
#         assert len(spec.stages) == 1
#         assert spec.stages[0].type == "response_replayer"
#         assert spec.stages[0].stage_stop_trigger is not None

#         attack = Attack.from_spec(spec)
#         assert attack.attack_name == "aim_server_response_replay"
#         assert isinstance(attack.stage_runtimes[0].stage, ResponseReplayerStage)
