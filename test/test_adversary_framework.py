"""Unit tests for opencda.core.attack.adversary_framework.

Covers attack_spec models, condition_evaluator, attack lifecycle,
attack_manager, utils, sniffer stage, and replayer stage.

BLOCKED: All tests in this file require carla/torch/traci (transitive imports).
        Import chain: models -> response_replayer/stage.py -> behavior_service_protocol
        -> behavior/__init__.py -> services/aim_client -> traci -> carla.
        TODO: Add conftest.py with mock imports or refactor __init__.py to fix.
        Tests are commented out until the dependency issue is resolved.
"""

from __future__ import annotations

# from copy import deepcopy
# from typing import Any
# from unittest.mock import MagicMock

# import pytest

# --- BLOCKED IMPORTS (require carla/torch/traci) ---
# from opencda.scenario_testing.types import NodeSnapshot, SimulationSnapshot
# from opencda.core.attack.adversary_framework.models import (
#     AttackSpec,
#     AttackStageResult,
#     ConditionSpec,
#     RuntimeStatus,
#     StageRuntime,
#     StageSpec,
#     Status,
#     TargetSpec,
#     TriggerSourceSpec,
# )
# from opencda.core.attack.adversary_framework.condition_evaluator import (
#     evaluate_condition,
#     normalize_target_node_ids,
# )
# from opencda.core.attack.adversary_framework.attack import Attack
# from opencda.core.attack.adversary_framework.attack_manager import AttackManager
# from opencda.core.attack.adversary_framework.utils import (
#     install_output_interceptor,
#     match_services,
#     resolve_targets,
#     wrap_method_output,
#     service_supports_capabilities,
# )
# from opencda.core.attack.adversary_framework.stages.sniffer.stage import SnifferStage
# from opencda.core.attack.adversary_framework.stages.sniffer.types import ObservedOutput
# from opencda.core.attack.adversary_framework.stages.response_replayer.stage import (
#     ResponseReplayerStage,
# )
# from opencda.core.application.behavior.capability import Capability


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------


# def _make_snapshot(
#     vehicle_nodes: tuple[NodeSnapshot, ...] | None = None,
#     rsu_nodes: tuple[NodeSnapshot, ...] | None = None,
# ) -> SimulationSnapshot:
#     return SimulationSnapshot(
#         tick=1,
#         vehicle_nodes=vehicle_nodes or (),
#         rsu_nodes=rsu_nodes or (),
#     )


# def _make_node(
#     node_id: str = "v1",
#     node_type: str = "vehicle",
#     service_states: dict[str, Any] | None = None,
# ) -> NodeSnapshot:
#     return NodeSnapshot(
#         node_id=node_id,
#         node_type=node_type,
#         service_states=service_states or {},
#     )


# def _make_leaf_condition(
#     kind: str = "snapshot",
#     node_type: str = "vehicle",
#     field: str = "f",
#     verb: str = "eq",
#     value: Any = None,
#     service_type: str | None = "svc",
#     attack_name: str | None = None,
#     stage_id: str | None = None,
# ) -> ConditionSpec:
#     return ConditionSpec(
#         source=TriggerSourceSpec(
#             kind=kind,
#             node_type=node_type,
#             field=field,
#             service_type=service_type,
#             attack_name=attack_name,
#             stage_id=stage_id,
#         ),
#         verb=verb,
#         value=value,
#     )


# def _make_mock_service(
#     service_type: str = "test_service",
#     capability_bindings: dict[Capability, Any] | None = None,
# ) -> Any:
#     """Create a mock BehaviorService-like object."""
#     service = MagicMock()
#     service.service_type = service_type
#     service.capability_bindings = capability_bindings or {}
#     return service


# def _make_simple_attack(
#     name: str = "test_attack",
#     start_trigger: ConditionSpec | None = None,
#     stop_trigger: ConditionSpec | None = None,
#     stage_specs: tuple[StageSpec, ...] | None = None,
# ) -> tuple[Attack, Any]:
#     """Create an Attack with a single mock stage.

#     Returns (attack, mock_stage).
#     """
#     if stage_specs is None:
#         stage_specs = (StageSpec(id="stage_0", type="mock"),)

#     spec = AttackSpec(
#         name=name,
#         start_trigger=start_trigger,
#         stop_trigger=stop_trigger,
#         stages=stage_specs,
#     )

#     mock_stage = MagicMock()
#     mock_stage.stage_name = "mock_stage"
#     mock_stage.required_capabilities = ()
#     mock_stage.execute.return_value = AttackStageResult(
#         stage_name="mock_stage",
#         status=Status.SUCCESS,
#         reason="ok",
#     )

#     attack = Attack(spec=spec, stages=(mock_stage,))
#     return attack, mock_stage


# # ---------------------------------------------------------------------------
# # A-SPEC: attack_spec.py  (NEEDS: carla/torch/traci)
# # ---------------------------------------------------------------------------


# class TestConditionSpec:
#     """Tests for ConditionSpec validation."""

#     def test_a_spec_01_leaf_and_all_raises(self):
#         """ConditionSpec with leaf predicate and all group raises ValueError."""
#         with pytest.raises(ValueError, match="both a leaf predicate"):
#             ConditionSpec(
#                 source=TriggerSourceSpec(kind="snapshot", field="f"),
#                 verb="eq",
#                 all=(ConditionSpec(source=TriggerSourceSpec(kind="snapshot", field="g"), verb="eq"),),
#             )

#     def test_a_spec_02_all_and_any_raises(self):
#         """ConditionSpec with both all and any raises ValueError."""
#         with pytest.raises(ValueError, match="both 'all' and 'any'"):
#             ConditionSpec(
#                 all=(ConditionSpec(source=TriggerSourceSpec(kind="snapshot", field="f"), verb="eq"),),
#                 any=(ConditionSpec(source=TriggerSourceSpec(kind="snapshot", field="g"), verb="eq"),),
#             )

#     def test_a_spec_03_leaf_without_source_raises(self):
#         """Leaf ConditionSpec without source raises ValueError."""
#         with pytest.raises(ValueError, match="must define 'source'"):
#             ConditionSpec(verb="eq")

#     def test_a_spec_04_leaf_without_verb_raises(self):
#         """Leaf ConditionSpec without verb raises ValueError."""
#         with pytest.raises(ValueError, match="must define 'verb'"):
#             ConditionSpec(source=TriggerSourceSpec(kind="snapshot", field="f"))

#     def test_a_spec_05_empty_spec_raises(self):
#         """Empty ConditionSpec (no leaf, no groups) raises ValueError."""
#         with pytest.raises(ValueError, match="must define either"):
#             ConditionSpec()

#     def test_a_spec_06_from_dict_attack_spec(self):
#         """from_dict creates AttackSpec with matching fields."""
#         data = {
#             "name": "my_attack",
#             "start_trigger": {
#                 "source": {"kind": "snapshot", "field": "x"},
#                 "verb": "eq",
#                 "value": 1,
#             },
#             "stop_trigger": {
#                 "source": {"kind": "snapshot", "field": "x"},
#                 "verb": "eq",
#                 "value": 0,
#             },
#             "stages": [
#                 {"id": "s1", "type": "sniffer"},
#                 {"id": "s2", "type": "response_replayer"},
#             ],
#         }
#         spec = AttackSpec.from_dict(data)
#         assert spec.name == "my_attack"
#         assert spec.start_trigger is not None
#         assert spec.start_trigger.verb == "eq"
#         assert spec.start_trigger.value == 1
#         assert len(spec.stages) == 2
#         assert spec.stages[0].id == "s1"
#         assert spec.stages[1].type == "response_replayer"

#     def test_a_spec_07_from_dict_stages_and_targets(self):
#         """from_dict creates StageSpec, TriggerSourceSpec, and TargetSpec."""
#         data = {
#             "name": "atk",
#             "targets": {
#                 "kind": "service_state_field",
#                 "source": {"kind": "snapshot", "node_type": "vehicle", "field": "ids"},
#                 "resolve_to": {"node_type": "vehicle", "service_type": "aim_client"},
#             },
#             "stages": [
#                 {
#                     "id": "s1",
#                     "type": "sniffer",
#                     "requirements": {
#                         "source": {"kind": "snapshot", "field": "x"},
#                         "verb": "exists",
#                     },
#                     "stage_start_trigger": {
#                         "source": {"kind": "snapshot", "field": "y"},
#                         "verb": "eq",
#                         "value": 1,
#                     },
#                 },
#             ],
#         }
#         spec = AttackSpec.from_dict(data)
#         assert spec.targets is not None
#         assert spec.targets.kind == "service_state_field"
#         assert spec.targets.resolve_to_node_type == "vehicle"
#         assert spec.stages[0].requirements is not None
#         assert spec.stages[0].stage_start_trigger is not None

#     def test_a_spec_08_from_dict_nested_all_any(self):
#         """from_dict handles nested all/any recursively."""
#         data = {
#             "all": [
#                 {"source": {"kind": "snapshot", "field": "a"}, "verb": "exists"},
#                 {
#                     "any": [
#                         {"source": {"kind": "snapshot", "field": "b"}, "verb": "eq", "value": 1},
#                         {"source": {"kind": "snapshot", "field": "c"}, "verb": "gt", "value": 0},
#                     ],
#                 },
#             ],
#         }
#         spec = ConditionSpec.from_dict(data)
#         assert len(spec.all) == 2
#         assert len(spec.all[1].any) == 2

#     def test_a_spec_09_normalize_target_node_ids(self):
#         """normalize_target_node_ids handles string, collection, and other types."""
#         assert normalize_target_node_ids("v1") == {"v1"}
#         assert normalize_target_node_ids(["v1", "v2"]) == {"v1", "v2"}
#         assert normalize_target_node_ids(("v1", "v3")) == {"v1", "v3"}
#         assert normalize_target_node_ids({"v1", "v2"}) == {"v1", "v2"}
#         assert normalize_target_node_ids(123) == set()
#         assert normalize_target_node_ids(None) == set()


# # ---------------------------------------------------------------------------
# # A-COND: condition_evaluator.py  (NEEDS: carla/torch/traci)
# # ---------------------------------------------------------------------------


# class TestConditionEvaluator:
#     """Tests for condition evaluation."""

#     def _make_runtime(
#         self,
#         status: RuntimeStatus = RuntimeStatus.INACTIVE,
#         previous_status: RuntimeStatus = RuntimeStatus.INACTIVE,
#         attack_name: str = "atk",
#         stage_runtimes: tuple[StageRuntime, ...] | None = None,
#     ) -> Any:
#         runtime = MagicMock()
#         runtime.attack_name = attack_name
#         runtime.status = status
#         runtime.previous_status = previous_status
#         runtime.stage_runtimes = stage_runtimes or ()
#         return runtime

#     def test_a_cond_01_verb_eq_true(self):
#         """verb eq returns True when values match."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="eq", value=10, field="f")
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 10}}),)
#         )
#         assert evaluate_condition(runtime, condition, None, snapshot) is True

#     def test_a_cond_02_verb_gt(self):
#         """verb gt returns True when current > expected."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="gt", value=5, field="f")
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 10}}),)
#         )
#         assert evaluate_condition(runtime, condition, None, snapshot) is True

#     def test_a_cond_03_verb_gte(self):
#         """verb gte: 5 >= 5 is True, 4 >= 5 is False."""
#         runtime = self._make_runtime()

#         cond_true = _make_leaf_condition(verb="gte", value=5, field="f")
#         snap_true = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 5}}),)
#         )
#         assert evaluate_condition(runtime, cond_true, None, snap_true) is True

#         cond_false = _make_leaf_condition(verb="gte", value=5, field="f")
#         snap_false = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 4}}),)
#         )
#         assert evaluate_condition(runtime, cond_false, None, snap_false) is False

#     def test_a_cond_04_verbs_lt_lte(self):
#         """verb lt: 5 < 10 True; lte: 5 <= 5 True."""
#         runtime = self._make_runtime()

#         lt_cond = _make_leaf_condition(verb="lt", value=10, field="f")
#         snap = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 5}}),)
#         )
#         assert evaluate_condition(runtime, lt_cond, None, snap) is True

#         lte_cond = _make_leaf_condition(verb="lte", value=5, field="f")
#         assert evaluate_condition(runtime, lte_cond, None, snap) is True

#     def test_a_cond_05_verb_exists(self):
#         """verb exists returns True/False based on field presence."""
#         runtime = self._make_runtime()

#         cond_exists = _make_leaf_condition(verb="exists", field="f")
#         snap_present = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )
#         assert evaluate_condition(runtime, cond_exists, None, snap_present) is True

#         snap_absent = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {}}),)
#         )
#         assert evaluate_condition(runtime, cond_exists, None, snap_absent) is False

#     def test_a_cond_06_verb_changed(self):
#         """verb changed returns True when values differ."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="changed", field="f")

#         prev = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 1}}),)
#         )
#         curr = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 2}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr) is True

#         curr_same = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 1}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr_same) is False

#     def test_a_cond_07_verb_became(self):
#         """verb became: True if prev != value and curr == value."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="became", value=42, field="f")

#         prev = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 1}}),)
#         )
#         curr = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 42}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr) is True

#         curr_wrong = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 1}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr_wrong) is False

#     def test_a_cond_08_verb_increased_no_delta(self):
#         """verb increased (no delta): True when prev < curr."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="increased", field="f")

#         prev = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 5}}),)
#         )
#         curr = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 10}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr) is True

#     def test_a_cond_09_verb_decreased_no_delta(self):
#         """verb decreased (no delta): True when prev > curr."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="decreased", field="f")

#         prev = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 10}}),)
#         )
#         curr = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 5}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr) is True

#     def test_a_cond_10_verb_added(self):
#         """verb added: True when specific element appears in set."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="added", value="item_b", field="f")

#         prev = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": ["item_a"]}}),)
#         )
#         curr = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": ["item_a", "item_b"]}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr) is True

#     def test_a_cond_11_verb_removed(self):
#         """verb removed: True when specific element disappears."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="removed", value="item_b", field="f")

#         prev = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": ["item_a", "item_b"]}}),)
#         )
#         curr = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": ["item_a"]}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr) is True

#     def test_a_cond_12_added_removed_none_value(self):
#         """added/removed with expected_value=None: True if any change."""
#         runtime = self._make_runtime()

#         added_cond = _make_leaf_condition(verb="added", value=None, field="f")
#         prev = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": ["a"]}}),)
#         )
#         curr = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": ["a", "b"]}}),)
#         )
#         assert evaluate_condition(runtime, added_cond, prev, curr) is True

#         removed_cond = _make_leaf_condition(verb="removed", value=None, field="f")
#         assert evaluate_condition(runtime, removed_cond, curr, prev) is True

#     def test_a_cond_13_verb_increased_with_delta(self):
#         """verb increased with delta: True if increased >= delta."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="increased", value=5, field="f")

#         prev = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 0}}),)
#         )
#         curr = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 10}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr) is True

#         curr_small = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 3}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr_small) is False

#     def test_a_cond_14_verb_decreased_with_delta(self):
#         """verb decreased with delta: True if decreased >= delta."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="decreased", value=5, field="f")

#         prev = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 10}}),)
#         )
#         curr = _make_snapshot(
#             vehicle_nodes=(_make_node(node_id="v1", service_states={"svc": {"f": 0}}),)
#         )
#         assert evaluate_condition(runtime, condition, prev, curr) is True

#     def test_a_cond_15_recursive_all_true(self):
#         """Recursive all (AND): all true -> True."""
#         runtime = self._make_runtime()
#         condition = ConditionSpec(
#             all=(
#                 _make_leaf_condition(verb="gt", value=0, field="f"),
#                 _make_leaf_condition(verb="exists", field="f"),
#             ),
#         )
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 10}}),)
#         )
#         assert evaluate_condition(runtime, condition, None, snapshot) is True

#     def test_a_cond_16_all_one_false(self):
#         """all (AND): one false -> False."""
#         runtime = self._make_runtime()
#         condition = ConditionSpec(
#             all=(
#                 _make_leaf_condition(verb="gt", value=100, field="f"),
#                 _make_leaf_condition(verb="exists", field="f"),
#             ),
#         )
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 10}}),)
#         )
#         assert evaluate_condition(runtime, condition, None, snapshot) is False

#     def test_a_cond_17_any_one_true(self):
#         """any (OR): one true -> True."""
#         runtime = self._make_runtime()
#         condition = ConditionSpec(
#             any=(
#                 _make_leaf_condition(verb="gt", value=100, field="f"),
#                 _make_leaf_condition(verb="exists", field="f"),
#             ),
#         )
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 10}}),)
#         )
#         assert evaluate_condition(runtime, condition, None, snapshot) is True

#     def test_a_cond_18_any_all_false(self):
#         """any (OR): all false -> False."""
#         runtime = self._make_runtime()
#         condition = ConditionSpec(
#             any=(
#                 _make_leaf_condition(verb="gt", value=100, field="f"),
#                 ConditionSpec(
#                     all=(_make_leaf_condition(verb="gt", value=200, field="f"),)
#                 ),
#             ),
#         )
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 10}}),)
#         )
#         assert evaluate_condition(runtime, condition, None, snapshot) is False

#     def test_a_cond_19_nested_all_any_deep_recursion(self):
#         """Deep nested all/any: any->True because gt->True."""
#         runtime = self._make_runtime()
#         condition = ConditionSpec(
#             any=(
#                 ConditionSpec(
#                     all=(
#                         _make_leaf_condition(verb="gt", value=0, field="f"),
#                     ),
#                 ),
#                 _make_leaf_condition(verb="eq", value=-1, field="f"),
#             ),
#         )
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 10}}),)
#         )
#         assert evaluate_condition(runtime, condition, None, snapshot) is True

#     def test_a_cond_20_prev_snapshot_none_changed(self):
#         """When prev_snapshot is None, changed is always True (MISSING != value)."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="changed", field="f")
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 10}}),)
#         )
#         assert evaluate_condition(runtime, condition, None, snapshot) is True

#     def test_a_cond_21_runtime_attack_status(self):
#         """Runtime source: condition reads attack status."""
#         runtime = self._make_runtime(status=RuntimeStatus.ACTIVE)
#         condition = ConditionSpec(
#             source=TriggerSourceSpec(kind="attack", field="status", attack_name="atk"),
#             verb="eq",
#             value="active",
#         )
#         snapshot = _make_snapshot()
#         assert evaluate_condition(runtime, condition, None, snapshot) is True

#     def test_a_cond_22_runtime_stage_status(self):
#         """Runtime source: condition reads stage status."""
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s1"
#         stage_spec = StageSpec(id="s1", type="sniffer")
#         stage_rt = StageRuntime(
#             spec=stage_spec,
#             stage=mock_stage,
#             status=RuntimeStatus.ACTIVE,
#         )
#         runtime = self._make_runtime(
#             stage_runtimes=(stage_rt,),
#         )
#         condition = ConditionSpec(
#             source=TriggerSourceSpec(kind="stage", field="status", stage_id="s1", attack_name="atk"),
#             verb="eq",
#             value="active",
#         )
#         snapshot = _make_snapshot()
#         assert evaluate_condition(runtime, condition, None, snapshot) is True

#     def test_a_cond_23_runtime_filter_attack_name(self):
#         """Runtime source: condition sees only the matching attack."""
#         runtime = self._make_runtime(status=RuntimeStatus.ACTIVE, attack_name="atk1")
#         condition = ConditionSpec(
#             source=TriggerSourceSpec(kind="attack", field="status", attack_name="atk_other"),
#             verb="eq",
#             value="active",
#         )
#         snapshot = _make_snapshot()
#         # Wrong attack_name -> no values collected -> exists=False
#         assert evaluate_condition(runtime, condition, None, snapshot) is False

#     def test_a_cond_24_runtime_filter_stage_id(self):
#         """Runtime source: condition sees only the matching stage."""
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s1"
#         stage_spec_1 = StageSpec(id="s1", type="sniffer")
#         stage_rt = StageRuntime(
#             spec=stage_spec_1,
#             stage=mock_stage,
#             status=RuntimeStatus.ACTIVE,
#         )
#         runtime = self._make_runtime(stage_runtimes=(stage_rt,))
#         condition = ConditionSpec(
#             source=TriggerSourceSpec(kind="stage", field="status", stage_id="s_other", attack_name="atk"),
#             verb="exists",
#         )
#         snapshot = _make_snapshot()
#         assert evaluate_condition(runtime, condition, None, snapshot) is False

#     def test_a_cond_25_unsupported_verb_raises(self):
#         """Unsupported verb raises ValueError."""
#         runtime = self._make_runtime()
#         condition = _make_leaf_condition(verb="unsupported_verb", field="f")
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 10}}),)
#         )
#         with pytest.raises(ValueError, match="Unsupported"):
#             evaluate_condition(runtime, condition, None, snapshot)

#     def test_a_cond_26_unsupported_source_kind_raises(self):
#         """Unsupported source kind raises ValueError."""
#         runtime = self._make_runtime()
#         condition = ConditionSpec(
#             source=TriggerSourceSpec(kind="invalid_kind", field="f", service_type="svc"),
#             verb="eq",
#             value=1,
#         )
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )
#         with pytest.raises(ValueError):
#             evaluate_condition(runtime, condition, None, snapshot)


# # ---------------------------------------------------------------------------
# # A-ATK: attack.py  (NEEDS: carla/torch/traci)
# # ---------------------------------------------------------------------------


# class TestAttack:
#     """Tests for Attack lifecycle."""

#     def test_a_atk_01_stage_count_mismatch_raises(self):
#         """ValueError when stage count differs from spec."""
#         spec = AttackSpec(
#             name="atk",
#             stages=(
#                 StageSpec(id="s0", type="mock"),
#                 StageSpec(id="s1", type="mock"),
#             ),
#         )
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s"
#         with pytest.raises(ValueError, match="stage specs"):
#             Attack(spec=spec, stages=(mock_stage,))

#     def test_a_atk_02_duplicate_stage_ids_raises(self):
#         """ValueError when stage IDs are duplicated."""
#         spec = AttackSpec(
#             name="atk",
#             stages=(
#                 StageSpec(id="dup", type="mock"),
#                 StageSpec(id="dup", type="mock"),
#             ),
#         )
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s"
#         with pytest.raises(ValueError, match="duplicate stage ids"):
#             Attack(spec=spec, stages=(mock_stage, mock_stage))

#     def test_a_atk_03_from_spec_factory(self):
#         """from_spec returns an Attack with stage_runtimes."""
#         spec = AttackSpec(
#             name="test_sniffer",
#             stages=(StageSpec(id="s0", type="sniffer"),),
#         )
#         attack = Attack.from_spec(spec)
#         assert attack.attack_name == "test_sniffer"
#         assert len(attack.stage_runtimes) == 1
#         assert attack.stage_runtimes[0].spec.id == "s0"

#     def test_a_atk_04_mark_transitions(self):
#         """mark_started/mark_active/mark_succeeded transition correctly."""
#         attack, _ = _make_simple_attack()

#         attack.mark_started()
#         assert attack.is_active is True
#         assert attack.status == RuntimeStatus.STARTED

#         attack.mark_active()
#         assert attack.is_active is True
#         assert attack.status == RuntimeStatus.ACTIVE

#         attack.mark_succeeded()
#         assert attack.is_active is False
#         assert attack.status == RuntimeStatus.SUCCESS

#     def test_a_atk_05_mark_failed_stopped(self):
#         """mark_failed / mark_stopped set correct state."""
#         attack, _ = _make_simple_attack()
#         attack.mark_active()

#         attack.mark_failed()
#         assert attack.is_active is False
#         assert attack.status == RuntimeStatus.FAIL

#         # Reset and test stopped
#         attack2, _ = _make_simple_attack()
#         attack2.mark_active()
#         attack2.mark_stopped()
#         assert attack2.is_active is False
#         assert attack2.status == RuntimeStatus.STOPPED

#     def test_a_atk_06_reset_runtime(self):
#         """reset_runtime resets all stages to INACTIVE."""
#         attack, mock_stage = _make_simple_attack()
#         attack.mark_started()
#         attack.mark_active()
#         attack.run_stage_lifecycle(None, _make_snapshot(), ())

#         attack.reset_runtime()
#         assert attack.status == RuntimeStatus.INACTIVE
#         assert attack.is_active is False
#         for sr in attack.stage_runtimes:
#             assert sr.status == RuntimeStatus.INACTIVE
#             assert sr.last_result is None

#     def test_a_atk_07_should_start_inactive_with_trigger(self):
#         """should_start returns True when inactive and triggers match."""
#         trigger = _make_leaf_condition(verb="exists")
#         attack, _ = _make_simple_attack(start_trigger=trigger)
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )
#         assert attack.should_start(None, snapshot) is True

#     def test_a_atk_08_should_start_already_active_false(self):
#         """should_start returns False when already active."""
#         trigger = _make_leaf_condition(verb="exists")
#         attack, _ = _make_simple_attack(start_trigger=trigger)
#         attack.mark_active()
#         assert attack.should_start(None, _make_snapshot()) is False

#     def test_a_atk_09_should_stop_active_with_trigger(self):
#         """should_stop returns True when active and stop trigger fires."""
#         stop_trigger = _make_leaf_condition(verb="eq", value=0)
#         attack, _ = _make_simple_attack(stop_trigger=stop_trigger)
#         attack.mark_active()
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 0}}),)
#         )
#         assert attack.should_stop(None, snapshot) is True

#     def test_a_atk_10_should_stop_inactive_false(self):
#         """should_stop returns False when not active."""
#         attack, _ = _make_simple_attack()
#         assert attack.should_stop(None, _make_snapshot()) is False

#     def test_a_atk_11_stage_launch_with_requirements_and_trigger(self):
#         """Stage launches when requirements and trigger are satisfied."""
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s0"
#         mock_stage.required_capabilities = ()
#         mock_stage.execute.return_value = AttackStageResult(stage_name="s0", status=Status.SUCCESS)

#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(
#                 StageSpec(
#                     id="s0",
#                     type="mock",
#                     requirements=_make_leaf_condition(verb="exists"),
#                 ),
#             ),
#         )
#         attack = Attack(spec=spec, stages=(mock_stage,))
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )

#         dummy_service = _make_mock_service()
#         attack.mark_started()
#         attack.mark_active()
#         results = attack.run_stage_lifecycle(None, snapshot, (dummy_service,))

#         assert len(results) > 0
#         mock_stage.execute.assert_called_once()

#     def test_a_atk_12_stage_not_launched_requirements_unmet(self):
#         """Stage stays INACTIVE when requirements are not met."""
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s0"
#         mock_stage.required_capabilities = ()

#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(
#                 StageSpec(
#                     id="s0",
#                     type="mock",
#                     requirements=_make_leaf_condition(verb="eq", value=999),
#                 ),
#             ),
#         )
#         attack = Attack(spec=spec, stages=(mock_stage,))
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )

#         attack.mark_started()
#         attack.mark_active()
#         results = attack.run_stage_lifecycle(None, snapshot, ())

#         mock_stage.execute.assert_not_called()
#         assert results == ()

#     def test_a_atk_13_stage_not_launched_trigger_unmet(self):
#         """Non-first stage stays INACTIVE when stage_start_trigger not met."""
#         mock_stage_0 = MagicMock()
#         mock_stage_0.stage_name = "s0"
#         mock_stage_0.required_capabilities = ()
#         mock_stage_0.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.SUCCESS
#         )

#         mock_stage_1 = MagicMock()
#         mock_stage_1.stage_name = "s1"
#         mock_stage_1.required_capabilities = ()

#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(
#                 StageSpec(id="s0", type="mock"),
#                 StageSpec(
#                     id="s1",
#                     type="mock",
#                     stage_start_trigger=_make_leaf_condition(verb="eq", value=999),
#                 ),
#             ),
#         )
#         attack = Attack(spec=spec, stages=(mock_stage_0, mock_stage_1))
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )

#         attack.mark_started()
#         attack.mark_active()
#         attack.run_stage_lifecycle(None, snapshot, ())

#         mock_stage_1.execute.assert_not_called()
#         assert attack.stage_runtimes[1].status == RuntimeStatus.INACTIVE

#     def test_a_atk_14_no_matching_services_fail(self):
#         """No matching services -> FAIL, attack marks failed."""
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s0"
#         mock_stage.required_capabilities = (Capability.RESPONSE_OBSERVE,)
#         mock_stage.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.SUCCESS
#         )

#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(StageSpec(id="s0", type="mock"),),
#         )
#         attack = Attack(spec=spec, stages=(mock_stage,))
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )

#         attack.mark_started()
#         attack.mark_active()
#         results = attack.run_stage_lifecycle(None, snapshot, ())  # no services

#         assert len(results) > 0
#         assert attack.status == RuntimeStatus.FAIL
#         mock_stage.execute.assert_not_called()

#     def test_a_atk_15_propagation_fail_from_stage(self):
#         """FAIL from a stage propagates to attack, remaining stages not launched."""
#         mock_stage_0 = MagicMock()
#         mock_stage_0.stage_name = "s0"
#         mock_stage_0.required_capabilities = ()
#         mock_stage_0.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.FAIL, reason="failed"
#         )

#         mock_stage_1 = MagicMock()
#         mock_stage_1.stage_name = "s1"
#         mock_stage_1.required_capabilities = ()

#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(StageSpec(id="s0", type="mock"), StageSpec(id="s1", type="mock")),
#         )
#         attack = Attack(spec=spec, stages=(mock_stage_0, mock_stage_1))
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )

#         dummy_service = _make_mock_service()
#         attack.mark_started()
#         attack.mark_active()
#         results = attack.run_stage_lifecycle(None, snapshot, (dummy_service,))

#         assert attack.status == RuntimeStatus.FAIL
#         mock_stage_1.execute.assert_not_called()

#     def test_a_atk_16_propagation_stop_from_stage(self):
#         """STOP from a stage propagates to attack as mark_stopped."""
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s0"
#         mock_stage.required_capabilities = ()
#         mock_stage.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.STOP, reason="stopped"
#         )

#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(StageSpec(id="s0", type="mock"),),
#         )
#         attack = Attack(spec=spec, stages=(mock_stage,))
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )

#         dummy_service = _make_mock_service()
#         attack.mark_started()
#         attack.mark_active()
#         attack.run_stage_lifecycle(None, snapshot, (dummy_service,))

#         assert attack.status == RuntimeStatus.STOPPED

#     def test_a_atk_17_all_stages_success_attack_succeeded(self):
#         """When all stages succeed, attack is marked succeeded."""
#         mock_s0 = MagicMock()
#         mock_s0.stage_name = "s0"
#         mock_s0.required_capabilities = ()
#         mock_s0.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.SUCCESS
#         )

#         mock_s1 = MagicMock()
#         mock_s1.stage_name = "s1"
#         mock_s1.required_capabilities = ()
#         mock_s1.execute.return_value = AttackStageResult(
#             stage_name="s1", status=Status.SUCCESS
#         )

#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(StageSpec(id="s0", type="mock"), StageSpec(id="s1", type="mock")),
#         )
#         attack = Attack(spec=spec, stages=(mock_s0, mock_s1))
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )

#         dummy_service = _make_mock_service()
#         attack.mark_started()
#         attack.mark_active()
#         attack.run_stage_lifecycle(None, snapshot, (dummy_service,))

#         assert attack.status == RuntimeStatus.SUCCESS

#     def test_a_atk_18_stop_trigger_deactivates_stage(self):
#         """Stage with stop_trigger: after trigger fires, stage becomes SUCCESS and deactivates."""
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s0"
#         mock_stage.required_capabilities = ()
#         mock_stage.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.SUCCESS
#         )

#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(
#                 StageSpec(
#                     id="s0",
#                     type="mock",
#                     stage_stop_trigger=_make_leaf_condition(verb="eq", value=1),
#                 ),
#             ),
#         )
#         attack = Attack(spec=spec, stages=(mock_stage,))

#         # First tick: start and run stage -> ACTIVE (has stop_trigger)
#         snapshot_start = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )
#         dummy_service = _make_mock_service()
#         attack.mark_started()
#         attack.mark_active()
#         attack.run_stage_lifecycle(None, snapshot_start, (dummy_service,))
#         assert attack.stage_runtimes[0].status == RuntimeStatus.ACTIVE

#         # Second tick: stop_trigger fires -> stage SUCCESS, deactivated
#         attack.run_stage_lifecycle(snapshot_start, snapshot_start, (dummy_service,))
#         assert attack.stage_runtimes[0].status == RuntimeStatus.SUCCESS
#         mock_stage.deactivate.assert_called()

#     def test_a_atk_19_sequential_stage_chain(self):
#         """stage_1 launches AFTER stage_0 succeeds."""
#         mock_s0 = MagicMock()
#         mock_s0.stage_name = "s0"
#         mock_s0.required_capabilities = ()
#         mock_s0.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.SUCCESS
#         )

#         mock_s1 = MagicMock()
#         mock_s1.stage_name = "s1"
#         mock_s1.required_capabilities = ()
#         mock_s1.execute.return_value = AttackStageResult(
#             stage_name="s1", status=Status.SUCCESS
#         )

#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(StageSpec(id="s0", type="mock"), StageSpec(id="s1", type="mock")),
#         )
#         attack = Attack(spec=spec, stages=(mock_s0, mock_s1))
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )

#         dummy_service = _make_mock_service()
#         attack.mark_started()
#         attack.mark_active()
#         results = attack.run_stage_lifecycle(None, snapshot, (dummy_service,))

#         # Both stages should have been executed (s0 succeeded, then s1 launched)
#         mock_s0.execute.assert_called_once()
#         mock_s1.execute.assert_called_once()
#         assert attack.stage_runtimes[0].status == RuntimeStatus.SUCCESS
#         assert attack.stage_runtimes[1].status == RuntimeStatus.SUCCESS

#     def test_a_atk_20_success_with_stop_trigger_stays_active(self):
#         """Stage with stop_trigger stays ACTIVE after first execute (checks stop_trigger on next tick)."""
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s0"
#         mock_stage.required_capabilities = ()
#         mock_stage.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.SUCCESS
#         )

#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(
#                 StageSpec(
#                     id="s0",
#                     type="mock",
#                     stage_stop_trigger=_make_leaf_condition(verb="eq", value=999),
#                 ),
#             ),
#         )
#         attack = Attack(spec=spec, stages=(mock_stage,))
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )

#         dummy_service = _make_mock_service()
#         attack.mark_started()
#         attack.mark_active()
#         attack.run_stage_lifecycle(None, snapshot, (dummy_service,))

#         # Stage should be ACTIVE (has stop_trigger, stop not yet triggered)
#         assert attack.stage_runtimes[0].status == RuntimeStatus.ACTIVE


# # ---------------------------------------------------------------------------
# # A-MGR: attack_manager.py  (NEEDS: carla/torch/traci)
# # ---------------------------------------------------------------------------


# class TestAttackManager:
#     """Tests for AttackManager."""

#     def _make_service_resolver(self, services: tuple[Any, ...]) -> Any:
#         """Create a service_resolver that returns services by id."""
#         service_map = {s.service_type: s for s in services}
#         return lambda node_id, service_type: service_map.get(service_type)

#     def test_a_mgr_01_full_lifecycle_success(self):
#         """Full cycle: inactive -> started -> active -> success."""
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s0"
#         mock_stage.required_capabilities = ()
#         mock_stage.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.SUCCESS
#         )

#         spec = AttackSpec(
#             name="atk",
#             targets=TargetSpec(
#                 kind="service_state_field",
#                 source=TriggerSourceSpec(
#                     kind="snapshot",
#                     node_type="vehicle",
#                     service_type="svc",
#                     field="target_ids",
#                 ),
#                 resolve_to_node_type="vehicle",
#                 resolve_to_service_name="svc",
#             ),
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(StageSpec(id="s0", type="mock"),),
#         )

#         mock_service = _make_mock_service("svc")
#         service_resolver = self._make_service_resolver((mock_service,))

#         attack = Attack(spec=spec, stages=(mock_stage,))
#         manager = AttackManager()

#         snapshot = _make_snapshot(
#             vehicle_nodes=(
#                 _make_node(
#                     node_id="v1",
#                     service_states={"svc": {"target_ids": "v1", "f": 1}},
#                 ),
#             ),
#         )

#         results = manager.evaluate((attack,), snapshot, service_resolver=service_resolver)
#         assert len(results) == 1
#         assert results[0].status == Status.SUCCESS

#     def test_a_mgr_02_attack_stopped_by_stop_trigger(self):
#         """Attack stops when stop_trigger fires."""
#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s0"
#         mock_stage.required_capabilities = ()
#         mock_stage.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.SUCCESS
#         )

#         spec = AttackSpec(
#             name="atk",
#             targets=TargetSpec(
#                 kind="service_state_field",
#                 source=TriggerSourceSpec(
#                     kind="snapshot",
#                     node_type="vehicle",
#                     service_type="svc",
#                     field="target_ids",
#                 ),
#                 resolve_to_node_type="vehicle",
#                 resolve_to_service_name="svc",
#             ),
#             start_trigger=_make_leaf_condition(verb="gt", value=0),
#             stop_trigger=_make_leaf_condition(verb="eq", value=0),
#             stages=(
#                 StageSpec(
#                     id="s0",
#                     type="mock",
#                     stage_stop_trigger=_make_leaf_condition(verb="eq", value=99999),
#                 ),
#             ),
#         )

#         mock_service = _make_mock_service("svc")
#         service_resolver = self._make_service_resolver((mock_service,))
#         attack = Attack(spec=spec, stages=(mock_stage,))
#         manager = AttackManager()

#         # Tick 1: start
#         snap_start = _make_snapshot(
#             vehicle_nodes=(
#                 _make_node(
#                     node_id="v1",
#                     service_states={"svc": {"target_ids": "v1", "f": 10}},
#                 ),
#             ),
#         )
#         manager.evaluate((attack,), snap_start, service_resolver=service_resolver)
#         assert attack.status == RuntimeStatus.ACTIVE

#         # Tick 2: stop trigger fires (f=0)
#         snap_stop = _make_snapshot(
#             vehicle_nodes=(
#                 _make_node(
#                     node_id="v1",
#                     service_states={"svc": {"target_ids": "v1", "f": 0}},
#                 ),
#             ),
#         )
#         results = manager.evaluate((attack,), snap_stop, service_resolver=service_resolver)
#         assert len(results) == 1
#         assert results[0].status == Status.STOP

#     def test_a_mgr_03_attack_not_started_trigger_unmet(self):
#         """Attack is ignored when start trigger is not met."""
#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="eq", value=999),
#             stages=(StageSpec(id="s0", type="mock"),),
#         )
#         mock_stage = MagicMock()
#         attack = Attack(spec=spec, stages=(mock_stage,))
#         manager = AttackManager()

#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )
#         results = manager.evaluate((attack,), snapshot, service_resolver=lambda n, t: None)
#         assert results == ()

#     def test_a_mgr_04_no_targets_fail(self):
#         """No resolved targets -> FAIL."""
#         spec = AttackSpec(
#             name="atk",
#             targets=TargetSpec(
#                 kind="service_state_field",
#                 source=TriggerSourceSpec(
#                     kind="snapshot",
#                     node_type="vehicle",
#                     service_type="svc",
#                     field="target_ids",
#                 ),
#                 resolve_to_node_type="vehicle",
#                 resolve_to_service_name="svc",
#             ),
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(StageSpec(id="s0", type="mock"),),
#         )
#         mock_stage = MagicMock()
#         attack = Attack(spec=spec, stages=(mock_stage,))
#         manager = AttackManager()

#         # No target_ids in snapshot -> no targets resolved
#         snapshot = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )
#         results = manager.evaluate((attack,), snapshot, service_resolver=lambda n, t: None)
#         assert len(results) == 1
#         assert results[0].status == Status.FAIL

#     def test_a_mgr_05_no_restart_after_stop(self):
#         """Attack is not restarted after being stopped."""
#         spec = AttackSpec(
#             name="atk",
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stop_trigger=_make_leaf_condition(verb="eq", value=0),
#             stages=(StageSpec(id="s0", type="mock"),),
#         )
#         mock_stage = MagicMock()
#         mock_stage.required_capabilities = ()
#         mock_stage.execute.return_value = AttackStageResult(
#             stage_name="s0", status=Status.SUCCESS
#         )
#         attack = Attack(spec=spec, stages=(mock_stage,))
#         manager = AttackManager()

#         # No targets -> trigger fires but resolve_targets returns empty -> FAIL result
#         # (attack status stays INACTIVE; manager does not call mark_failed)
#         snap1 = _make_snapshot(
#             vehicle_nodes=(_make_node(service_states={"svc": {"f": 1}}),)
#         )
#         results = manager.evaluate((attack,), snap1, service_resolver=lambda n, t: None)
#         assert attack.status == RuntimeStatus.INACTIVE
#         assert len(results) == 1
#         assert results[0].status == Status.FAIL

#         # On next tick, the attack should not start (still no targets)
#         prev_status = attack.status
#         manager.evaluate((attack,), snap1, service_resolver=lambda n, t: None)
#         assert attack.status == prev_status

#     def test_a_mgr_06_active_attack_run_stage_lifecycle_called(self):
#         """Active attack with unmet stop_trigger continues executing stages."""
#         call_count = 0

#         mock_stage = MagicMock()
#         mock_stage.stage_name = "s0"
#         mock_stage.required_capabilities = ()

#         def fake_execute(services):
#             nonlocal call_count
#             call_count += 1
#             return AttackStageResult(stage_name="s0", status=Status.SUCCESS)

#         mock_stage.execute = fake_execute

#         # Stage with stop_trigger that never fires
#         spec = AttackSpec(
#             name="atk",
#             targets=TargetSpec(
#                 kind="service_state_field",
#                 source=TriggerSourceSpec(
#                     kind="snapshot",
#                     node_type="vehicle",
#                     service_type="svc",
#                     field="target_ids",
#                 ),
#                 resolve_to_node_type="vehicle",
#                 resolve_to_service_name="svc",
#             ),
#             start_trigger=_make_leaf_condition(verb="exists"),
#             stages=(
#                 StageSpec(
#                     id="s0",
#                     type="mock",
#                     stage_stop_trigger=_make_leaf_condition(verb="eq", value=99999),
#                 ),
#             ),
#         )
#         attack = Attack(spec=spec, stages=(mock_stage,))
#         manager = AttackManager()

#         mock_service = _make_mock_service("svc")
#         service_resolver = lambda node_id, svc_type: mock_service

#         snap = _make_snapshot(
#             vehicle_nodes=(_make_node(
#                 node_id="v1",
#                 service_states={"svc": {"target_ids": "v1", "f": 1}},
#             ),)
#         )
#         manager.evaluate((attack,), snap, service_resolver=service_resolver)
#         assert call_count >= 1


# # ---------------------------------------------------------------------------
# # A-UTIL: utils.py  (NEEDS: carla/torch/traci)
# # ---------------------------------------------------------------------------


# class TestUtils:
#     """Tests for utility functions."""

#     def test_a_util_01_install_output_interceptor_patches(self):
#         """install_output_interceptor patches the service method."""
#         service = _make_mock_service(
#             capability_bindings={Capability.RESPONSE_OBSERVE: lambda: "original"}
#         )
#         # Manually set the method on the service instance
#         original_method = service.capability_bindings[Capability.RESPONSE_OBSERVE]
#         original_method.__name__ = "response_observe"
#         service.response_observe = original_method

#         rewritten = []

#         def rewrite(result):
#             rewritten.append(result)
#             return "rewritten"

#         restore = install_output_interceptor(
#             service, Capability.RESPONSE_OBSERVE, rewrite_result=rewrite
#         )

#         # The patched method should return rewritten output
#         # But since we're using MagicMock, we need to test through the actual mechanism
#         assert callable(restore)
#         restore()  # Clean up

#     def test_a_util_02_restore_returns_original(self):
#         """Restore callback returns the method to its original state."""
#         def original_method():
#             return "original"

#         original_method.__name__ = "test_method"

#         service = MagicMock()
#         service.test_method = original_method
#         service.capability_bindings = {Capability.RESPONSE_OBSERVE: original_method}

#         rewritten_called = []

#         def rewrite(result):
#             rewritten_called.append(result)
#             return "rewritten"

#         restore = install_output_interceptor(
#             service, Capability.RESPONSE_OBSERVE, rewrite_result=rewrite
#         )

#         # Restore and verify method is back
#         restore()
#         assert service.test_method is original_method

#     def test_a_util_03_idempotent_restore(self):
#         """Calling restore twice does not raise an error."""
#         def original_method():
#             return "original"

#         original_method.__name__ = "test_method2"

#         service = MagicMock()
#         service.test_method2 = original_method
#         service.capability_bindings = {Capability.RESPONSE_OBSERVE: original_method}

#         restore = install_output_interceptor(
#             service, Capability.RESPONSE_OBSERVE, rewrite_result=lambda r: r
#         )
#         restore()
#         restore()  # Should not raise

#     def test_a_util_04_wrap_method_output(self):
#         """wrap_method_output passes args/kwargs and rewrites result."""
#         def original_fn(a, b, c=None):
#             return (a, b, c)

#         wrapped = wrap_method_output(original_fn, rewrite_result=lambda r: f"wrapped: {r}")
#         result = wrapped(1, 2, c=3)
#         assert result == "wrapped: (1, 2, 3)"

#     def test_a_util_05_get_capability_binding_key_error(self):
#         """get_capability_binding raises RuntimeError when capability is missing."""
#         service = _make_mock_service(capability_bindings={})
#         with pytest.raises(RuntimeError, match="does not expose capability"):
#             from opencda.core.attack.adversary_framework.utils import get_capability_binding
#             get_capability_binding(service, Capability.RESPONSE_OBSERVE)

#     def test_a_util_06_match_services(self):
#         """match_services returns only services with all required capabilities."""
#         svc_full = _make_mock_service(
#             capability_bindings={
#                 Capability.RESPONSE_OBSERVE: lambda: None,
#                 Capability.RESPONSE_SUBMIT: lambda: None,
#             }
#         )
#         svc_partial = _make_mock_service(
#             capability_bindings={Capability.RESPONSE_OBSERVE: lambda: None}
#         )
#         svc_empty = _make_mock_service(capability_bindings={})

#         required = (Capability.RESPONSE_OBSERVE, Capability.RESPONSE_SUBMIT)
#         matched = match_services([svc_full, svc_partial, svc_empty], required)
#         assert len(matched) == 1
#         assert matched[0] is svc_full

#     def test_a_util_07_resolve_targets_unsupported_kind(self):
#         """resolve_targets raises ValueError for unsupported kind."""
#         target_spec = TargetSpec(
#             kind="unsupported_kind",
#             source=TriggerSourceSpec(kind="snapshot"),
#             resolve_to_node_type="v",
#             resolve_to_service_name="s",
#         )
#         with pytest.raises(ValueError, match="Unsupported target resolution kind"):
#             resolve_targets(target_spec, _make_snapshot(), lambda n, t: None)

#     def test_a_util_08_resolve_targets_correct(self):
#         """resolve_targets resolves through service_resolver."""
#         target_spec = TargetSpec(
#             kind="service_state_field",
#             source=TriggerSourceSpec(
#                 kind="snapshot",
#                 node_type="vehicle",
#                 service_type="svc",
#                 field="target_ids",
#             ),
#             resolve_to_node_type="vehicle",
#             resolve_to_service_name="aim_client",
#         )
#         mock_service = _make_mock_service("aim_client")
#         service_resolver = lambda node_id, svc_type: mock_service

#         snapshot = _make_snapshot(
#             vehicle_nodes=(
#                 _make_node(
#                     node_id="v1",
#                     service_states={"svc": {"target_ids": "v1"}},
#                 ),
#             ),
#         )
#         result = resolve_targets(target_spec, snapshot, service_resolver)
#         assert len(result) == 1
#         assert result[0] is mock_service


# # ---------------------------------------------------------------------------
# # A-SNIFF: stages/sniffer.py  (NEEDS: carla/torch/traci)
# # ---------------------------------------------------------------------------


# class TestSnifferStage:
#     """Tests for SnifferStage."""

#     def _make_sniffer_service(self) -> Any:
#         """Create a mock service with RESPONSE_OBSERVE capability."""
#         def observe_method():
#             return {"data": "original"}

#         observe_method.__name__ = "response_observe"

#         service = MagicMock()
#         service.service_type = "aim_client"
#         service.capability_bindings = {Capability.RESPONSE_OBSERVE: observe_method}
#         service.response_observe = observe_method
#         return service

#     def test_a_sniff_01_execute_deactivates_then_installs(self):
#         """execute calls deactivate first, then installs interceptors."""
#         stage = SnifferStage()
#         service = self._make_sniffer_service()

#         result = stage.execute((service,))

#         assert result.status == Status.SUCCESS
#         assert len(stage._restore_callbacks) > 0

#         # Calling the patched method should record observations
#         output = service.response_observe()
#         assert len(stage.observed_outputs) == 1
#         assert stage.observed_outputs[0].output == {"data": "original"}

#         # Cleanup
#         stage.deactivate()

#     def test_a_sniff_02_no_services_fail(self):
#         """SnifferStage returns FAIL when no services are provided."""
#         stage = SnifferStage()
#         result = stage.execute(())
#         assert result.status == Status.FAIL

#     def test_a_sniff_03_observe_output_stores_deepcopy(self):
#         """_observe_output stores a deepcopy but returns the original."""
#         stage = SnifferStage()
#         service = self._make_sniffer_service()
#         stage.execute((service,))

#         original_data = {"data": [1, 2, 3]}
#         output = stage._observe_output(service, original_data)

#         # Output should be unchanged
#         assert output is original_data

#         # Observed output should be a deepcopy
#         assert len(stage.observed_outputs) == 1
#         assert stage.observed_outputs[0].output == original_data
#         assert stage.observed_outputs[0].output is not original_data

#         stage.deactivate()

#     def test_a_sniff_04_deactivate_preserves_observed_outputs(self):
#         """deactivate removes interceptors but keeps observed_outputs."""
#         stage = SnifferStage()
#         service = self._make_sniffer_service()
#         stage.execute((service,))

#         service.response_observe()
#         assert len(stage.observed_outputs) == 1

#         stage.deactivate()
#         assert len(stage.observed_outputs) == 1  # Not cleared
#         assert len(stage._restore_callbacks) == 0  # Interceptors removed

#     def test_a_sniff_05_deactivate_called_before_new_interceptors(self):
#         """execute calls deactivate before installing new interceptors."""
#         stage = SnifferStage()
#         service = self._make_sniffer_service()

#         # First execute
#         stage.execute((service,))
#         service.response_observe()
#         assert len(stage.observed_outputs) == 1

#         # Second execute: deactivate should clear interceptors first
#         stage.execute((service,))
#         # The old callbacks are cleared, new ones installed
#         assert len(stage._restore_callbacks) == 1

#         stage.deactivate()


# # ---------------------------------------------------------------------------
# # A-RPL: stages/response_replayer.py  (NEEDS: carla/torch/traci)
# # ---------------------------------------------------------------------------


# class TestResponseReplayerStage:
#     """Tests for ResponseReplayerStage."""

#     def _make_replayer_service(self) -> Any:
#         """Create a mock service with RESPONSE_SUBMIT capability."""
#         def submit_method():
#             return {"response": "original"}

#         submit_method.__name__ = "response_submit"

#         service = MagicMock()
#         service.service_type = "aim_server"
#         service.capability_bindings = {Capability.RESPONSE_SUBMIT: submit_method}
#         service.response_submit = submit_method
#         return service

#     def test_a_rpl_01_execute_deactivates_then_installs(self):
#         """execute calls deactivate first, then installs interceptors."""
#         stage = ResponseReplayerStage()
#         service = self._make_replayer_service()

#         result = stage.execute((service,))

#         assert result.status == Status.SUCCESS
#         assert len(stage._restore_callbacks) > 0

#         stage.deactivate()

#     def test_a_rpl_02_no_services_fail(self):
#         """ResponseReplayerStage returns FAIL when no services are provided."""
#         stage = ResponseReplayerStage()
#         result = stage.execute(())
#         assert result.status == Status.FAIL

#     def test_a_rpl_03_first_call_passthrough(self):
#         """First call returns the original output (no previous to replay)."""
#         stage = ResponseReplayerStage()
#         service = self._make_replayer_service()
#         stage.execute((service,))

#         output = service.response_submit()
#         assert output == {"response": "original"}

#         stage.deactivate()

#     def test_a_rpl_04_second_call_replays_previous(self):
#         """Second call returns a deepcopy of the first output."""
#         stage = ResponseReplayerStage()
#         service = self._make_replayer_service()
#         stage.execute((service,))

#         first = service.response_submit()
#         second = service.response_submit()

#         assert second == first
#         assert second is not first  # Should be a deepcopy

#         stage.deactivate()

#     def test_a_rpl_05_third_call_replays_second(self):
#         """Third call returns a deepcopy of the second call's captured output."""
#         stage = ResponseReplayerStage()
#         service = self._make_replayer_service()
#         stage.execute((service,))

#         first = service.response_submit()  # pass-through (no previous)
#         second = service.response_submit()  # replays first
#         third = service.response_submit()   # replays second

#         # All calls return the same underlying output since the method always
#         # returns {"response": "original"}. Third replays the second capture.
#         assert third == {"response": "original"}
#         assert third == second
#         assert third is not second  # deepcopy

#         stage.deactivate()

#     def test_a_rpl_06_deactivate_clears_history(self):
#         """deactivate clears replay history."""
#         stage = ResponseReplayerStage()
#         service = self._make_replayer_service()
#         stage.execute((service,))

#         service.response_submit()
#         assert len(stage._previous_outputs_by_service) > 0

#         stage.deactivate()
#         assert len(stage._previous_outputs_by_service) == 0

#     def test_a_rpl_07_separate_history_per_service(self):
#         """Each service has its own replay buffer (keyed by id)."""
#         call_count_a = 0
#         call_count_b = 0

#         def observe_a():
#             nonlocal call_count_a
#             call_count_a += 1
#             return {"response": f"a_{call_count_a}"}

#         observe_a.__name__ = "response_submit"

#         def observe_b():
#             nonlocal call_count_b
#             call_count_b += 1
#             return {"response": f"b_{call_count_b}"}

#         observe_b.__name__ = "response_submit"

#         service_a = MagicMock()
#         service_a.service_type = "type_a"
#         service_a.capability_bindings = {Capability.RESPONSE_SUBMIT: observe_a}
#         service_a.response_submit = observe_a

#         service_b = MagicMock()
#         service_b.service_type = "type_b"
#         service_b.capability_bindings = {Capability.RESPONSE_SUBMIT: observe_b}
#         service_b.response_submit = observe_b

#         stage = ResponseReplayerStage()
#         stage.execute((service_a, service_b))

#         out_a1 = service_a.response_submit()  # a_1 (pass-through)
#         out_b1 = service_b.response_submit()  # b_1 (pass-through)

#         out_a2 = service_a.response_submit()  # replays a_1
#         out_b2 = service_b.response_submit()  # replays b_1

#         assert out_a2 == {"response": "a_1"}
#         assert out_b2 == {"response": "b_1"}

#         stage.deactivate()
