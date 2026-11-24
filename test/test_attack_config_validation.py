"""Configuration checks that run without a simulation or live services."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Mapping, NamedTuple
from unittest.mock import Mock, patch

import pytest

from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from opencda.core.attack.adversary_framework import AttackManager
from opencda.core.attack.adversary_framework.condition_evaluator import evaluate_condition
from opencda.scenario_testing.types import NodeSnapshot, SimulationSnapshot
from opencda.scenario_testing.utils.yaml_utils import read_config_mapping


def _prepare(stages=(), *, scenario_config=None, **attack):
    config = {"attack": {"name": "example", "stages": list(stages), **attack}}
    return AttackManager().prepare_attacks([(Path("example.yaml"), config)], scenario_config or {})[0]


def _rewrite(path: str) -> dict:
    return {"id": "example_stage", "type": "spoofer", "params": {"rewrites": [{"path": path, "value": 0}]}}


def _reference(stage_id: str) -> dict:
    return {"source": {"kind": "stage", "stage_id": stage_id}, "verb": "exists"}


@pytest.mark.parametrize(
    ("stages", "path"),
    [
        ([{"id": "x", "type": "absent"}], "stages[0].type"),
        ([{"id": "x", "type": "sniffer"}] * 2, "stages[1].id"),
        ([{"id": "x", "type": "sniffer", "capabilities": ["invalid"]}], "stages[0].capabilities[0]"),
        ([{"id": "x", "type": "dropper", "params": {"drop_rate": 2.0}}], "stages[0].params.drop_rate"),
        ([{"id": "x", "type": "dropper", "params": {"drop_reate": 0.3}}], "stages[0].params.drop_reate"),
        ([{"id": "x", "type": "dropper", "params": {"drop_rate": 0.3, "sed": 7}}], "stages[0].params.sed"),
        (
            [{"id": "x", "type": "spoofer", "params": {"rewrites": [{"path": "payload", "value": 2}], "operation": "multiply"}}],
            "stages[0].params.operation",
        ),
        (
            [{"id": "x", "type": "spoofer", "params": {"rewrites": [{"path": "payload", "value": 2, "operaton": "multiply"}]}}],
            "stages[0].params.rewrites[0].operaton",
        ),
        ([_rewrite("no_such_field")], "stages[0].params.rewrites[0].path"),
        ([_rewrite("dst_owner_id.child")], "stages[0].params.rewrites[0].path"),
        ([_rewrite("payload[bad]")], "stages[0].params.rewrites[0].path"),
    ],
)
def test_config_error_identifies_the_original_key(stages, path):
    with pytest.raises(ValueError) as exc:
        _prepare(stages)
    assert str(exc.value).startswith(f"example.yaml: attack.{path}:")


def test_config_error_identifies_a_key_outside_attack(tmp_path):
    config = tmp_path / "example.yaml"
    config.write_text("attack:\n  name: example\nstop_trigger:\n  source: {kind: snapshot, field: node_id}\n  verb: exists\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        AttackManager().prepare_attacks([(config, read_config_mapping(config))], {})
    assert str(exc.value).startswith(f"{config}: stop_trigger:")


@pytest.mark.parametrize("params", [None, {}, {"drop_rate": 0.3}, {"seed": 7}])
def test_dropper_parameters_remain_optional(params):
    assert _prepare([{"id": "example", "type": "dropper", "params": params}]).stages


@pytest.mark.parametrize("key", ["requirements", "stage_start_trigger", "stage_stop_trigger"])
def test_reference_inside_stage_condition(key):
    stages = [{"id": "example", "type": "sniffer", key: {"any": [{"all": [_reference("missing")]}]}}]
    with pytest.raises(ValueError) as exc:
        _prepare(stages)
    assert f"stages[0].{key}.any[0].all[0].source.stage_id" in str(exc.value)


@pytest.mark.parametrize("key", ["requirements", "start_trigger", "stop_trigger"])
def test_attack_condition_can_reference_any_declared_stage(key):
    with pytest.raises(ValueError) as exc:
        _prepare(**{key: _reference("missing")})
    assert f"attack.{key}.source.stage_id" in str(exc.value)
    stage = {"id": "later", "type": "sniffer", "capabilities": "command.observe"}
    targets = {"kind": "service_state_field", "source": {"kind": "snapshot", "field": "node_id"}, "resolve_to": {"node_type": "vehicle"}}
    scenario = {"scenario": {"single_cav_list": [{"behavior_services": {"movement_controller": {}}}]}}
    attack = _prepare([stage], scenario_config=scenario, targets=targets, **{key: _reference("later")})
    assert attack.attack_name == "example"


def test_mixed_condition_is_rejected_before_parser_discards_a_key():
    with pytest.raises(ValueError) as exc:
        _prepare(start_trigger={"all": [_reference("x")], "any": [_reference("y")]})
    assert "example.yaml: attack.start_trigger.any:" in str(exc.value)


def test_yaml_reader_preserves_numeric_literals_and_identifies_invalid_root(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("threshold: 1e-3\n", encoding="utf-8")
    assert read_config_mapping(config) == {"threshold": 0.001}
    config.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        read_config_mapping(config)
    assert str(config) in str(exc.value)


def test_yaml_rejects_duplicate_keys_but_allows_merge_overrides(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("base: &base {value: 1}\ncondition: {<<: *base, value: 2}\n", encoding="utf-8")
    assert read_config_mapping(config)["condition"] == {"value": 2}
    config.write_text("condition:\n  value: 1\n  value: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate key 'value'") as exc:
        read_config_mapping(config)
    assert f'{config}", line 3' in str(exc.value)


@pytest.mark.parametrize(("key", "value"), [("field", ["node_id"]), ("node_id", 123), ("feild", "node_id"), ("field", "nodeid")])
def test_source_fields_fail_at_the_original_location(key, value):
    condition = {"source": {"kind": "snapshot", key: value}, "verb": "exists"}
    with pytest.raises(ValueError, match=rf"attack\.requirements\.source\.{key}:"):
        _prepare(requirements=condition)


def test_unknown_key_and_missing_targets_are_detected():
    trigger = {"source": {"kind": "snapshot", "field": "node_id"}, "verb": "exists"}
    stage = {"id": "example", "type": "sniffer"}
    with pytest.raises(ValueError, match=r"attack\.start_triger:"):
        _prepare([stage], start_triger=trigger)
    with pytest.raises(ValueError, match=r"attack\.targets:"):
        _prepare([stage], start_trigger=trigger)


def test_missing_nested_required_key():
    with pytest.raises(ValueError) as exc:
        _prepare(stop_trigger={"any": [{"source": {}, "verb": "exists"}]})
    assert "example.yaml: attack.stop_trigger.any[0].source.kind:" in str(exc.value)


def test_unreachable_trigger_warns_and_allows_preparation(caplog):
    source = {"kind": "snapshot", "service_type": "aim_server", "field": "tracked_vehicle_count"}
    scenario = {"scenario": {"rsu_list": [{"behavior_services": {"aim_server": {}}}]}}
    attack = _prepare(scenario_config=scenario, stop_trigger={"source": source, "verb": "lt", "value": 0})
    assert attack.attack_name == "example"
    assert "example.yaml: attack.stop_trigger.value" in caplog.text
    caplog.clear()
    _prepare(scenario_config=scenario, stop_trigger={"source": source, "verb": "lte", "value": 0})
    assert not caplog.records


def test_conflicting_equalities_warn_only_for_a_single_value(caplog):
    source = {"kind": "snapshot", "field": "node_type"}
    conditions = [{"source": source, "verb": "eq", "value": value} for value in ("vehicle", "rsu")]
    _prepare(start_trigger={"all": conditions})
    assert not caplog.records  # Different participants can satisfy the two predicates.
    source["node_id"] = "one_node"
    _prepare(start_trigger={"any": conditions})
    assert not caplog.records
    _prepare(start_trigger={"all": conditions})
    assert "example.yaml: attack.start_trigger.all[1].value" in caplog.text


def test_only_existing_participants_contribute_service_declarations():
    targets = {
        "kind": "service_state_field",
        "source": {"kind": "snapshot", "field": "node_id"},
        "resolve_to": {"node_type": "vehicle", "service_type": "example"},
    }
    definition = [(Path("example.yaml"), {"attack": {"name": "example", "targets": targets}})]
    config = {"vehicle_base": {"behavior_services": {"example": {}}}, "scenario": {"rsu_list": [{}]}}
    with pytest.raises(ValueError) as exc:
        AttackManager().prepare_attacks(definition, config)
    assert "example.yaml: attack.targets.resolve_to.service_type" in str(exc.value)
    config["scenario"]["platoon_list"] = [{"members": [{}]}]
    assert len(AttackManager().prepare_attacks(definition, config)) == 1


@pytest.mark.parametrize(
    ("kind", "verb", "key"),
    [("snapshott", "exists", "source.kind"), ("snapshot", "greater_than", "verb"), ("stage", "added", "verb"), ("attack", "removed", "verb")],
)
def test_condition_vocabulary_reports_nested_key(kind, verb, key):
    condition = {"source": {"kind": kind}, "verb": verb}
    with pytest.raises(ValueError) as exc:
        _prepare([{"id": "example", "type": "sniffer", "stage_stop_trigger": {"any": [condition]}}])
    assert str(exc.value).startswith(f"example.yaml: attack.stages[0].stage_stop_trigger.any[0].{key}:")
    if verb in {"added", "removed"}:
        assert _prepare(start_trigger={"source": {"kind": "snapshot", "field": "service_states"}, "verb": verb})


def test_snapshot_condition_service_is_scoped_to_its_node_type():
    scenario = {"scenario": {"single_cav_list": [{}], "rsu_list": [{}]}, "rsu_base": {"behavior_services": {"example": {}}}}
    source = {"kind": "snapshot", "service_type": "exmaple"}
    condition = {"all": [{"source": source, "verb": "exists"}]}
    with pytest.raises(ValueError) as exc:
        _prepare(scenario_config=scenario, requirements=condition)
    assert str(exc.value).startswith("example.yaml: attack.requirements.all[0].source.service_type:")
    source["service_type"] = "example"
    assert _prepare(scenario_config=scenario, requirements=condition)
    source["node_type"] = "vehicle"
    with pytest.raises(ValueError, match=r"requirements\.all\[0\]\.source\.service_type:"):
        _prepare(scenario_config=scenario, requirements=condition)
    source["node_type"] = "rsu"
    assert _prepare(scenario_config=scenario, requirements=condition)


@pytest.mark.parametrize("service", [{}, {"service_type": "example"}])
def test_unknown_snapshot_node_type_is_rejected_even_when_a_service_matches(service):
    scenario = {"scenario": {"rsu_list": [{"behavior_services": {"example": {}}}]}}
    source = {"kind": "snapshot", "node_type": "vehcile", "field": "node_id", **service}
    with pytest.raises(ValueError, match=r"attack\.requirements\.all\[0\]\.source\.node_type:"):
        _prepare(scenario_config=scenario, requirements={"all": [{"source": source, "verb": "exists"}]})
    target = {"kind": "service_state_field", "source": source, "resolve_to": {"node_type": "vehicle"}}
    with pytest.raises(ValueError, match=r"attack\.targets\.source\.node_type:"):
        _prepare(scenario_config=scenario, targets=target)
    source["node_type"] = None
    assert _prepare(scenario_config=scenario, targets=target)


@pytest.mark.parametrize("verb", ["gt", "gte", "lt", "lte"])
@pytest.mark.parametrize("operand", [{}, {"vaule": 2}, {"value": None}])
def test_ordered_comparison_requires_a_value(verb, operand):
    condition = {"source": {"kind": "snapshot", "field": "node_id"}, "verb": verb, **operand}
    key = "vaule" if "vaule" in operand else "value"
    with pytest.raises(ValueError, match=rf"attack\.start_trigger\.any\[0\]\.{key}:"):
        _prepare(start_trigger={"any": [condition]})
    assert _prepare(start_trigger={"source": condition["source"], "verb": verb, "value": "vehicle_0"})


@pytest.mark.parametrize("verb", ["gt", "gte", "lt", "lte"])
def test_ordered_comparison_checks_the_declared_service_field_type(aim_service_types, verb):
    scenario = {"scenario": {"rsu_list": [{"behavior_services": {"aim_server": {}}}]}}
    source = {"kind": "snapshot", "service_type": "aim_server", "field": "tracked_vehicle_count"}
    condition = {"source": source, "verb": verb, "value": "2"}
    with pytest.raises(ValueError, match=r"example\.yaml: attack\.start_trigger\.any\[0\]\.value:.*tracked_vehicle_count.*int.*str"):
        _prepare(scenario_config=scenario, start_trigger={"any": [condition]})
    for threshold in (0, 2.5):
        assert _prepare(scenario_config=scenario, start_trigger={**condition, "value": threshold})


@pytest.mark.parametrize(
    "source", [{"kind": "snapshot", "field": "node_id"}, {"kind": "stage", "field": "status"}, {"kind": "attack", "field": "attack_name"}]
)
def test_ordered_comparison_preserves_string_ordering(source):
    with pytest.raises(ValueError, match=r"attack\.requirements\.value:"):
        _prepare(requirements={"source": source, "verb": "lt", "value": 1})
    assert _prepare(requirements={"source": source, "verb": "lt", "value": "z"})


def test_ordered_comparison_with_dynamic_service_state():
    scenario = {"scenario": {"single_cav_list": [{"behavior_services": {"extension": {}}}]}}
    condition = {"source": {"kind": "snapshot", "service_type": "extension", "field": "reading"}, "verb": "gt", "value": 0}
    assert _prepare(scenario_config=scenario, requirements=condition)


def test_ordered_comparison_rejects_a_nullable_operand():
    scenario = {"scenario": {"single_cav_list": [{"behavior_services": {"movement_controller": {}}}]}}
    source = {"kind": "snapshot", "service_type": "movement_controller", "field": "owner_id"}
    with pytest.raises(ValueError, match=r"attack\.requirements\.value:.*NoneType"):
        _prepare(scenario_config=scenario, requirements={"source": source, "verb": "gt", "value": "vehicle"})


@pytest.mark.parametrize("named_tuple", [True, False])
def test_snapshot_supported_field_is_validated_and_read(monkeypatch, named_tuple):
    class Reading(NamedTuple):
        value: int

    class TupleService:
        def get_state(self) -> Reading:
            return Reading(3)

    class MappingService:
        def get_state(self) -> Mapping[str, int]:
            return MappingProxyType({"value": 3})

    service = TupleService if named_tuple else MappingService
    monkeypatch.setitem(BehaviorServiceRegistry._registry, "example", service)
    scenario = {"scenario": {"single_cav_list": [{"behavior_services": {"example": {}}}]}}
    condition = {"source": {"kind": "snapshot", "service_type": "example", "field": "value"}, "verb": "gt", "value": 2}
    attack = _prepare(scenario_config=scenario, requirements=condition)
    snapshot = SimulationSnapshot(1, vehicle_nodes=(NodeSnapshot("vehicle", "vehicle", {"example": service().get_state()}),))
    assert evaluate_condition(attack, attack.spec.requirements, None, snapshot)
    with pytest.raises(ValueError, match=r"attack\.requirements\.value:"):
        _prepare(scenario_config=scenario, requirements={**condition, "value": "2"})


def test_condition_types_include_collection_elements_and_custom_ordering(monkeypatch):
    from opencda.core.application.behavior.types import Location

    @dataclass
    class State:
        readings: list[int]
        position: Location

    class Service:
        def get_state(self) -> State:
            return State([1], Location())

    monkeypatch.setitem(BehaviorServiceRegistry._registry, "example", Service)
    scenario = {"scenario": {"single_cav_list": [{"behavior_services": {"example": {}}}]}}
    source = {"kind": "snapshot", "service_type": "example", "field": "readings"}
    with pytest.raises(ValueError, match=r"attack\.requirements\.value:"):
        _prepare(scenario_config=scenario, requirements={"source": source, "verb": "gt", "value": ["1"]})
    assert _prepare(scenario_config=scenario, requirements={"source": source, "verb": "gt", "value": [1]})
    source["field"] = "position"
    with pytest.raises(ValueError, match=r"attack\.requirements\.source\.field:"):
        _prepare(scenario_config=scenario, requirements={"source": source, "verb": "increased"})


@pytest.mark.parametrize("verb", ["added", "removed"])
def test_collection_transitions_validate_the_field_and_element(aim_service_types, verb):
    scenario = {"scenario": {"rsu_list": [{"behavior_services": {"aim_server": {}}}]}}
    source = {"kind": "snapshot", "service_type": "aim_server", "field": "tracked_vehicle_count"}
    condition = {"source": source, "verb": verb}
    with pytest.raises(ValueError, match=r"attack\.start_trigger\.source\.field:"):
        _prepare(scenario_config=scenario, start_trigger=condition)
    source["field"] = "tracked_vehicle_ids"
    assert _prepare(scenario_config=scenario, start_trigger=condition)
    assert _prepare(scenario_config=scenario, start_trigger={**condition, "value": "vehicle_1"})
    with pytest.raises(ValueError, match=r"attack\.start_trigger\.value:"):
        _prepare(scenario_config=scenario, start_trigger={**condition, "value": ["vehicle_1"]})


@pytest.mark.parametrize("verb", ["increased", "decreased"])
def test_delta_transitions_validate_the_field_and_threshold(aim_service_types, verb):
    scenario = {"scenario": {"rsu_list": [{"behavior_services": {"aim_server": {}}}]}}
    source = {"kind": "snapshot", "service_type": "aim_server", "field": "tracked_vehicle_count"}
    condition = {"source": source, "verb": verb}
    assert _prepare(scenario_config=scenario, start_trigger=condition)
    assert _prepare(scenario_config=scenario, start_trigger={**condition, "value": 0.5})
    with pytest.raises(ValueError, match=r"attack\.start_trigger\.value:"):
        _prepare(scenario_config=scenario, start_trigger={**condition, "value": "1"})
    source["field"] = "tracked_vehicle_ids"
    with pytest.raises(ValueError, match=r"attack\.start_trigger\.source\.field:"):
        _prepare(scenario_config=scenario, start_trigger=condition)


def test_known_snapshot_service_fields_checked_in_conditions_and_targets(aim_service_types):
    scenario = {"scenario": {"rsu_list": [{"behavior_services": {"aim_server": {}}}]}}
    source = {"kind": "snapshot", "service_type": "aim_server", "field": "tracked_vehcle_count"}
    with pytest.raises(ValueError, match=r"example\.yaml: attack\.requirements\.source\.field:"):
        _prepare(scenario_config=scenario, requirements={"source": source, "verb": "exists"})
    targets = {"kind": "service_state_field", "source": source, "resolve_to": {"node_type": "vehicle"}}
    with pytest.raises(ValueError, match=r"example\.yaml: attack\.targets\.source\.field:"):
        _prepare(scenario_config=scenario, targets=targets)
    source["field"] = "tracked_vehicle_count"
    with pytest.raises(ValueError, match=r"attack\.targets\.source\.field:.*node ID"):
        _prepare(scenario_config=scenario, targets=targets)
    source["field"] = "tracked_vehicle_ids"
    assert _prepare(scenario_config=scenario, targets=targets)


def test_obvious_condition_contradictions_warn(aim_service_types, caplog):
    scenario = {"scenario": {"rsu_list": [{"behavior_services": {"aim_server": {}}}]}}
    source = {"kind": "snapshot", "service_type": "aim_server", "field": "tracked_vehicle_count", "node_id": "rsu"}
    bounds = [{"source": source, "verb": verb, "value": value} for verb, value in (("gt", 5), ("lt", 2))]
    _prepare(scenario_config=scenario, requirements={"all": bounds})
    assert "attack.requirements.all[1].value" in caplog.text
    condition = {"source": {"kind": "stage", "stage_id": "example", "field": "status"}, "verb": "eq", "value": "success"}
    _prepare([{"id": "example", "type": "sniffer", "stage_start_trigger": condition}])
    assert "attack.stages[0].stage_start_trigger.value" in caplog.text


def test_condition_value_types_that_cannot_match_warn(aim_service_types, caplog):
    scenario = {"scenario": {"rsu_list": [{"behavior_services": {"aim_server": {}}}]}}
    source = {"kind": "snapshot", "service_type": "aim_server", "field": "tracked_vehicle_count"}
    _prepare(scenario_config=scenario, requirements={"source": source, "verb": "eq", "value": "1"})
    assert "attack.requirements.value" in caplog.text
    caplog.clear()
    source["field"] = "tracked_vehicle_ids"
    _prepare(scenario_config=scenario, requirements={"source": source, "verb": "added", "value": 1})
    assert "attack.requirements.value" in caplog.text


@pytest.mark.parametrize(("key", "value"), [("kind", "unknown"), ("selection", "one")])
def test_target_modes_are_checked_during_preparation(key, value):
    target = {"kind": "service_state_field", "source": {"kind": "snapshot", "field": "node_id"}, "resolve_to": {"node_type": "vehicle"}}
    with pytest.raises(ValueError) as exc:
        _prepare(targets={**target, key: value})
    assert str(exc.value).startswith(f"example.yaml: attack.targets.{key}:")


@pytest.mark.parametrize(
    ("source_kind", "node_type", "key"),
    [("snapshott", "vehicle", "source.kind"), ("stage", "rsu", "source.kind"), ("snapshot", "vehcile", "resolve_to.node_type")],
)
def test_target_source_and_destination_report_the_invalid_key(source_kind, node_type, key):
    target = {"kind": "service_state_field", "source": {"kind": source_kind, "field": "node_id"}, "resolve_to": {"node_type": node_type}}
    with pytest.raises(ValueError) as exc:
        _prepare(targets=target)
    assert str(exc.value).startswith(f"example.yaml: attack.targets.{key}:")


def test_target_services_and_contracts_are_scoped_to_the_requested_node_type():
    scenario = {"scenario": {"single_cav_list": [{"behavior_services": {"movement_controller": {}}}], "rsu_list": [{}]}}
    resolve_to = {"node_type": "rsu", "service_type": "movement_controller"}
    target = {"kind": "service_state_field", "source": {"kind": "snapshot", "field": "node_id"}, "resolve_to": resolve_to}
    stage = {**_rewrite("payload"), "capabilities": "command.observe"}
    with pytest.raises(ValueError, match=r"attack\.targets\.resolve_to\.service_type:"):
        _prepare([stage], scenario_config=scenario, targets=target)
    del resolve_to["service_type"]
    with pytest.raises(ValueError, match=r"attack\.stages\[0\]\.capabilities:"):
        _prepare([stage], scenario_config=scenario, targets=target)
    resolve_to["node_type"] = "vehicle"
    assert _prepare([stage], scenario_config=scenario, targets=target)
    target["source"] = {"kind": "snapshot", "node_type": "rsu", "service_type": "movement_controller"}
    with pytest.raises(ValueError, match=r"attack\.targets\.source\.service_type:"):
        _prepare([stage], scenario_config=scenario, targets=target)


@pytest.fixture
def aim_service_types(monkeypatch):
    # The test suite disables service discovery. Register the real services without loading prediction models.
    monkeypatch.setattr(BehaviorServiceRegistry, "_registry", BehaviorServiceRegistry._registry.copy())
    model_api = SimpleNamespace(AIMModel=object, get_model=Mock(side_effect=AssertionError("Model loaded during validation")))
    with patch.dict(sys.modules, {"AIM": model_api}):
        from opencda.core.application.behavior.services.aim_client import AIMClient
        from opencda.core.application.behavior.services.aim_server import AIMServer

        yield AIMClient, AIMServer


def test_bundled_definitions_pass_static_checks(aim_service_types):
    directory = Path(__file__).resolve().parents[1] / "opencda/core/attack/adversary_framework/attacks"
    definitions = [(file, read_config_mapping(file)) for file in sorted(directory.glob("*/config.yaml"))]
    config = {"scenario": {"single_cav_list": [{"behavior_services": {"aim_client": {}}}], "rsu_list": [{"behavior_services": {"aim_server": {}}}]}}
    attacks = AttackManager().prepare_attacks(definitions, config)
    assert [attack.attack_name for attack in attacks] == [file.parent.name for file, _ in definitions]
    assert attacks


def _prepare_for_services(stage, services, name=None):
    target = {"kind": "service_state_field", "source": {"kind": "snapshot", "field": "node_id"}, "resolve_to": {"node_type": "vehicle"}}
    if name is not None:
        target["resolve_to"]["service_type"] = name
    definition = {"attack": {"name": "example", "targets": target, "stages": [stage]}}
    scenario = {"scenario": {"single_cav_list": [{}]}, "vehicle_base": {"behavior_services": {service: {} for service in services}}}
    return AttackManager().prepare_attacks([(Path("example.yaml"), definition)], scenario)


def test_stage_checks_operations_on_one_service():
    stage = {"id": "test", "type": "spoofer", "params": _rewrite("payload")["params"]}
    services = ["movement_controller", "default_movement_request"]
    with pytest.raises(ValueError, match=r"attack\.stages\[0\]\.type:"):
        _prepare_for_services(stage, services)
    stage["capabilities"] = ["command.observe", "command.submit"]
    with pytest.raises(ValueError, match=r"attack\.stages\[0\]\.capabilities:"):
        _prepare_for_services(stage, services)
    stage["capabilities"] = "command.submit"
    with pytest.raises(ValueError, match=r"attack\.stages\[0\]\.capabilities:"):
        _prepare_for_services(stage, services, "movement_controller")
    assert _prepare_for_services(stage, services)


@pytest.mark.parametrize(
    ("path", "valid"),
    [
        ("payload.speed", True),
        ("payload.speeed", False),
        ("payload.speed.x", False),
        ("payload.waypoints[20].location.x", True),
        ("payload.waypoints[20].location.unknown", False),
        ("payload.waypoints.location", False),
    ],
)
def test_known_message_fields_before_simulation(aim_service_types, path, valid):
    stage = _rewrite(path)
    if valid:
        assert _prepare_for_services(stage, ["aim_client"], "aim_client")
    else:
        with pytest.raises(ValueError) as exc:
            _prepare_for_services(stage, ["aim_client"], "aim_client")
        assert str(exc.value).startswith("example.yaml: attack.stages[0].params.rewrites[0].path:")


@pytest.mark.parametrize("operation", ["add", "multiply"])
@pytest.mark.parametrize("path", ["payload.speed", "payload.waypoints[0].location.x"])
def test_arithmetic_rewrites_require_compatible_operands(aim_service_types, path, operation):
    stage = _rewrite(path)
    rule = stage["params"]["rewrites"][0]
    rule.update(operation=operation, value="5")
    with pytest.raises(ValueError, match=r"example\.yaml: attack\.stages\[0\]\.params\.rewrites\[0\]\.value:") as exc:
        _prepare_for_services(stage, ["aim_client"])
    assert "(float) with a value of type 'str'" in str(exc.value)
    rule["value"] = 5
    assert _prepare_for_services(stage, ["aim_client"])


def test_arithmetic_on_an_optional_field_checks_every_declared_type():
    stage = {**_rewrite("payload.target_speed"), "capabilities": "command.observe"}
    rule = stage["params"]["rewrites"][0]
    rule.update(operation="multiply", value="5")
    with pytest.raises(ValueError, match=r"rewrites\[0\]\.value:"):
        _prepare_for_services(stage, ["movement_controller"])
    rule["value"] = 0
    with pytest.raises(ValueError, match=r"rewrites\[0\]\.value:.*NoneType"):
        _prepare_for_services(stage, ["movement_controller"])
    stage["params"]["rewrites"].insert(0, {"path": "payload.target_speed", "value": 1.0})
    assert _prepare_for_services(stage, ["movement_controller"])


@pytest.mark.parametrize(
    ("operation", "value", "valid"), [("add", 1, False), ("add", "-suffix", True), ("multiply", "1", False), ("multiply", 1, True)]
)
def test_arithmetic_rewrites_check_envelope_fields(operation, value, valid):
    stage = _rewrite("src_owner_id")
    stage["params"]["rewrites"][0].update(operation=operation, value=value)
    if valid:
        assert _prepare([stage])
    else:
        with pytest.raises(ValueError, match=r"example\.yaml: attack\.stages\[0\]\.params\.rewrites\[0\]\.value:.*src_owner_id.*str"):
            _prepare([stage])


def test_arithmetic_uses_the_value_type_from_a_previous_rewrite(aim_service_types):
    stage = _rewrite("payload.speed")
    rules = stage["params"]["rewrites"]
    rules[0]["value"] = "abc"
    rules.append({"path": "payload.speed", "operation": "multiply", "value": 2})
    assert _prepare_for_services(stage, ["aim_client"])
    rules.append({"path": "payload.speed", "operation": "add", "value": 1})
    with pytest.raises(ValueError, match=r"rewrites\[2\]\.value:"):
        _prepare_for_services(stage, ["aim_client"])


def test_arithmetic_uses_nested_values_written_by_an_earlier_rule():
    stage = {**_rewrite("payload.target_location"), "capabilities": "command.observe"}
    rules = stage["params"]["rewrites"]
    rules[0]["value"] = {"x": "one"}
    rules.append({"path": "payload.target_location.x", "operation": "add", "value": 1})
    with pytest.raises(ValueError, match=r"rewrites\[1\]\.value:"):
        _prepare_for_services(stage, ["movement_controller"])
    rules[0]["value"] = {"x": 1}
    assert _prepare_for_services(stage, ["movement_controller"])
    rules[0]["value"] = {}
    with pytest.raises(ValueError, match=r"rewrites\[1\]\.path:"):
        _prepare_for_services(stage, ["movement_controller"])


def test_custom_operator_rejects_an_incompatible_operand(aim_service_types):
    stage = _rewrite("payload.position")
    stage["params"]["rewrites"][0].update(operation="add", value=1)
    with pytest.raises(ValueError, match=r"rewrites\[0\]\.value:.*Location"):
        _prepare_for_services(stage, ["aim_client"])


def test_rewrite_checks_direct_assignment_to_an_immutable_sequence(aim_service_types, monkeypatch):
    from opencda.core.application.behavior.capability import Capability
    from opencda.core.application.behavior.types import Location

    @dataclass
    class Message:
        points: tuple[Location, ...]

    _, server = aim_service_types
    monkeypatch.setattr(server, "capability_payloads", {Capability.RESPONSE_SUBMIT: Message})
    stage = {**_rewrite("payload.points[0]"), "capabilities": "response.submit"}
    with pytest.raises(ValueError, match=r"rewrites\[0\]\.path:.*cannot be written"):
        _prepare_for_services(stage, ["aim_server"])
    stage["params"]["rewrites"][0]["path"] = "payload.points[0].x"
    assert _prepare_for_services(stage, ["aim_server"])


def test_dynamic_candidate_does_not_hide_errors_in_known_message_types(monkeypatch, caplog):
    monkeypatch.setitem(BehaviorServiceRegistry._registry, "extension", type("Extension", (), {}))
    stage = {**_rewrite("payload.target_speed"), "capabilities": "command.observe"}
    rule = stage["params"]["rewrites"][0]
    rule.update(operation="add", value=1)
    with pytest.raises(ValueError, match=r"rewrites\[0\]\.value:"):
        _prepare_for_services(stage, ["movement_controller", "extension"])
    rule["path"] = "payload.target_speeed"
    assert _prepare_for_services(stage, ["movement_controller", "extension"])
    assert "example.yaml: attack.stages[0].params.rewrites[0].path" in caplog.text


def test_payload_depends_on_service_for_the_same_operation(aim_service_types):
    services = ["aim_client", "self_informer"]
    stage = _rewrite("payload.speed")
    with pytest.raises(ValueError, match=r"rewrites\[0\]\.path:"):
        _prepare_for_services(stage, services, "self_informer")
    assert _prepare_for_services(stage, services)


def test_optional_payload_field_uses_declared_type(caplog):
    from opencda.core.application.behavior.services.movement_controller import MovementControllerRequestMessage

    stage = _rewrite("payload.target_location.x")
    stage["capabilities"] = "command.observe"
    (attack,) = _prepare_for_services(stage, ["movement_controller"])
    message = TransportMessage("source", "service", "destination", "service", MovementControllerRequestMessage(None, None))
    assert attack.stages[0]._spoof_message(message) is message
    assert "example.yaml: attack.stages[0].params.rewrites[0].path" in caplog.text
    assert "field is absent" in caplog.text
    stage["params"]["rewrites"][0]["path"] = "payload.target_location.unknown"
    with pytest.raises(ValueError, match=r"rewrites\[0\]\.path:"):
        _prepare_for_services(stage, ["movement_controller"])


def test_builtin_contracts_match_exposed_operations(aim_service_types):
    for name in ("aim_client", "aim_server", "movement_controller", "default_movement_request", "self_informer"):
        cls = BehaviorServiceRegistry.get_service_class(name)
        # Binding the methods needs no model, participant or initialized world.
        assert cls.capability_payloads.keys() == object.__new__(cls).capability_bindings.keys()


@pytest.mark.parametrize(
    ("stage", "error_path"),
    [
        ({"id": "missing_type"}, "type"),
        (_rewrite("payload"), "type"),  # The default operation is unavailable on this service.
        ({**_rewrite("payload.target_speeed"), "capabilities": "command.observe"}, "params.rewrites[0].path"),
    ],
)
def test_invalid_attack_file_stops_scenario_before_world_creation(tmp_path, monkeypatch, stage, error_path):
    import yaml
    from omegaconf import OmegaConf

    from opencda.scenario_testing.scenario import Scenario

    monkeypatch.chdir(tmp_path)
    config = Path("opencda/core/attack/adversary_framework/attacks/example/config.yaml")
    config.parent.mkdir(parents=True)
    target = {
        "kind": "service_state_field",
        "source": {"kind": "snapshot", "field": "node_id"},
        "resolve_to": {"node_type": "vehicle", "service_type": "movement_controller"},
    }
    config.write_text(yaml.safe_dump({"attack": {"name": "example", "targets": target, "stages": [stage]}}), encoding="utf-8")
    init_world = Mock()
    monkeypatch.setattr(Scenario, "_init_scenario_manager", init_world)
    if not hasattr(OmegaConf, "to_container"):
        monkeypatch.setattr(OmegaConf, "to_container", lambda data, **kwargs: dict(data), raising=False)
    with pytest.raises(ValueError) as exc:
        params = {"attacks": ["example"], "scenario": {"single_cav_list": [{"behavior_services": {"movement_controller": {}}}]}}
        Scenario(SimpleNamespace(test_scenario="example"), OmegaConf.create(params), current_time="test")
    assert f"{config}: attack.stages[0].{error_path}:" in str(exc.value)
    scenario_file = tmp_path / "custom-scenario.yaml"
    with pytest.raises(RuntimeError) as exc:
        Scenario(SimpleNamespace(test_scenario="example"), OmegaConf.create({"attacks": [None]}), current_time="test", config_file=scenario_file)
    assert f"{scenario_file}: attacks[0]:" in str(exc.value)
    init_world.assert_not_called()


def test_missing_runtime_value_is_reported_once_without_disabling_the_rule(caplog):
    stage = _prepare([_rewrite("payload.items[0]")]).stages[0]
    message = TransportMessage("source", "service", "destination", "service", {"items": []})
    assert stage._spoof_message(message) is message
    assert stage._spoof_message(message) is message
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "example.yaml: attack.stages[0].params.rewrites[0].path" in warnings[0]
    assert "index is out of range" in warnings[0]
    message.payload["items"].append(1)
    assert stage._spoof_message(message).payload["items"] == [0]
    assert message.payload["items"] == [1]


def test_runtime_arithmetic_error_names_the_rule_when_the_payload_type_is_dynamic():
    stage = _rewrite("payload.sample")
    stage["params"]["rewrites"][0].update(operation="add", value=1)
    (attack_stage,) = _prepare([stage]).stages
    message = TransportMessage("source", "service", "destination", "service", {"sample": "text"})
    with pytest.raises(TypeError, match=r"example\.yaml: attack\.stages\[0\]\.params\.rewrites\[0\]\.value:"):
        attack_stage._spoof_message(message)
