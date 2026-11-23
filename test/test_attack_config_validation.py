"""Configuration checks that run without a simulation or live services."""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from opencda.core.attack.adversary_framework import AttackManager
from opencda.scenario_testing.utils.yaml_utils import read_config_mapping


def _prepare(stages=(), **attack):
    config = {"attack": {"name": "example", "stages": list(stages), **attack}}
    return AttackManager().prepare_attacks([(Path("example.yaml"), config)], {})[0]


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
        ([_rewrite("no_such_field")], "stages[0].params.rewrites[0].path"),
        ([_rewrite("dst_owner_id.child")], "stages[0].params.rewrites[0].path"),
        ([_rewrite("payload[bad]")], "stages[0].params.rewrites[0].path"),
    ],
)
def test_config_error_identifies_the_original_key(stages, path):
    with pytest.raises(ValueError) as exc:
        _prepare(stages)
    assert str(exc.value).startswith(f"example.yaml: attack.{path}:")


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
    attack = _prepare([{"id": "later", "type": "sniffer"}], **{key: _reference("later")})
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


def test_missing_nested_required_key():
    with pytest.raises(ValueError) as exc:
        _prepare(stop_trigger={"any": [{"source": {}, "verb": "exists"}]})
    assert "example.yaml: attack.stop_trigger.any[0].source.kind:" in str(exc.value)


def test_unreachable_trigger_warns_and_allows_preparation(caplog):
    source = {"kind": "snapshot", "service_type": "aim_server", "field": "tracked_vehicle_count"}
    attack = _prepare(stop_trigger={"source": source, "verb": "lt", "value": 0})
    assert attack.attack_name == "example"
    assert "example.yaml: attack.stop_trigger.value" in caplog.text
    caplog.clear()
    _prepare(stop_trigger={"source": source, "verb": "lte", "value": 0})
    assert not caplog.records


def test_conflicting_equalities_warn_only_for_a_single_value(caplog):
    source = {"kind": "snapshot", "field": "example"}
    conditions = [{"source": source, "verb": "eq", "value": value} for value in (1, 2)]
    _prepare(start_trigger={"all": conditions})
    assert not caplog.records  # Different participants can satisfy the two predicates.
    source["node_id"] = "one_node"
    _prepare(start_trigger={"any": conditions})
    assert not caplog.records
    _prepare(start_trigger={"all": conditions})
    assert "example.yaml: attack.start_trigger.all[1].value" in caplog.text


def test_only_existing_participants_contribute_service_declarations():
    targets = {"kind": "service_state_field", "source": {"kind": "snapshot"}, "resolve_to": {"node_type": "vehicle", "service_type": "example"}}
    definition = [(Path("example.yaml"), {"attack": {"name": "example", "targets": targets}})]
    config = {"vehicle_base": {"behavior_services": {"example": {}}}, "scenario": {"rsu_list": [{}]}}
    with pytest.raises(ValueError) as exc:
        AttackManager().prepare_attacks(definition, config)
    assert "example.yaml: attack.targets.resolve_to.service_type" in str(exc.value)
    config["scenario"]["platoon_list"] = [{"members": [{}]}]
    assert len(AttackManager().prepare_attacks(definition, config)) == 1


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
    target = {"kind": "service_state_field", "source": {"kind": "snapshot"}, "resolve_to": {"node_type": "vehicle"}}
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
        "source": {"kind": "snapshot"},
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
