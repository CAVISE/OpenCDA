"""Checks on declarations available before a simulation is created."""

from __future__ import annotations

import logging
import operator
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from opencda.core.application.behavior.capability import CapabilityPayloads
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.scenario_testing.types import NodeSnapshot

from .attack import Attack
from .condition_evaluator import CONDITION_VERBS, ORDERED_COMPARISONS, ConditionRuntime, compare_value
from .models import AttackSpec, ConditionSpec, RuntimeStatus, StageRuntime, TriggerSourceSpec
from .models.config_error import ConfigError
from .stages.spoofer.stage import RewriteRule, SpooferStage
from .utils import TARGET_KIND, TARGET_SELECTIONS

logger = logging.getLogger("cavise.opencda.opencda.core.attack.adversary_framework.validation")


def declared_service_names(config: Mapping[str, Any]) -> dict[str, set[str]]:
    """Collect service declarations of existing participants by node type."""
    scenario = config.get("scenario") or {}
    vehicles = list(scenario.get("single_cav_list") or ())
    for platoon in scenario.get("platoon_list") or ():
        vehicles.extend(platoon.get("members", config.get("platoon_base", {}).get("members", ())))

    services: dict[str, set[str]] = {"vehicle": set(), "rsu": set()}
    for node_type, participants in (("vehicle", vehicles), ("rsu", scenario.get("rsu_list") or ())):
        if participants:
            names = services[node_type]
            for base in (config.get("agent_base", {}), config.get(f"{node_type}_base", {})):
                names.update(base.get("behavior_services", {}))
            for participant in participants:
                names.update(participant.get("behavior_services", {}))
    return services


def _services_for_node(services: Mapping[str, set[str]], node_type: str | None) -> set[str]:
    if node_type is not None and node_type in services:
        return services[node_type]
    return set().union(*services.values())


def _check_service_reference(services: Mapping[str, set[str]], node_type: str | None, name: str | None, path: str) -> None:
    if name is not None and name not in _services_for_node(services, node_type):
        owner = f"node type '{node_type}'" if node_type in services else "any scenario participant"
        raise ConfigError(path, f"Service '{name}' is not declared for {owner}.")


def _check_snapshot_source(source: TriggerSourceSpec, services: Mapping[str, set[str]], path: str) -> None:
    if source.node_type is not None:
        _check_node_type(source.node_type, f"{path}.node_type")
    _check_service_reference(services, source.node_type, source.service_type, f"{path}.service_type")


def _check_node_type(node_type: str, path: str) -> None:
    if node_type not in {"vehicle", "rsu"}:
        raise ConfigError(path, f"Unknown node type '{node_type}'. Expected 'vehicle' or 'rsu'.")


def validate_spec(spec: AttackSpec, services: Mapping[str, set[str]], origin: str) -> None:
    targets = spec.targets
    if targets is not None:
        if targets.kind != TARGET_KIND:
            raise ConfigError("targets.kind", f"Unsupported target resolution kind '{targets.kind}'.")
        if targets.selection not in TARGET_SELECTIONS:
            raise ConfigError("targets.selection", f"Unsupported target selection '{targets.selection}'.")
        if targets.source.kind != "snapshot":
            raise ConfigError("targets.source.kind", f"Unsupported target source '{targets.source.kind}'. Expected 'snapshot'.")
        _check_snapshot_source(targets.source, services, "targets.source")
        _check_node_type(targets.resolve_to_node_type, "targets.resolve_to.node_type")
        _check_service_reference(services, targets.resolve_to_node_type, targets.resolve_to_service_name, "targets.resolve_to.service_type")

    stage_ids = {stage.id for stage in spec.stages}
    for key in ("requirements", "start_trigger", "stop_trigger"):
        condition = getattr(spec, key)
        if condition is not None:
            _validate_condition(condition, key, spec.name, stage_ids, origin, services)
    for index, stage in enumerate(spec.stages):
        for key in ("requirements", "stage_start_trigger", "stage_stop_trigger"):
            condition = getattr(stage, key)
            if condition is not None:
                _validate_condition(condition, f"stages[{index}].{key}", spec.name, stage_ids, origin, services)


def validate_service_contracts(attack: Attack, services: Mapping[str, set[str]]) -> None:
    """Check the normalized stage operations and paths against declared simulation service types."""
    targets = attack.spec.targets
    if targets is None or not attack.stages:
        return
    name_filter = targets.resolve_to_service_name
    names = {name_filter} if name_filter is not None else _services_for_node(services, targets.resolve_to_node_type)
    contracts: list[CapabilityPayloads] = []
    has_dynamic_bindings = False
    for name in sorted(names):
        try:
            service = BehaviorServiceRegistry.get_service_class(name)
        except KeyError as exc:
            key = "targets.resolve_to.service_type" if name_filter is not None else "targets.resolve_to"
            raise ConfigError(key, f"Service type '{name}' is not registered.") from exc
        contract = getattr(service, "capability_payloads", None)
        if contract is None:
            has_dynamic_bindings = True
        else:
            contracts.append(contract)
    if has_dynamic_bindings:
        # Compatibility with this extension cannot be ruled out until its bindings exist.
        return

    for index, runtime in enumerate(attack.stage_runtimes):
        stage = runtime.stage
        required = set(stage.capabilities)
        compatible = [contract for contract in contracts if required.issubset(contract)]
        location = f"stages[{index}]"
        if not compatible:
            key = "type" if runtime.spec.capabilities is None else "capabilities"
            operations = ", ".join(capability.value for capability in stage.capabilities)
            raise ConfigError(f"{location}.{key}", f"No declared target service provides the required operations together ({operations}).")
        if isinstance(stage, SpooferStage):
            payloads = {payload for contract in compatible for capability in required if (payload := contract[capability]) is not None}
            _validate_payload_rewrites(stage.rewrite_rules, payloads, location)


def _validate_payload_rewrites(rules: tuple[RewriteRule, ...], payloads: set[type], location: str) -> None:
    # Each message sees the rules in order. A previous rewrite can change the type used by a later one.
    written: dict[type, dict[tuple[str, ...], tuple[Any, ...]]] = {payload: {} for payload in payloads}
    for index, rule in enumerate(rules):
        if rule.path[0] != "payload":
            continue
        key = f"{location}.params.rewrites[{index}]"
        found = False
        for payload, types_by_path in written.items():
            types = _field_types(payload, rule.path[1:])
            if not types:
                continue
            found = True
            if rule.operation == "set":
                results = (type(rule.value),)
            else:
                for parent in reversed(types_by_path):
                    if rule.path[: len(parent)] == parent:
                        types = tuple(leaf for schema in types_by_path[parent] for leaf in _field_types(schema, rule.path[len(parent) :]))
                        break
                if not types:
                    continue  # An earlier rewrite removed the field. The usual missing-path diagnostic applies.
                results = tuple(result for field_type in types if (result := _arithmetic_result_type(field_type, rule)) is not None)
                if not results:
                    names = ", ".join(sorted({field_type.__name__ for field_type in types}))
                    raise ConfigError(
                        f"{key}.value",
                        f"Cannot apply '{rule.operation}' to '{'.'.join(rule.path)}' ({names}) with a value of type '{type(rule.value).__name__}'.",
                    )
            # Keep the most recent assignment last, including when the same path is written twice.
            types_by_path.pop(rule.path, None)
            types_by_path[rule.path] = results
        if not found:
            raise ConfigError(f"{key}.path", f"Field path '{'.'.join(rule.path)}' is absent from the declared message types.")


def _arithmetic_result_type(field_type: Any, rule: RewriteRule) -> Any:
    """Check Python's built-in operand types without using message data or invoking extension code."""
    left = get_origin(field_type) or field_type
    right = type(rule.value)
    builtin_types = (bool, int, float, complex, str, bytes, list, tuple, dict, set, frozenset, type(None))
    if left in builtin_types and right in builtin_types:
        operation = operator.add if rule.operation == "add" else operator.mul
        try:
            return type(operation(left(), right()))
        except TypeError:
            return None
    if is_dataclass(left) and right in builtin_types:
        method = "__add__" if rule.operation == "add" else "__mul__"
        if not hasattr(left, method) and not hasattr(left, "__index__"):
            return None
    return Any


def _field_types(schema: Any, path: tuple[str, ...]) -> tuple[Any, ...]:
    """Return possible leaf types, an empty tuple for a missing path, or Any for runtime data."""
    origin = get_origin(schema)
    args = get_args(schema)
    if origin in (Union, UnionType):
        return tuple(leaf for member in args for leaf in _field_types(member, path))
    if schema is Any or schema is object:
        return (Any,)
    if not path:
        return (schema,)

    field, rest = path[0], path[1:]
    if is_dataclass(schema):
        if field not in {item.name for item in fields(schema)}:
            return ()
        try:
            annotations = get_type_hints(schema)
        except (NameError, TypeError):
            return (Any,)
        return _field_types(annotations[field], rest)

    container = origin or schema
    if isinstance(container, type) and issubclass(container, Mapping):
        return _field_types(args[1], rest) if len(args) == 2 else (Any,)
    if isinstance(container, type) and issubclass(container, (str, bytes, int, float, complex, type(None))):
        return ()
    if isinstance(container, type) and issubclass(container, Sequence):
        if not field.isdigit():
            return ()
        if container is tuple and args and args[-1] is not Ellipsis:
            index = int(field)
            return _field_types(args[index], rest) if index < len(args) else ()
        return _field_types(args[0], rest) if args else (Any,)
    return (Any,)


def _condition_field_types(source: TriggerSourceSpec) -> tuple[Any, ...]:
    if source.field is not None and not isinstance(source.field, str):
        return (Any,)
    if source.kind in {"attack", "stage"} and source.field == "status":
        return (str,)  # The evaluator exposes the status enum's string value.
    if source.kind == "attack":
        if source.field is None:
            return (Attack,)
        return (get_type_hints(ConditionRuntime).get(source.field, Any),)

    schema: Any = StageRuntime if source.kind == "stage" else NodeSnapshot
    if source.kind == "snapshot" and source.service_type is not None:
        try:
            service = BehaviorServiceRegistry.get_service_class(source.service_type)
            schema = get_type_hints(service.get_state).get("return", Any)
        except (KeyError, NameError, TypeError):
            return (Any,)
    return _field_types(schema, (source.field,) if source.field is not None else ())


def _validate_ordered_comparison(source: TriggerSourceSpec, verb: str, value: Any, path: str) -> None:
    if value is None:
        raise ConfigError(path, f"Comparison '{verb}' requires a non-null value.")
    types = _condition_field_types(source)
    if not types:
        return  # Values outside the declared schema are only available at runtime.
    for annotation in types:
        kind = get_origin(annotation) or annotation
        if kind is RuntimeStatus:
            kind = str
        if kind in (bool, int, float, str, bytes, list, tuple, dict, set, frozenset, complex, type(None)):
            try:
                compare_value(kind(), verb, value)
            except TypeError:
                continue
            return
        if not is_dataclass(kind) and kind is not Attack:
            return
        # A custom comparison can depend on the value in the simulation.
        if any(getattr(kind, method) is not getattr(object, method) for method in ("__lt__", "__le__", "__gt__", "__ge__")):
            return
    names = ", ".join(getattr(annotation, "__name__", str(annotation)) for annotation in types)
    raise ConfigError(path, f"Cannot apply '{verb}' to '{source.field or source.kind}' ({names}) with a value of type '{type(value).__name__}'.")


def _validate_condition(
    condition: ConditionSpec, path: str, attack_name: str, stage_ids: set[str], origin: str, services: Mapping[str, set[str]]
) -> None:
    for index, child in enumerate(condition.all):
        source = child.source
        if source is None or child.verb != "eq":
            continue
        single_value = (
            source.kind == "attack"
            or (source.kind == "stage" and source.stage_id is not None)
            or (source.kind == "snapshot" and source.node_id is not None)
        )
        if single_value and any(
            sibling.source == source and sibling.verb == "eq" and sibling.value != child.value for sibling in condition.all[:index]
        ):
            logger.warning("%s.%s.all[%d].value: the same source must equal two different values in this 'all' group.", origin, path, index)

    for group in ("all", "any"):
        for index, child in enumerate(getattr(condition, group)):
            _validate_condition(child, f"{path}.{group}[{index}]", attack_name, stage_ids, origin, services)

    source = condition.source
    if source is None:
        return
    if source.kind not in CONDITION_VERBS:
        raise ConfigError(f"{path}.source.kind", f"Unsupported condition source '{source.kind}'.")
    if condition.verb not in CONDITION_VERBS[source.kind]:
        raise ConfigError(f"{path}.verb", f"Unsupported verb '{condition.verb}' for source '{source.kind}'.")
    if source.kind == "snapshot":
        _check_snapshot_source(source, services, f"{path}.source")
    if condition.verb in ORDERED_COMPARISONS:
        _validate_ordered_comparison(source, condition.verb, condition.value, f"{path}.value")
    if source.kind in {"attack", "stage"}:
        if source.attack_name is not None and source.attack_name != attack_name:
            raise ConfigError(f"{path}.source.attack_name", "Runtime conditions can only reference the current attack.")
        if source.kind == "stage" and source.stage_id is not None and source.stage_id not in stage_ids:
            raise ConfigError(f"{path}.source.stage_id", f"Unknown stage id '{source.stage_id}'.")
        if source.field == "status" and condition.verb in {"eq", "became"}:
            if condition.value not in tuple(status.value for status in RuntimeStatus if status != RuntimeStatus.STARTED):
                logger.warning("%s.%s.value: this status cannot be observed by the condition.", origin, path)

    if source.kind == "snapshot" and source.service_type == "aim_server" and source.field == "tracked_vehicle_count":
        value = condition.value
        if not isinstance(value, (int, float)):
            return
        if (condition.verb == "lt" and value <= 0) or (condition.verb in {"lte", "eq", "became"} and value < 0):
            logger.warning("%s.%s.value: the tracked vehicle count is always non-negative. This condition cannot be satisfied.", origin, path)
