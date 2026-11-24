"""Checks on declarations available before a simulation is created."""

from __future__ import annotations

import logging
import operator
from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass, fields, is_dataclass
from inspect import getattr_static, isabstract
from types import MappingProxyType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from opencda.core.application.behavior.capability import CapabilityPayloads
from opencda.core.application.behavior.registry import BehaviorServiceRegistry
from opencda.core.application.behavior.transport_message import TransportMessage
from opencda.scenario_testing.types import NodeSnapshot

from .attack import Attack
from .condition_evaluator import CONDITION_VERBS, ORDERED_COMPARISONS, compare_value
from .models import AttackSpec, ConditionSpec, RuntimeStatus, StageRuntime, TriggerSourceSpec
from .models.config_error import ConfigError
from .stages.spoofer.stage import RewriteRule, SpooferStage
from .utils import TARGET_KIND, TARGET_SELECTIONS

logger = logging.getLogger("cavise.opencda.opencda.core.attack.adversary_framework.validation")

_COLLECTION_TRANSITIONS = {"added", "removed"}
_DELTA_TRANSITIONS = {"increased", "decreased"}
_BUILTIN_OPERAND_TYPES = (bool, int, float, complex, str, bytes, list, tuple, dict, set, frozenset, type(None))
_NUMBER_TYPES = (bool, int, float, complex)
_ORDER_METHODS = {"gt": ("__gt__", "__lt__"), "gte": ("__ge__", "__le__"), "lt": ("__lt__", "__gt__"), "lte": ("__le__", "__ge__")}


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


def _check_node_type(node_type: str, path: str) -> None:
    if node_type not in {"vehicle", "rsu"}:
        raise ConfigError(path, f"Unknown node type '{node_type}'. Expected 'vehicle' or 'rsu'.")


def validate_spec(spec: AttackSpec, services: Mapping[str, set[str]], origin: str) -> None:
    targets = spec.targets
    if spec.start_trigger is not None and spec.stages and targets is None:
        raise ConfigError("targets", "A runnable attack with stages requires a target source.")
    if spec.stages and spec.start_trigger is None and targets is not None:
        logger.warning("%s.start_trigger: no start trigger is configured.", origin)
    if spec.start_trigger is not None and targets is not None and not spec.stages:
        logger.warning("%s.stages: no stages are configured.", origin)
    if targets is not None:
        if targets.kind != TARGET_KIND:
            raise ConfigError("targets.kind", f"Unsupported target resolution kind '{targets.kind}'.")
        if targets.selection not in TARGET_SELECTIONS:
            raise ConfigError("targets.selection", f"Unsupported target selection '{targets.selection}'.")
        if targets.source.kind != "snapshot":
            raise ConfigError("targets.source.kind", f"Unsupported target source '{targets.source.kind}'. Expected 'snapshot'.")
        source_types = _validate_source(targets.source, services, "targets.source")
        if not any(_can_supply_node_ids(schema) for schema in source_types):
            raise ConfigError("targets.source.field", "Expected a field that can supply a node ID string or a collection of node ID strings.")
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
                start_stage = stage.id if key != "stage_stop_trigger" else None
                _validate_condition(condition, f"stages[{index}].{key}", spec.name, stage_ids, origin, services, start_stage)


def validate_service_contracts(attack: Attack, services: Mapping[str, set[str]], origin: str) -> None:
    """Check the normalized stage operations and paths against declared simulation service types."""
    for index, stage in enumerate(attack.stages):
        if isinstance(stage, SpooferStage):
            _validate_rewrites(stage.rewrite_rules, {TransportMessage}, f"stages[{index}]", payload=False)

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
    for index, runtime in enumerate(attack.stage_runtimes):
        stage = runtime.stage
        required = set(stage.capabilities)
        compatible = [contract for contract in contracts if required.issubset(contract)]
        location = f"stages[{index}]"
        if not compatible and not has_dynamic_bindings:
            key = "type" if runtime.spec.capabilities is None else "capabilities"
            operations = ", ".join(capability.value for capability in stage.capabilities)
            raise ConfigError(f"{location}.{key}", f"No declared target service provides the required operations together ({operations}).")
        if isinstance(stage, SpooferStage):
            payloads = {payload for contract in compatible for capability in required if (payload := contract[capability]) is not None}
            _validate_rewrites(stage.rewrite_rules, payloads, location, payload=True, dynamic_origin=origin if has_dynamic_bindings else None)


@dataclass(frozen=True)
class _ConfiguredValue:
    value: Any


def _rewritten_field_types(schema: type, path: tuple[str, ...], written: dict[tuple[str, ...], tuple[Any, ...]]) -> tuple[Any, ...]:
    for parent in reversed(written):
        if path[: len(parent)] == parent:
            return tuple(leaf for variant in written[parent] for leaf in _field_types(variant, path[len(parent) :]))
    return _field_types(schema, path)


def _validate_rewrites(
    rules: tuple[RewriteRule, ...], schemas: set[type], location: str, *, payload: bool, dynamic_origin: str | None = None
) -> None:
    # Each message sees the rules in order. A previous rewrite can change the type used by a later one.
    written: dict[type, dict[tuple[str, ...], tuple[Any, ...]]] = {schema: {} for schema in schemas}
    for index, rule in enumerate(rules):
        if (rule.path[0] == "payload") != payload:
            continue
        key = f"{location}.params.rewrites[{index}]"
        path = rule.path[1:] if payload else rule.path
        found = False
        for schema, types_by_path in written.items():
            types = _rewritten_field_types(schema, path, types_by_path)
            if not types:
                continue
            found = True
            if path:
                for annotation in _rewritten_field_types(schema, path[:-1], types_by_path):
                    if _is_readonly_field(annotation, path[-1]) and _field_types(annotation, (path[-1],)):
                        raise ConfigError(f"{key}.path", f"Field '{'.'.join(rule.path)}' cannot be written on the declared message type.")
            if rule.operation == "set":
                results = (_ConfiguredValue(rule.value),)
            else:
                inferred = [_arithmetic_result_types(_schema_type(field_type), rule) for field_type in types]
                if any(not variants for variants in inferred):
                    names = ", ".join(sorted({getattr(_schema_type(field_type), "__name__", str(field_type)) for field_type in types}))
                    raise ConfigError(
                        f"{key}.value",
                        f"Cannot apply '{rule.operation}' to '{'.'.join(rule.path)}' ({names}) with a value of type '{type(rule.value).__name__}'.",
                    )
                results = tuple(dict.fromkeys(variant for variants in inferred for variant in variants))
            # Keep the most recent assignment last, including when the same path is written twice.
            for written_path in tuple(types_by_path):
                if written_path[: len(path)] == path:
                    del types_by_path[written_path]
            types_by_path[path] = results
        if not found:
            if dynamic_origin is not None:
                if schemas:
                    logger.warning("%s.%s.path: only services without a declared message schema might provide this field.", dynamic_origin, key)
                continue
            raise ConfigError(f"{key}.path", f"Field path '{'.'.join(rule.path)}' is absent from the declared message types.")


def _schema_type(schema: Any) -> Any:
    return type(schema.value) if isinstance(schema, _ConfiguredValue) else schema


def _is_readonly_field(schema: Any, field_name: str) -> bool:
    schema = _schema_type(schema)
    kind = get_origin(schema) or schema
    if not isinstance(kind, type):
        return False
    if issubclass(kind, Mapping):
        return kind is MappingProxyType
    if field_name.isdigit() and issubclass(kind, Sequence):
        return not isabstract(kind) and not issubclass(kind, MutableSequence)
    if issubclass(kind, tuple) and field_name in getattr(kind, "_fields", ()):
        return True
    attribute = getattr_static(kind, field_name, None)
    if isinstance(attribute, property):
        return attribute.fset is None
    return kind in (bool, int, float, str, bytes, complex)


def _arithmetic_result_types(field_type: Any, rule: RewriteRule) -> tuple[Any, ...]:
    """Check Python's built-in operand types without using message data or invoking extension code."""
    left = get_origin(field_type) or field_type
    right = type(rule.value)
    if left in _BUILTIN_OPERAND_TYPES and right in _BUILTIN_OPERAND_TYPES:
        operation = operator.add if rule.operation == "add" else operator.mul
        try:
            return (type(operation(left(), right())),)
        except TypeError:
            return ()
    if isabstract(left) or isabstract(right):
        return (Any,)
    if rule.operation == "multiply":
        for sequence, multiplier in ((left, right), (right, left)):
            if sequence in (str, bytes, list, tuple) and hasattr(multiplier, "__index__"):
                return (sequence,)
    direct, reflected = ("__add__", "__radd__") if rule.operation == "add" else ("__mul__", "__rmul__")
    return _operator_types(left, direct, right) + _operator_types(right, reflected, left)


def _attribute_types(schema: type, name: str) -> tuple[Any, ...]:
    try:
        annotations = get_type_hints(schema)
    except (NameError, TypeError):
        annotations = {}
    if is_dataclass(schema):
        if name in {field.name for field in fields(schema)}:
            return _field_types(annotations.get(name, Any), ())
    elif name in annotations:
        return _field_types(annotations[name], ())
    member = getattr_static(schema, name, None)
    if isinstance(member, property) and member.fget is not None:
        try:
            return _field_types(get_type_hints(member.fget).get("return", Any), ())
        except (NameError, TypeError):
            return (Any,)
    if member is not None:
        return (Any,)
    if is_dataclass(schema) or schema is Attack or issubclass(schema, Sequence) or schema in _BUILTIN_OPERAND_TYPES:
        return ()
    return (Any,)


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
    if isinstance(schema, _ConfiguredValue):
        value = schema.value
        if isinstance(value, dict):
            return _field_types(_ConfiguredValue(value[field]), rest) if field in value else ()
        if isinstance(value, (list, tuple)) and field.isdigit():
            index = int(field)
            return _field_types(_ConfiguredValue(value[index]), rest) if index < len(value) else ()
        return _field_types(type(value), path)
    container = origin or schema
    if not isinstance(container, type):
        return (Any,)
    if issubclass(container, Mapping):
        if len(args) == 2 and _definitely_different_for_equality(args[0], str):
            return ()
        return _field_types(args[1], rest) if len(args) == 2 else (Any,)
    if field.isdigit() and issubclass(container, Sequence) and not issubclass(container, (str, bytes, bytearray)):
        if issubclass(container, tuple) and args and args[-1] is not Ellipsis:
            index = int(field)
            return _field_types(args[index], rest) if index < len(args) else ()
        if hasattr(container, "_fields"):
            names = container._fields
            index = int(field)
            if index >= len(names):
                return ()
            return tuple(leaf for item in _attribute_types(container, names[index]) for leaf in _field_types(item, rest))
        return _field_types(args[0], rest) if args else (Any,)
    return tuple(leaf for item in _attribute_types(container, field) for leaf in _field_types(item, rest))


def _snapshot_service_state_type(service_type: str) -> Any:
    try:
        service = BehaviorServiceRegistry.get_service_class(service_type)
        return get_type_hints(service.get_state).get("return", Any)
    except (KeyError, NameError, TypeError):
        return Any


def _validate_source(source: TriggerSourceSpec, services: Mapping[str, set[str]], path: str) -> tuple[Any, ...]:
    schema: Any = {"attack": Attack, "stage": StageRuntime, "snapshot": NodeSnapshot}[source.kind]
    if source.kind == "snapshot":
        if source.node_type is not None:
            _check_node_type(source.node_type, f"{path}.node_type")
        _check_service_reference(services, source.node_type, source.service_type, f"{path}.service_type")
        if source.service_type is not None:
            schema = _snapshot_service_state_type(source.service_type)
    elif source.field == "status":
        return (str,)  # The evaluator exposes the status enum's string value.

    schemas = _field_types(schema, ())
    if source.kind == "snapshot" and source.service_type is not None:
        # A missing state is skipped; None stored in a field is still a value.
        schemas = tuple(state for state in schemas if state is not type(None))
    if source.field is None:
        return schemas
    types: list[Any] = []
    for state in schemas:
        kind = get_origin(state) or state
        if kind is Any or kind is object or not isinstance(kind, type):
            types.append(Any)
        elif issubclass(kind, Mapping):
            args = get_args(state)
            if not args or not _definitely_different_for_equality(args[0], str):
                types.extend(_field_types(args[1] if args else Any, ()))
        else:
            types.extend(_attribute_types(kind, source.field))
    if not types and schemas:
        raise ConfigError(f"{path}.field", f"Field '{source.field}' is not declared in the selected source.")
    return tuple(types)


def _can_supply_node_ids(schema: Any) -> bool:
    kind = get_origin(schema) or schema
    if _can_be_node_id(schema):
        return True
    containers = (list, tuple, set, frozenset)
    if not any(issubclass(kind, container) or issubclass(container, kind) for container in containers):
        return False
    args = tuple(arg for arg in get_args(schema) if arg is not Ellipsis)
    return not args or any(_can_be_node_id(arg) for arg in args)


def _can_be_node_id(schema: Any) -> bool:
    kind = get_origin(schema) or schema
    if kind in (Union, UnionType):
        return any(_can_be_node_id(member) for member in get_args(schema))
    return kind is Any or kind is object or not isinstance(kind, type) or issubclass(kind, str)


def _definitely_different_for_equality(left: Any, right: type) -> bool:
    if get_origin(left) in (Union, UnionType):
        return all(_definitely_different_for_equality(member, right) for member in get_args(left))
    left = get_origin(left) or left
    if (left in _NUMBER_TYPES and right in _NUMBER_TYPES) or (left in (set, frozenset) and right in (set, frozenset)):
        return False
    return left in _BUILTIN_OPERAND_TYPES and right in _BUILTIN_OPERAND_TYPES and left is not right


def _operator_types(left: Any, method: str, right: Any) -> tuple[Any, ...]:
    """Infer the result of a custom operator from its annotations without calling it."""
    if left is Any or left is object or right is Any or right is object or not isinstance(left, type) or not isinstance(right, type):
        return (Any,)
    if left in _BUILTIN_OPERAND_TYPES:
        return ()
    implementation = getattr(left, method, None)
    if implementation is None or implementation is getattr(object, method, None):
        return ()
    try:
        hints = get_type_hints(implementation)
    except (NameError, TypeError):
        return (Any,)
    for name, expected in hints.items():
        if name in {"self", "return"} or expected is Any:
            continue
        choices = _field_types(expected, ())
        if choices and all(choice is not Any and isinstance(choice, type) and not issubclass(right, choice) for choice in choices):
            return ()
    return _field_types(hints.get("return", Any), ())


def _can_order(left: Any, right: Any, verb: str) -> bool:
    left, right = get_origin(left) or left, get_origin(right) or right
    if isabstract(left) or isabstract(right):
        return True  # A concrete implementation can define the comparison.
    if left in _BUILTIN_OPERAND_TYPES and right in _BUILTIN_OPERAND_TYPES:
        try:
            compare_value(left(), verb, right())
        except TypeError:
            return False
        return True
    direct, reflected = _ORDER_METHODS[verb]
    return bool(_operator_types(left, direct, right) or _operator_types(right, reflected, left))


def _can_order_with_value(schema: Any, value: Any, verb: str) -> bool:
    if get_origin(schema) in (Union, UnionType):
        return all(_can_order_with_value(member, value, verb) for member in get_args(schema))
    if not _can_order(schema, type(value), verb):
        return False
    args = get_args(schema)
    kind = get_origin(schema) or schema
    if args and kind in (Sequence, type(value)) and isinstance(value, (list, tuple)):
        if kind is Sequence:
            return all(_can_order_with_value(args[0], operand, verb) for operand in value)
        items = (args[0],) * len(value) if isinstance(value, list) or args[-1] is Ellipsis else args
        return all(_can_order_with_value(item, operand, verb) for item, operand in zip(items, value))
    return True


def _validate_ordered_comparison(types: tuple[Any, ...], source: TriggerSourceSpec, verb: str, value: Any, path: str) -> None:
    if value is None:
        raise ConfigError(path, f"Comparison '{verb}' requires a non-null value.")
    invalid = [annotation for annotation in types if not _can_order_with_value(annotation, value, verb)]
    if invalid:
        names = ", ".join(getattr(annotation, "__name__", str(annotation)) for annotation in invalid)
        raise ConfigError(path, f"Cannot apply '{verb}' to '{source.field or source.kind}' ({names}) with a value of type '{type(value).__name__}'.")


def _can_be_set_member(annotation: Any) -> bool:
    kind = get_origin(annotation) or annotation
    if kind is tuple or kind is frozenset or kind in (Union, UnionType):
        return all(_can_be_set_member(member) for member in get_args(annotation) if member is not Ellipsis)
    return getattr(kind, "__hash__", object.__hash__) is not None


def _validate_collection_transition(types: tuple[Any, ...], verb: str, value: Any, path: str, origin: str) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise ConfigError(f"{path}.value", f"'{verb}' requires a single hashable element, such as an ID string.") from exc

    element_types: list[Any] = []
    for annotation in types:
        container = get_origin(annotation) or annotation
        if not isinstance(container, type) or container is Any:
            element_types.append(Any)
            continue
        if not issubclass(container, Iterable) and not hasattr(container, "__getitem__"):
            raise ConfigError(f"{path}.source.field", f"'{verb}' requires a collection, but the field can contain {container.__name__}.")
        items = get_args(annotation)
        if issubclass(container, Mapping):
            items = items[:1]  # The evaluator compares dictionary keys.
        if any(not _can_be_set_member(item) for item in items if item is not Ellipsis):
            raise ConfigError(f"{path}.source.field", f"'{verb}' requires a collection with hashable elements.")
        element_types.extend(item for item in items if item is not Ellipsis)
        if not items:
            element_types.append(Any)
    if value is not None and element_types and all(_definitely_different_for_equality(item, type(value)) for item in element_types):
        logger.warning("%s.%s.value: the value cannot match an element of the declared collection.", origin, path)


def _validate_delta_transition(types: tuple[Any, ...], verb: str, value: Any, path: str) -> None:
    delta = 1 if value is None else value
    for current in types:
        for previous in types:
            left, right = get_origin(current) or current, get_origin(previous) or previous
            if not _can_order(current, previous, "gt" if verb == "increased" else "lt"):
                raise ConfigError(f"{path}.source.field", f"'{verb}' requires field values that can be ordered.")
            if verb == "decreased":
                left, right = right, left
            if left in _BUILTIN_OPERAND_TYPES and right in _BUILTIN_OPERAND_TYPES:
                try:
                    difference_types = (type(operator.sub(left(), right())),)
                except TypeError as exc:
                    raise ConfigError(f"{path}.source.field", f"'{verb}' requires field values that can be subtracted.") from exc
            else:
                difference_types = _operator_types(left, "__sub__", right) + _operator_types(right, "__rsub__", left)
                if not difference_types and (isabstract(left) or isabstract(right)):
                    continue
                if not difference_types:
                    raise ConfigError(f"{path}.source.field", f"'{verb}' requires field values that can be subtracted.")
            if any(not _can_order_with_value(schema, delta, "gte") for schema in difference_types):
                raise ConfigError(f"{path}.value", f"'{verb}' cannot compare the difference with a value of type '{type(delta).__name__}'.")


def _conditions_conflict(left: ConditionSpec, right: ConditionSpec) -> bool:
    if left.verb == right.verb == "eq":
        return left.value != right.value
    if left.verb in {"lt", "lte"} and right.verb in {"gt", "gte"}:
        left, right = right, left
    if left.verb not in {"gt", "gte"} or right.verb not in {"lt", "lte"}:
        return False
    if not isinstance(left.value, (int, float)) or not isinstance(right.value, (int, float)):
        return False
    return left.value > right.value or (left.value == right.value and (left.verb == "gt" or right.verb == "lt"))


def _validate_condition(
    condition: ConditionSpec,
    path: str,
    attack_name: str,
    stage_ids: set[str],
    origin: str,
    services: Mapping[str, set[str]],
    starting_stage: str | None = None,
) -> None:
    for index, child in enumerate(condition.all):
        source = child.source
        if source is None:
            continue
        single_value = (
            source.kind == "attack"
            or (source.kind == "stage" and source.stage_id is not None)
            or (source.kind == "snapshot" and source.node_id is not None)
        )
        if single_value and any(sibling.source == source and _conditions_conflict(sibling, child) for sibling in condition.all[:index]):
            logger.warning("%s.%s.all[%d].value: the conditions on the same source in this 'all' group cannot both hold.", origin, path, index)

    for group in ("all", "any"):
        for index, child in enumerate(getattr(condition, group)):
            _validate_condition(
                child, f"{path}.{group}[{index}]", attack_name, stage_ids, origin, services, starting_stage if group == "all" else None
            )

    source = condition.source
    if source is None:
        return
    if source.kind not in CONDITION_VERBS:
        raise ConfigError(f"{path}.source.kind", f"Unsupported condition source '{source.kind}'.")
    if condition.verb not in CONDITION_VERBS[source.kind]:
        raise ConfigError(f"{path}.verb", f"Unsupported verb '{condition.verb}' for source '{source.kind}'.")
    status_check = source.kind in {"attack", "stage"} and source.field == "status" and condition.verb in {"eq", "became"}
    if source.kind in {"attack", "stage"}:
        if source.attack_name is not None and source.attack_name != attack_name:
            raise ConfigError(f"{path}.source.attack_name", "Runtime conditions can only reference the current attack.")
        if source.kind == "stage" and source.stage_id is not None and source.stage_id not in stage_ids:
            raise ConfigError(f"{path}.source.stage_id", f"Unknown stage id '{source.stage_id}'.")
        if status_check:
            if condition.value not in tuple(status.value for status in RuntimeStatus if status != RuntimeStatus.STARTED):
                logger.warning("%s.%s.value: this status cannot be observed by the condition.", origin, path)
            elif (
                starting_stage is not None
                and source.kind == "stage"
                and source.stage_id == starting_stage
                and condition.value != RuntimeStatus.INACTIVE
            ):
                logger.warning("%s.%s.value: this stage is inactive while its start conditions are evaluated.", origin, path)

    types = _validate_source(source, services, f"{path}.source")
    if condition.verb in ORDERED_COMPARISONS:
        _validate_ordered_comparison(types, source, condition.verb, condition.value, f"{path}.value")
    elif condition.verb in _COLLECTION_TRANSITIONS:
        _validate_collection_transition(types, condition.verb, condition.value, path, origin)
    elif condition.verb in _DELTA_TRANSITIONS:
        _validate_delta_transition(types, condition.verb, condition.value, path)
    elif (
        condition.verb in {"eq", "became"}
        and not status_check
        and types
        and all(_definitely_different_for_equality(item, type(condition.value)) for item in types)
    ):
        logger.warning("%s.%s.value: the value cannot equal a value of the declared field type.", origin, path)

    if source.kind == "snapshot" and source.service_type == "aim_server" and source.field == "tracked_vehicle_count":
        value = condition.value
        if not isinstance(value, (int, float)):
            return
        if (condition.verb == "lt" and value <= 0) or (condition.verb in {"lte", "eq", "became"} and value < 0):
            logger.warning("%s.%s.value: the tracked vehicle count is always non-negative. This condition cannot be satisfied.", origin, path)
