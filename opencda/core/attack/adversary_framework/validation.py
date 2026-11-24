"""Checks on declarations available before a simulation is created."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from opencda.core.application.behavior.capability import CapabilityPayloads
from opencda.core.application.behavior.registry import BehaviorServiceRegistry

from .attack import Attack
from .condition_evaluator import CONDITION_VERBS
from .models import AttackSpec, ConditionSpec, RuntimeStatus
from .models.config_error import ConfigError
from .stages.spoofer.stage import SpooferStage
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


def validate_spec(spec: AttackSpec, services: Mapping[str, set[str]], origin: str) -> None:
    targets = spec.targets
    if targets is not None:
        if targets.kind != TARGET_KIND:
            raise ConfigError("targets.kind", f"Unsupported target resolution kind '{targets.kind}'.")
        if targets.selection not in TARGET_SELECTIONS:
            raise ConfigError("targets.selection", f"Unsupported target selection '{targets.selection}'.")
        _check_service_reference(services, targets.resolve_to_node_type, targets.resolve_to_service_name, "targets.resolve_to.service_type")
        if targets.source.kind == "snapshot":
            _check_service_reference(services, targets.source.node_type, targets.source.service_type, "targets.source.service_type")

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
            payloads = {contract[capability] for contract in compatible for capability in required}
            for rule_index, rule in enumerate(stage.rewrite_rules):
                if rule.path[0] == "payload" and not any(_allows_field_path(payload, rule.path[1:]) for payload in payloads if payload is not None):
                    raise ConfigError(
                        f"{location}.params.rewrites[{rule_index}].path",
                        f"Field path '{'.'.join(rule.path)}' is absent from the declared message types.",
                    )


def _allows_field_path(schema: Any, path: tuple[str, ...]) -> bool:
    """Reject paths contradicted by a declared type; dynamic values are checked when used."""
    if not path or schema is Any or schema is object:
        return True
    origin = get_origin(schema)
    args = get_args(schema)
    if origin in (Union, UnionType):
        return any(_allows_field_path(member, path) for member in args if member is not type(None))

    field, rest = path[0], path[1:]
    if is_dataclass(schema):
        if field not in {item.name for item in fields(schema)}:
            return False
        try:
            annotations = get_type_hints(schema)
        except (NameError, TypeError):
            return True
        return _allows_field_path(annotations[field], rest)

    container = origin or schema
    if isinstance(container, type) and issubclass(container, Mapping):
        return _allows_field_path(args[1], rest) if len(args) == 2 else True
    if isinstance(container, type) and issubclass(container, (str, bytes, int, float, complex, type(None))):
        return False
    if isinstance(container, type) and issubclass(container, Sequence):
        if not field.isdigit():
            return False
        if container is tuple and args and args[-1] is not Ellipsis:
            index = int(field)
            return index < len(args) and _allows_field_path(args[index], rest)
        return _allows_field_path(args[0], rest) if args else True
    return True


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
        _check_service_reference(services, source.node_type, source.service_type, f"{path}.source.service_type")
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
