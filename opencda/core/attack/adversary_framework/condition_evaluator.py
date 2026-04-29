"""Evaluation helpers for attack requirements and triggers."""

from __future__ import annotations

from typing import Any, Protocol

from opencda.scenario_testing.types import NodeSnapshot, SimulationSnapshot

from .models import ConditionSpec, RuntimeStatus, StageRuntime, TriggerSourceSpec

_MISSING = object()


class ConditionRuntime(Protocol):
    """Runtime data required to evaluate attack conditions."""

    attack_name: str
    status: RuntimeStatus
    previous_status: RuntimeStatus
    stage_runtimes: tuple[StageRuntime, ...]


def evaluate_condition(
    runtime: ConditionRuntime,
    condition: ConditionSpec,
    previous_snapshot: SimulationSnapshot | None,
    current_snapshot: SimulationSnapshot,
) -> bool:
    """Evaluate a condition tree against snapshot and runtime state."""
    if condition.all:
        return all(evaluate_condition(runtime, item, previous_snapshot, current_snapshot) for item in condition.all)

    if condition.any:
        return any(evaluate_condition(runtime, item, previous_snapshot, current_snapshot) for item in condition.any)

    source = condition.source
    verb = condition.verb
    if source is None or verb is None:
        raise ValueError("Leaf ConditionSpec must define both 'source' and 'verb'.")

    if source.kind == "snapshot":
        return _evaluate_snapshot_predicate(
            source=source,
            verb=verb,
            value=condition.value,
            previous_snapshot=previous_snapshot,
            current_snapshot=current_snapshot,
        )

    if source.kind in {"attack", "stage"}:
        return _evaluate_runtime_predicate(
            runtime=runtime,
            source=source,
            verb=verb,
            value=condition.value,
        )

    raise ValueError(
        f"Attack runtime currently supports only 'snapshot', 'attack' and 'stage' condition sources, got '{source.kind}'."
    )


def collect_snapshot_values(
    source: TriggerSourceSpec,
    snapshot: SimulationSnapshot | None,
) -> list[tuple[str, Any]]:
    """Collect values matching a snapshot source."""
    if snapshot is None:
        return []

    matches: list[tuple[str, Any]] = []
    for node in _iter_matching_nodes(source, snapshot):
        if source.service_name is None:
            value = _extract_field_value(node, source.field)
            if value is not _MISSING:
                matches.append((node.node_id, value))
            continue

        service_state = node.service_states.get(source.service_name)
        if service_state is None:
            continue

        if source.field is None:
            matches.append((node.node_id, service_state))
            continue

        value = _extract_field_value(service_state, source.field)
        if value is not _MISSING:
            matches.append((node.node_id, value))

    return matches


def resolve_target_node_ids(
    source: TriggerSourceSpec,
    snapshot: SimulationSnapshot,
) -> set[str]:
    """Resolve node ids from a snapshot field used as attack target source."""
    target_node_ids: set[str] = set()

    for _, value in collect_snapshot_values(source, snapshot):
        target_node_ids.update(normalize_target_node_ids(value))

    return target_node_ids


def normalize_target_node_ids(value: Any) -> set[str]:
    """Normalize a snapshot field into a set of node ids."""
    if isinstance(value, str):
        return {value}

    if isinstance(value, (tuple, list, set, frozenset)):
        return {item for item in value if isinstance(item, str)}

    return set()


def _evaluate_runtime_predicate(
    *,
    runtime: ConditionRuntime,
    source: TriggerSourceSpec,
    verb: str,
    value: Any,
) -> bool:
    current_values, previous_values = _collect_runtime_values(runtime, source)

    if verb == "exists":
        return bool(current_values)

    if verb in {"eq", "gt", "gte", "lt", "lte"}:
        return any(_compare_value(current_value, verb, value) for current_value in current_values.values())

    if verb == "changed":
        return any(previous_values.get(key, _MISSING) != current_value for key, current_value in current_values.items())

    if verb == "became":
        return any(
            current_value == value and previous_values.get(key, _MISSING) != value
            for key, current_value in current_values.items()
        )

    if verb == "increased":
        delta = 1 if value is None else value
        return any(
            key in previous_values
            and current_value > previous_values[key]
            and (current_value - previous_values[key]) >= delta
            for key, current_value in current_values.items()
        )

    if verb == "decreased":
        delta = 1 if value is None else value
        return any(
            key in previous_values
            and current_value < previous_values[key]
            and (previous_values[key] - current_value) >= delta
            for key, current_value in current_values.items()
        )

    raise ValueError(f"Unsupported runtime verb '{verb}'.")


def _evaluate_snapshot_predicate(
    *,
    source: TriggerSourceSpec,
    verb: str,
    value: Any,
    previous_snapshot: SimulationSnapshot | None,
    current_snapshot: SimulationSnapshot,
) -> bool:
    current_values = dict(collect_snapshot_values(source, current_snapshot))

    if verb == "exists":
        return bool(current_values)

    if verb in {"eq", "gt", "gte", "lt", "lte"}:
        return any(_compare_value(current_value, verb, value) for current_value in current_values.values())

    previous_values = dict(collect_snapshot_values(source, previous_snapshot))

    if verb == "changed":
        return any(previous_values.get(key, _MISSING) != current_value for key, current_value in current_values.items())

    if verb == "increased":
        delta = 1 if value is None else value
        return any(
            key in previous_values
            and current_value > previous_values[key]
            and (current_value - previous_values[key]) >= delta
            for key, current_value in current_values.items()
        )

    if verb == "decreased":
        delta = 1 if value is None else value
        return any(
            key in previous_values
            and current_value < previous_values[key]
            and (previous_values[key] - current_value) >= delta
            for key, current_value in current_values.items()
        )

    if verb == "became":
        return any(
            current_value == value and previous_values.get(key, _MISSING) != value
            for key, current_value in current_values.items()
        )

    if verb == "added":
        return any(_collection_added(previous_values.get(key, _MISSING), current_value, value) for key, current_value in current_values.items())

    if verb == "removed":
        return any(
            _collection_removed(previous_values.get(key, _MISSING), current_value, value)
            for key, current_value in current_values.items()
        )

    raise ValueError(f"Unsupported snapshot verb '{verb}'.")


def _collect_runtime_values(
    runtime: ConditionRuntime,
    source: TriggerSourceSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if source.kind == "attack":
        if source.attack_name is not None and source.attack_name != runtime.attack_name:
            return {}, {}

        current_value = _extract_runtime_value(
            obj=runtime,
            field_name=source.field,
            fallback_status=runtime.status,
        )
        previous_value = _extract_runtime_value(
            obj=runtime,
            field_name=source.field,
            fallback_status=runtime.previous_status,
        )

        if current_value is _MISSING:
            return {}, {}

        return {runtime.attack_name: current_value}, {runtime.attack_name: previous_value}

    if source.kind != "stage":
        raise ValueError(f"Unsupported runtime source kind '{source.kind}'.")

    if source.attack_name is not None and source.attack_name != runtime.attack_name:
        return {}, {}

    current_values: dict[str, Any] = {}
    previous_values: dict[str, Any] = {}

    for stage_runtime in runtime.stage_runtimes:
        if source.stage_id is not None and stage_runtime.spec.id != source.stage_id:
            continue

        current_value = _extract_runtime_value(
            obj=stage_runtime,
            field_name=source.field,
            fallback_status=stage_runtime.status,
        )
        if current_value is _MISSING:
            continue

        current_values[stage_runtime.spec.id] = current_value
        previous_values[stage_runtime.spec.id] = _extract_runtime_value(
            obj=stage_runtime,
            field_name=source.field,
            fallback_status=stage_runtime.previous_status,
        )

    return current_values, previous_values


def _iter_matching_nodes(
    source: TriggerSourceSpec,
    snapshot: SimulationSnapshot,
) -> tuple[NodeSnapshot, ...]:
    if source.node_type == "vehicle":
        nodes = snapshot.vehicle_nodes
    elif source.node_type == "rsu":
        nodes = snapshot.rsu_nodes
    else:
        nodes = snapshot.vehicle_nodes + snapshot.rsu_nodes

    matched_nodes: list[NodeSnapshot] = []
    for node in nodes:
        if source.node_id is not None and node.node_id != source.node_id:
            continue
        matched_nodes.append(node)

    return tuple(matched_nodes)


def _extract_field_value(obj: Any, field_name: str | None) -> Any:
    if field_name is None:
        return obj
    if isinstance(obj, dict):
        return obj.get(field_name, _MISSING)
    return getattr(obj, field_name, _MISSING)


def _extract_runtime_value(
    *,
    obj: Any,
    field_name: str | None,
    fallback_status: RuntimeStatus,
) -> Any:
    if field_name == "status":
        return RuntimeStatus.ACTIVE.value if fallback_status == RuntimeStatus.STARTED else fallback_status.value
    return _extract_field_value(obj, field_name)


def _compare_value(current_value: Any, verb: str, expected_value: Any) -> bool:
    if verb == "eq":
        return current_value == expected_value
    if verb == "gt":
        return current_value > expected_value
    if verb == "gte":
        return current_value >= expected_value
    if verb == "lt":
        return current_value < expected_value
    if verb == "lte":
        return current_value <= expected_value
    raise ValueError(f"Unsupported comparison verb '{verb}'.")


def _collection_added(previous_value: Any, current_value: Any, expected_value: Any) -> bool:
    previous_items = set(previous_value) if previous_value is not _MISSING else set()
    current_items = set(current_value)
    added_items = current_items - previous_items
    if expected_value is None:
        return bool(added_items)
    return expected_value in added_items


def _collection_removed(previous_value: Any, current_value: Any, expected_value: Any) -> bool:
    previous_items = set(previous_value) if previous_value is not _MISSING else set()
    current_items = set(current_value)
    removed_items = previous_items - current_items
    if expected_value is None:
        return bool(removed_items)
    return expected_value in removed_items
