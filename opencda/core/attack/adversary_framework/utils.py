"""Low-level helper utilities for the adversary framework."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection, Iterable
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import logging
from typing import Any, TypeAlias

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.core.application.behavior.capability import Capability
from opencda.scenario_testing.types import SimulationSnapshot

from .condition_evaluator import collect_snapshot_values, resolve_target_node_ids
from .models import TargetSpec

ServiceResolver: TypeAlias = Callable[[str, str | None], tuple[BehaviorService[Any, Any], ...]]
AttackResultRewriter: TypeAlias = Callable[[Any], Any]
RestoreCallback: TypeAlias = Callable[[], None]
_MISSING = object()

logger = logging.getLogger("cavise.opencda.opencda.core.attack.adversary_framework.utils")


# TODO: Consider using a more robust cloning strategy if needed, such as copyreg or custom __deepcopy__ implementations on CARLA types.
def safe_clone(value: Any) -> Any:
    """Clone common Python structures while tolerating opaque runtime objects such as CARLA types."""
    if value is None or isinstance(value, (bool, int, float, str, bytes, bytearray, complex)):
        return value

    if isinstance(value, tuple):
        return tuple(safe_clone(item) for item in value)
    if isinstance(value, list):
        return [safe_clone(item) for item in value]
    if isinstance(value, dict):
        return {safe_clone(key): safe_clone(item) for key, item in value.items()}
    if isinstance(value, deque):
        return deque((safe_clone(item) for item in value), maxlen=value.maxlen)
    if is_dataclass(value):
        field_values = {field.name: safe_clone(getattr(value, field.name)) for field in fields(value)}
        return replace(value, **field_values)

    try:
        return deepcopy(value)
    except Exception:
        return value


def wrap_method_output(
    original_method: Callable[..., Any],
    *,
    rewrite_result: AttackResultRewriter,
) -> Callable[..., Any]:
    """Return a callable that rewrites the original method output."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        original_result = original_method(*args, **kwargs)
        return rewrite_result(original_result)

    return wrapped


def get_capability_binding(
    service: BehaviorService[Any, Any],
    capability: Capability,
) -> Callable[..., Any]:
    """Return the bound callable exposed by a service for a capability."""
    try:
        return service.capability_bindings[capability]
    except KeyError as exc:
        raise RuntimeError(f"Service '{service.service_type}' does not expose capability '{capability.value}'.") from exc


def install_output_interceptor(
    service: BehaviorService[Any, Any],
    capability: Capability,
    *,
    rewrite_result: AttackResultRewriter,
) -> RestoreCallback:
    """Patch a concrete service instance so the capability method rewrites its output."""
    binding = get_capability_binding(service, capability)
    method_name = getattr(binding, "__name__", None)
    if not method_name:
        raise RuntimeError(f"Could not resolve method name for capability '{capability.value}' on service '{service.service_type}'.")

    had_instance_override = method_name in vars(service)
    previous_attr = vars(service).get(method_name, _MISSING)
    wrapped_method = wrap_method_output(binding, rewrite_result=rewrite_result)

    setattr(
        service,
        method_name,
        wrapped_method,
    )

    def restore() -> None:
        current_attr = vars(service).get(method_name, _MISSING)
        if current_attr is wrapped_method:
            if had_instance_override:
                setattr(service, method_name, previous_attr)
            else:
                delattr(service, method_name)

    return restore


def service_supports_capabilities(
    service: BehaviorService[Any, Any],
    required_capabilities: Collection[Capability],
) -> bool:
    """Return whether a behavior service provides all required capabilities."""
    provided_capabilities = set(service.capability_bindings)
    return set(required_capabilities).issubset(provided_capabilities)


def match_services(
    services: Iterable[BehaviorService[Any, Any]],
    required_capabilities: Collection[Capability],
) -> list[BehaviorService[Any, Any]]:
    """Return all services that satisfy the required capability set."""
    return [service for service in services if service_supports_capabilities(service, required_capabilities)]


def resolve_targets(
    target_spec: TargetSpec | None,
    current_snapshot: SimulationSnapshot,
    service_resolver: ServiceResolver,
    *,
    attack_name: str | None = None,
) -> tuple[BehaviorService[Any, Any], ...]:
    """Resolve live target services according to a target spec."""
    if target_spec is None:
        logger.warning(
            "Attack %r target resolution skipped because no target spec is configured.",
            attack_name,
        )
        return ()

    if target_spec.kind != "service_state_field":
        raise ValueError(f"Unsupported target resolution kind '{target_spec.kind}'.")
    if target_spec.selection not in {"all", "first"}:
        raise ValueError(f"Unsupported target selection '{target_spec.selection}'.")

    source_values = collect_snapshot_values(target_spec.source, current_snapshot)
    if not source_values:
        logger.warning(
            "Attack %r target resolution produced no source values: node_type=%r service_type=%r field=%r.",
            attack_name,
            target_spec.source.node_type,
            target_spec.source.service_type,
            target_spec.source.field,
        )

    target_node_ids = resolve_target_node_ids(target_spec.source, current_snapshot)
    if not target_node_ids:
        logger.warning(
            "Attack %r target resolution normalized to an empty node-id set from source values=%r.",
            attack_name,
            source_values,
        )

    ordered_target_node_ids = sorted(target_node_ids)
    if target_spec.selection == "first":
        ordered_target_node_ids = ordered_target_node_ids[:1]

    target_services: list[BehaviorService[Any, Any]] = []
    missing_node_ids: list[str] = []
    resolved_service_label = target_spec.resolve_to_service_name if target_spec.resolve_to_service_name is not None else "<any>"

    for node_id in ordered_target_node_ids:
        services = service_resolver(node_id, target_spec.resolve_to_service_name)
        if services:
            target_services.extend(services)
        else:
            missing_node_ids.append(node_id)

    available_node_ids = _collect_available_node_ids(current_snapshot, target_spec.resolve_to_node_type)
    if missing_node_ids:
        logger.warning(
            "Attack %r could not resolve service_type=%r for node_ids=%s. Available %s node_ids in snapshot: %s.",
            attack_name,
            resolved_service_label,
            missing_node_ids,
            target_spec.resolve_to_node_type,
            available_node_ids,
        )

    if target_services:
        logger.debug(
            "Attack %r resolved %d target service(s) of type=%r from node_ids=%s.",
            attack_name,
            len(target_services),
            resolved_service_label,
            ordered_target_node_ids,
        )
    else:
        logger.warning(
            "Attack %r resolved zero target services for service_type=%r from node_ids=%s.",
            attack_name,
            resolved_service_label,
            ordered_target_node_ids,
        )

    return tuple(target_services)


def _collect_available_node_ids(
    snapshot: SimulationSnapshot,
    node_type: str,
) -> tuple[str, ...]:
    if node_type == "vehicle":
        return tuple(node.node_id for node in snapshot.vehicle_nodes)
    if node_type == "rsu":
        return tuple(node.node_id for node in snapshot.rsu_nodes)
    return tuple(node.node_id for node in snapshot.vehicle_nodes + snapshot.rsu_nodes)
