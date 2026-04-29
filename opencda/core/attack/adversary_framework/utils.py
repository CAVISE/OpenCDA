"""Low-level helper utilities for the adversary framework."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from typing import Any, TypeAlias

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.core.application.behavior.capability import Capability
from opencda.scenario_testing.types import SimulationSnapshot

from .condition_evaluator import resolve_target_node_ids
from .models import TargetSpec

ServiceResolver: TypeAlias = Callable[[str, str], BehaviorService[Any, Any] | None]
AttackResultRewriter: TypeAlias = Callable[[Any], Any]
RestoreCallback: TypeAlias = Callable[[], None]
_MISSING = object()


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
        raise RuntimeError(f"Service '{service.service_name}' does not expose capability '{capability.value}'.") from exc


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
        raise RuntimeError(f"Could not resolve method name for capability '{capability.value}' on service '{service.service_name}'.")

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
) -> tuple[BehaviorService[Any, Any], ...]:
    """Resolve live target services according to a target spec."""
    if target_spec is None:
        return ()

    if target_spec.kind != "service_state_field":
        raise ValueError(f"Unsupported target resolution kind '{target_spec.kind}'.")

    target_node_ids = resolve_target_node_ids(target_spec.source, current_snapshot)
    target_services: list[BehaviorService[Any, Any]] = []

    for node_id in sorted(target_node_ids):
        service = service_resolver(node_id, target_spec.resolve_to_service_name)
        if service is not None:
            target_services.append(service)

    return tuple(target_services)
