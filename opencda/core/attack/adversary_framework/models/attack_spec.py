"""Structured configuration primitives for attacks and stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TypeVar

from opencda.core.application.behavior.capability import Capability

from .config_error import ConfigError, at_config_path

T = TypeVar("T")


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Expected a mapping.")
    return value


def _required(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ConfigError(key, "Required field is missing.")
    return data[key]


def _string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a string.")
    return value


def _check_keys(data: Mapping[str, Any], allowed: set[str]) -> None:
    for key in data:
        if key not in allowed:
            raise ConfigError(str(key), "Unknown field.")


def _parse_items(items: Any, parse: Callable[[Any], T]) -> tuple[T, ...]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("Expected a list.")
    result = []
    for index, item in enumerate(items):
        with at_config_path(f"[{index}]"):
            result.append(parse(item))
    return tuple(result)


def _optional(data: Mapping[str, Any], key: str, parse: Callable[[Any], T]) -> T | None:
    value = data.get(key)
    if value is None:
        return None
    with at_config_path(key):
        return parse(value)


@dataclass(frozen=True, slots=True)
class TriggerSourceSpec:
    """Source descriptor for trigger and requirement evaluation."""

    kind: str
    node_type: str | None = None
    node_id: str | None = None
    service_type: str | None = None
    field: str | None = None
    attack_name: str | None = None
    stage_id: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TriggerSourceSpec:
        data = _mapping(data)
        kind = str(_required(data, "kind"))
        keys = {"kind", "field"}
        if kind == "snapshot":
            keys.update(("node_type", "node_id", "service_type"))
        elif kind in {"attack", "stage"}:
            keys.add("attack_name")
            if kind == "stage":
                keys.add("stage_id")
        if kind in {"snapshot", "attack", "stage"}:
            _check_keys(data, keys)
        return cls(
            kind=kind,
            node_type=_optional(data, "node_type", _string),
            node_id=_optional(data, "node_id", _string),
            service_type=_optional(data, "service_type", _string),
            field=_optional(data, "field", _string),
            attack_name=_optional(data, "attack_name", _string),
            stage_id=_optional(data, "stage_id", _string),
        )


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """Recursive condition tree for triggers and requirements."""

    source: TriggerSourceSpec | None = None
    verb: str | None = None
    value: Any = None
    all: tuple[ConditionSpec, ...] = ()
    any: tuple[ConditionSpec, ...] = ()

    def __post_init__(self) -> None:
        is_leaf = self.source is not None or self.verb is not None
        has_groups = bool(self.all) or bool(self.any)

        if is_leaf and has_groups:
            raise ValueError("ConditionSpec cannot define both a leaf predicate and nested condition groups.")

        if bool(self.all) and bool(self.any):
            raise ValueError("ConditionSpec cannot define both 'all' and 'any' groups at the same level.")

        if is_leaf and self.source is None:
            raise ValueError("Leaf ConditionSpec must define 'source'.")

        if is_leaf and self.verb is None:
            raise ValueError("Leaf ConditionSpec must define 'verb'.")

        if not is_leaf and not has_groups:
            raise ValueError("ConditionSpec must define either a leaf predicate or a nested condition group.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConditionSpec:
        data = _mapping(data)
        if "all" in data and "any" in data:
            raise ConfigError("any", "Cannot be combined with 'all' in the same condition.")
        if "all" in data or "any" in data:
            group = "all" if "all" in data else "any"
            for key in ("source", "verb"):
                if key in data:
                    raise ConfigError(key, f"Cannot be combined with '{group}' in the same condition.")
            _check_keys(data, {group})
            with at_config_path(group):
                children = _parse_items(data[group], cls.from_dict)
                return cls(all=children) if group == "all" else cls(any=children)

        _check_keys(data, {"source", "verb", "value"})
        source_data = _required(data, "source")
        with at_config_path("source"):
            source = TriggerSourceSpec.from_dict(source_data)
        return cls(source=source, verb=str(_required(data, "verb")), value=data.get("value"))


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Target resolution specification for an attack."""

    kind: str
    source: TriggerSourceSpec
    resolve_to_node_type: str  # noqa: DC01
    resolve_to_service_name: str | None
    selection: str = "all"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TargetSpec:
        data = _mapping(data)
        _check_keys(data, {"kind", "source", "resolve_to", "selection"})
        source_data = _required(data, "source")
        with at_config_path("source"):
            source = TriggerSourceSpec.from_dict(source_data)
        resolve_data = _required(data, "resolve_to")
        with at_config_path("resolve_to"):
            resolve_to = _mapping(resolve_data)
            _check_keys(resolve_to, {"node_type", "service_type"})
            node_type = str(_required(resolve_to, "node_type"))
            service_type = _optional(resolve_to, "service_type", _string)
        return cls(
            kind=str(_required(data, "kind")),
            source=source,
            resolve_to_node_type=node_type,
            resolve_to_service_name=service_type,
            selection=str(data.get("selection", "all")),
        )


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Structured description of a single attack stage."""

    id: str
    type: str
    capabilities: tuple[Capability, ...] | None = None
    params: Mapping[str, Any] | None = None
    requirements: ConditionSpec | None = None
    stage_start_trigger: ConditionSpec | None = None
    stage_stop_trigger: ConditionSpec | None = None

    def __post_init__(self) -> None:
        if self.capabilities is not None and not self.capabilities:
            raise ValueError("StageSpec must define at least one capability.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageSpec:
        data = _mapping(data)
        _check_keys(data, {"id", "type", "capabilities", "params", "requirements", "stage_start_trigger", "stage_stop_trigger"})
        configured_capabilities = data.get("capabilities")
        capabilities: tuple[Capability, ...] | None = None
        if configured_capabilities is not None:
            with at_config_path("capabilities"):
                if isinstance(configured_capabilities, str):
                    capabilities = (Capability(configured_capabilities),)
                else:
                    capabilities = _parse_items(configured_capabilities, Capability)
                if not capabilities:
                    raise ValueError("Expected at least one capability.")
        return cls(
            id=str(_required(data, "id")),
            type=str(_required(data, "type")),
            capabilities=capabilities,
            params=_optional(data, "params", lambda value: dict(_mapping(value))),
            requirements=_optional(data, "requirements", ConditionSpec.from_dict),
            stage_start_trigger=_optional(data, "stage_start_trigger", ConditionSpec.from_dict),
            stage_stop_trigger=_optional(data, "stage_stop_trigger", ConditionSpec.from_dict),
        )


@dataclass(frozen=True, slots=True)
class AttackSpec:
    """Structured description of an attack."""

    name: str
    requirements: ConditionSpec | None = None
    start_trigger: ConditionSpec | None = None
    stop_trigger: ConditionSpec | None = None
    targets: TargetSpec | None = None
    stages: tuple[StageSpec, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttackSpec:
        data = _mapping(data)
        _check_keys(data, {"name", "requirements", "start_trigger", "stop_trigger", "targets", "stages"})
        with at_config_path("stages"):
            stages = _parse_items(data.get("stages", ()), StageSpec.from_dict)
        return cls(
            name=str(_required(data, "name")),
            requirements=_optional(data, "requirements", ConditionSpec.from_dict),
            start_trigger=_optional(data, "start_trigger", ConditionSpec.from_dict),
            stop_trigger=_optional(data, "stop_trigger", ConditionSpec.from_dict),
            targets=_optional(data, "targets", TargetSpec.from_dict),
            stages=stages,
        )
