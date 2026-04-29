"""Structured configuration primitives for attacks and stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TriggerSourceSpec:
    """Source descriptor for trigger and requirement evaluation."""

    kind: str
    node_type: str | None = None
    node_id: str | None = None
    service_name: str | None = None
    field: str | None = None
    attack_name: str | None = None
    stage_id: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TriggerSourceSpec:
        return cls(
            kind=str(data["kind"]),
            node_type=data.get("node_type"),
            node_id=data.get("node_id"),
            service_name=data.get("service_name"),
            field=data.get("field"),
            attack_name=data.get("attack_name"),
            stage_id=data.get("stage_id"),
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
        if "all" in data:
            return cls(all=tuple(cls.from_dict(item) for item in data["all"]))

        if "any" in data:
            return cls(any=tuple(cls.from_dict(item) for item in data["any"]))

        return cls(
            source=TriggerSourceSpec.from_dict(data["source"]),
            verb=str(data["verb"]),
            value=data.get("value"),
        )


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Target resolution specification for an attack."""

    kind: str
    source: TriggerSourceSpec
    resolve_to_node_type: str
    resolve_to_service_name: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TargetSpec:
        resolve_to = data["resolve_to"]
        return cls(
            kind=str(data["kind"]),
            source=TriggerSourceSpec.from_dict(data["source"]),
            resolve_to_node_type=str(resolve_to["node_type"]),
            resolve_to_service_name=str(resolve_to["service_name"]),
        )


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Structured description of a single attack stage."""

    id: str
    type: str
    requirements: ConditionSpec | None = None
    stage_start_trigger: ConditionSpec | None = None
    stage_stop_trigger: ConditionSpec | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageSpec:
        requirements = data.get("requirements")
        stage_start_trigger = data.get("stage_start_trigger")
        stage_stop_trigger = data.get("stage_stop_trigger")

        return cls(
            id=str(data["id"]),
            type=str(data["type"]),
            requirements=ConditionSpec.from_dict(requirements) if requirements is not None else None,
            stage_start_trigger=ConditionSpec.from_dict(stage_start_trigger) if stage_start_trigger is not None else None,
            stage_stop_trigger=ConditionSpec.from_dict(stage_stop_trigger) if stage_stop_trigger is not None else None,
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
        requirements = data.get("requirements")
        start_trigger = data.get("start_trigger")
        stop_trigger = data.get("stop_trigger")
        targets = data.get("targets")

        return cls(
            name=str(data["name"]),
            requirements=ConditionSpec.from_dict(requirements) if requirements is not None else None,
            start_trigger=ConditionSpec.from_dict(start_trigger) if start_trigger is not None else None,
            stop_trigger=ConditionSpec.from_dict(stop_trigger) if stop_trigger is not None else None,
            targets=TargetSpec.from_dict(targets) if targets is not None else None,
            stages=tuple(StageSpec.from_dict(stage) for stage in data.get("stages", ())),
        )
