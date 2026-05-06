from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


ServiceStateMap: TypeAlias = dict[str, Any]


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    node_id: str
    node_type: str
    service_states: ServiceStateMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    tick: int
    vehicle_nodes: tuple[NodeSnapshot, ...] = field(default_factory=tuple)
    rsu_nodes: tuple[NodeSnapshot, ...] = field(default_factory=tuple)
