from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypeAlias, TypedDict
import numpy.typing as npt
import torch

from opencda.core.common.coperception_data_processor import LiveMemorySnapshot


AdvCPMemoryRecord: TypeAlias = OrderedDict[str, LiveMemorySnapshot | bool]
AdvCPScenarioData: TypeAlias = OrderedDict[str, AdvCPMemoryRecord]
AdvCPMemoryData: TypeAlias = OrderedDict[int, AdvCPScenarioData]


@dataclass
class AdvCPVisualizationContext:
    mode: str | None = None
    attacker_ids: list[str] = field(default_factory=list)
    fake_box_tensor: torch.Tensor | None = None  # noqa: DC01
    removed_box_tensor: torch.Tensor | None = None


class AdvCPAgentState(TypedDict):
    agent_id: str
    timestamp: str
    yaml_path: str | None
    params: Mapping[str, Any]
    lidar_pose: Sequence[float]
    ego_pose: Sequence[float]


class AdvCPBoxSpec(TypedDict, total=False):
    relative: Sequence[float]  # noqa: DC01
    absolute: Sequence[float]  # noqa: DC01
    size: Sequence[float]


class AdvCPConfig(TypedDict, total=False):
    mode: str
    default_size: Sequence[float]  # noqa: DC01
    boxes: list[AdvCPBoxSpec]
    attacker_ids: list[str]
    advshape: bool | str  # noqa: DC01
    density: str
    dense_distance: float
    sync: bool  # noqa: DC01
    init: bool
    online: bool
    step: int
    max_perturb: float
    lr: float  # noqa: DC01
    feature_size: int
    car_mesh_path: str
    car_mesh_divide_path: str
    remove_adv_shape_perturb_path: str  # noqa: DC01
    remove_adv_shape_divide_path: str  # noqa: DC01
    model_path: str  # noqa: DC01
    mesh_divide_path: str  # noqa: DC01


class AdvCPIntermediateAttackState(TypedDict, total=False):
    previous_memory_data: AdvCPMemoryData | None
    current_memory_data: AdvCPMemoryData | None
    init_perturbation: dict[str, list[npt.NDArray]] | None


AdvCPAttackResult: TypeAlias = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    AdvCPVisualizationContext,
]
