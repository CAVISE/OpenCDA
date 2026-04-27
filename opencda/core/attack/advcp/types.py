from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence, TypeAlias, TypedDict
import numpy.typing as npt
import torch

from opencda.core.common.coperception_data_processor import LiveMemorySnapshot


AdvCPMemoryRecord: TypeAlias = OrderedDict[str, LiveMemorySnapshot | bool]
AdvCPScenarioData: TypeAlias = OrderedDict[str, AdvCPMemoryRecord]
AdvCPMemoryData: TypeAlias = OrderedDict[int, AdvCPScenarioData]


class AdvCPVisualizationContext(TypedDict):
    attacker_ids: list[str]
    fake_box_tensor: torch.Tensor | None  # noqa: DC01
    mode: str | None


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
    density: int | str
    dense_distance: float
    sync: bool  # noqa: DC01
    init: bool
    online: bool
    step: int
    random_seed: int
    max_perturb: float
    lr: float  # noqa: DC01
    feature_size: int
    car_mesh_path: str
    car_mesh_divide_path: str
    model_path: str  # noqa: DC01
    mesh_divide_path: str  # noqa: DC01


class AdvCPIntermediateAttackState(TypedDict, total=False):
    previous_memory_data: AdvCPMemoryData | None
    current_memory_data: AdvCPMemoryData | None
    init_perturbation: list[npt.NDArray] | None


AdvCPAttackResult: TypeAlias = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    AdvCPVisualizationContext,
]
