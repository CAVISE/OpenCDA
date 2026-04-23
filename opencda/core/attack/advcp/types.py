from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence, TypeAlias, TypedDict

import numpy as np
import torch

from opencda.core.common.coperception_data_processor import LiveMemorySnapshot


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
    relative: Sequence[float]
    absolute: Sequence[float]
    size: Sequence[float]


class AdvCPConfig(TypedDict, total=False):
    mode: str
    default_size: Sequence[float]
    boxes: list[AdvCPBoxSpec]
    attacker_id: str | None
    density: int | str
    dense_distance: float
    sync: bool
    init: bool
    online: bool
    step: int
    max_perturb: float
    lr: float
    feature_size: int
    car_mesh_path: str
    car_mesh_divide_path: str
    model_path: str
    mesh_divide_path: str


class AdvCPIntermediateAttackState(TypedDict, total=False):
    previous_memory_data: OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]] | None
    init_perturbation: list[np.ndarray] | None


AdvCPAttackResult: TypeAlias = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    AdvCPVisualizationContext,
]
