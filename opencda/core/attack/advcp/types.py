from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeAlias, TypedDict

import numpy as np
import torch


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


AdvCPAttackResult: TypeAlias = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    AdvCPVisualizationContext,
]


class AdvCPIntermediateAttackState(TypedDict, total=False):
    previous_memory_data: dict[Any, Any] | None
    init_perturbation: list[np.ndarray] | None
