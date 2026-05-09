"""
Typed dictionaries, dataclasses, and type aliases shared by every AdvCP
attack module.

The aliases defined here are meant to make type signatures elsewhere in
the package self-documenting. They use ``TypeAlias`` (not ``NewType``):
the goal is readability, not strict subtype enforcement, so passing a
plain ``str`` where a semantic alias is expected does not require
explicit wrapping.

See the package README for how these types fit together end-to-end.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypeAlias, TypedDict

import numpy.typing as npt
import torch

from opencda.core.common.coperception_data_processor import LiveMemorySnapshot


# Identifiers ---------------------------------------------------------------

# Stable string identifier for any agent in the scenario (a CAV or the
# ego vehicle). Matches the agent id used by OpenCOOD and by the rest of
# the OpenCDA pipeline.
AgentId: TypeAlias = str

# Subset of ``AgentId`` denoting an agent that AdvCP is configured to
# treat as an attacker. Listed under ``attacker_ids`` in the AdvCP YAML.
AttackerId: TypeAlias = str

# Attacker id as it appears inside the OpenCOOD-collated batch. Equal to
# the ``AttackerId`` for non-ego attackers, or to the literal ``"ego"``
# when the attacker is the ego vehicle (because OpenCOOD keys the ego
# entry under ``"ego"`` regardless of its true id).
BatchAttackerId: TypeAlias = str

# OpenCOOD timestamp string (zero-padded integer-like string keyed inside
# scenario memory snapshots).
Timestamp: TypeAlias = str


# Box representations -------------------------------------------------------

# Single 3D bounding box in a sensor frame, encoded as a length-7 array:
# ``[x, y, z, length, width, height, yaw]``. ``(x, y, z)`` is the bottom
# centre of the box, ``yaw`` is in radians measured around the ``z``
# axis. This is the representation produced by
# ``AdvCPAttackHelper.world_box_to_sensor_box`` and consumed by every
# attack module.
BoxLwhBottomCenter: TypeAlias = npt.NDArray

# 3D bounding box in the model-friendly representation (centre-up
# convention, axis order matching ``dataset.post_processor.params["order"]``).
# Produced by ``AdvCPAttackHelper.convert_box_for_model``.
BoxModelTensor: TypeAlias = torch.Tensor


# Memory snapshots ----------------------------------------------------------

# Per-agent snapshot at a single tick. The dictionary holds either the
# OpenCOOD ``LiveMemorySnapshot`` (raw lidar, params, yaml path) keyed
# by timestamp, or the boolean ``ego`` flag keyed under ``"ego"``.
AdvCPMemoryRecord: TypeAlias = OrderedDict[str, LiveMemorySnapshot | bool]

# Scenario-wide snapshot at a single tick: a mapping from each agent id
# to its per-agent record. Has at most one ego flag per scenario.
AdvCPScenarioData: TypeAlias = OrderedDict[AgentId, AdvCPMemoryRecord]

# The full memory data structure passed into AdvCP. Keyed by batch
# index (an ``int``); typically holds a single batch element. AdvCP
# reads the first (and only) value via ``next(iter(memory_data.values()))``.
AdvCPMemoryData: TypeAlias = OrderedDict[int, AdvCPScenarioData]


@dataclass
class AdvCPVisualizationContext:
    """
    State carried alongside an attack's prediction tensors.

    Consumed by the visualizer (to colour fake / removed boxes
    differently from regular predictions and ground truth) and by the
    metrics framework (to compute attack success rate and target
    confidence).

    Attributes
    ----------
    mode : {"spoofing", "removal", None}
        The active AdvCP mode. ``None`` means the attack was not run on
        this tick.
    attacker_ids : list of AttackerId
        Attackers that were actually applied on this tick. Differs
        from the configured ``attacker_ids`` whenever an attacker is
        missing from the current batch (e.g. out of communication
        range).
    fake_box_tensor : torch.Tensor or None
        Spoofing target boxes for the current tick, in corner
        representation ``(N, 8, 3)``. Set only when ``mode ==
        "spoofing"`` and the attack succeeded.
    removed_box_tensor : torch.Tensor or None
        Removal target boxes for the current tick, in corner
        representation ``(N, 8, 3)``. Set only when ``mode ==
        "removal"`` and the attack succeeded.
    """

    mode: str | None = None
    attacker_ids: list[AttackerId] = field(default_factory=list)
    fake_box_tensor: torch.Tensor | None = None  # noqa: DC01
    removed_box_tensor: torch.Tensor | None = None


class AdvCPAgentState(TypedDict):
    """
    Resolved per-agent state assembled from the scenario memory snapshot.

    Returned by ``AdvCPAttackHelper.load_agent_state`` and passed to box
    resolution helpers. The ``params`` mapping is the loaded scenario
    YAML for the agent at the active timestamp.

    Attributes
    ----------
    agent_id : AgentId
        Stable agent identifier.
    timestamp : Timestamp
        OpenCOOD timestamp string for the active tick.
    yaml_path : str or None
        Path to the source YAML, when available.
    params : Mapping
        Loaded YAML contents for this agent at this timestamp.
    lidar_pose : Sequence of float
        World-frame pose ``[x, y, z, roll, yaw, pitch]`` of the lidar
        sensor.
    ego_pose : Sequence of float
        World-frame pose of the vehicle. Defaults to ``lidar_pose`` when
        the YAML does not contain a ``true_ego_pos`` field.
    """

    agent_id: AgentId
    timestamp: Timestamp
    yaml_path: str | None
    params: Mapping[str, Any]
    lidar_pose: Sequence[float]
    ego_pose: Sequence[float]


class AdvCPBoxSpec(TypedDict, total=False):
    """
    User-facing target box specification, as it appears in the AdvCP
    YAML under ``boxes``.

    Exactly one of ``relative`` or ``absolute`` must be provided. The
    optional ``size`` overrides the global ``default_size``.

    Attributes
    ----------
    relative : Sequence of float
        Six-vector ``[x, y, z, roll, yaw, pitch]`` interpreted relative
        to the ego pose.
    absolute : Sequence of float
        Six-vector ``[x, y, z, roll, yaw, pitch]`` in world coordinates.
    size : Sequence of float
        ``[length, width, height]`` of the box.
    """

    relative: Sequence[float]  # noqa: DC01
    absolute: Sequence[float]  # noqa: DC01
    size: Sequence[float]


class AdvCPConfig(TypedDict, total=False):
    """
    Resolved AdvCP configuration.

    Mirrors the AdvCP YAML schema after defaults have been applied by
    ``AdvCoperceptionModelManager.load_config``. ``total=False`` because
    fusion-specific keys (``advshape``, ``density``,
    ``remove_adv_shape_*``, ``car_mesh_*``) are only required by the
    fusion mode that uses them.

    Notable keys
    ------------
    mode : {"spoofing", "removal"}
        Attack mode.
    attacker_ids : list of AttackerId
        Configured attackers. Resolved against the current batch on
        each tick; missing attackers are reported and skipped.
    boxes : list of AdvCPBoxSpec
        Target boxes.
    default_size : Sequence of float
        Box size used when an entry under ``boxes`` does not override it.
    advshape : bool
        Early-fusion removal: use the adversarial-shape mesh instead of
        a plain wall mesh.
    density : str
        Early-fusion: synthetic-point density. One of ``"replace"``,
        ``"dense_a"``, ``"dense_all"``, ``"sampled"``.
    dense_distance : float
        Early-fusion: distance offset used when injecting auxiliary
        rays from neighbouring CAV poses.
    sync : bool
        Intermediate fusion: optimize against the previous tick's batch
        when both ticks contain the configured attackers (preserves
        gradient quality when the model is sensitive to feature
        coordinates).
    init : bool
        Intermediate fusion: warm-start the perturbation by ray-tracing
        a synthetic mesh into the attacker's lidar before extracting
        the base perturbation tensor.
    online : bool
        Intermediate fusion: persist the best perturbation across
        ticks so the next call starts close to the previous solution.
    step : int
        Intermediate fusion: number of Adam steps per tick.
    max_perturb : float
        Intermediate fusion: L-infinity bound on perturbation values.
    lr : float
        Intermediate fusion: Adam learning rate.
    feature_size : int
        Intermediate fusion: half-side of the perturbation patch in
        feature pixels.
    car_mesh_path, car_mesh_divide_path : str
        Filesystem paths for the bundled car mesh assets used by early
        fusion. Resolved relative to the AdvCP YAML directory.
    remove_adv_shape_perturb_path, remove_adv_shape_divide_path : str
        Optional filesystem paths for the bundled adversarial-shape
        mesh assets used by early-fusion removal when ``advshape`` is
        ``True``.
    model_path, mesh_divide_path : str
        Backwards-compatible aliases honoured by ``load_config``.
    """

    mode: str
    default_size: Sequence[float]  # noqa: DC01
    boxes: list[AdvCPBoxSpec]
    attacker_ids: list[AttackerId]
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
    """
    Per-manager state preserved across intermediate-fusion ticks.

    The intermediate-fusion attack uses two reusable buffers
    (``previous_memory_data`` / ``current_memory_data``) to keep one
    extra tick of history without reallocating, and stores the best
    perturbation as a numpy initial value for the next tick when
    ``online`` is enabled.

    Attributes
    ----------
    previous_memory_data : AdvCPMemoryData or None
        Snapshot of the memory data passed in at the previous tick.
        Used for synchronous optimization (``sync: true``) when the
        configured attackers were present at both ticks.
    current_memory_data : AdvCPMemoryData
        Snapshot of the memory data passed in at the current tick.
    init_perturbation : dict[AttackerId, list of npt.NDArray] or None
        For each attacker that converged at the previous tick, the
        best perturbation tensor (clamped and halved) stored as numpy
        arrays. Loaded as the initial value for the next tick's
        optimizer when ``online`` is enabled.
    """

    previous_memory_data: AdvCPMemoryData | None
    current_memory_data: AdvCPMemoryData | None
    init_perturbation: dict[AttackerId, list[npt.NDArray]] | None


# Result tuple returned by every fusion-specific attack runner. The
# layout matches what ``CoperceptionModelManager`` already expects from
# the non-AdvCP inference helpers, with the visualization context
# appended as the fourth element.
AdvCPAttackResult: TypeAlias = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    AdvCPVisualizationContext,
]
