"""
Shared utilities used by every AdvCP fusion strategy.

Two helper classes live here:

- ``AdvCPAttackHelper``: configuration validation, attacker resolution,
  scenario-memory access, and target-box construction.
- ``AdvCPCarMeshHelper``: synthetic 3D mesh construction and ray
  tracing utilities used by the early-fusion attack to inject or
  suppress points in the attacker's lidar.

Both classes are purely static.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, NoReturn, Sequence, cast

import numpy as np
import numpy.typing as npt
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils
from opencood.utils.transformation_utils import x_to_world

from opencda.core.attack.advcp.types import (
    AdvCPAgentState,
    AdvCPBoxSpec,
    AdvCPConfig,
    AdvCPMemoryData,
    AdvCPScenarioData,
    AgentId,
    AttackerId,
    BatchAttackerId,
    BoxLwhBottomCenter,
    BoxModelTensor,
)
from opencda.core.common.coperception_data_processor import LiveMemorySnapshot

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.advcp_manager")


class AdvCPAttackHelper:
    """
    Configuration, attacker resolution, and target-box utilities shared
    by every AdvCP fusion strategy.
    """

    DENSITY_ALIASES = {
        "replace": 0,
        "dense_a": 1,
        "denseall": 2,
        "dense_all": 2,
        "sampled": 3,
    }

    @staticmethod
    def require_config_value(config: Mapping[str, Any], key: str, config_name: str = "AdvCP config") -> Any:
        """
        Look up ``key`` in ``config`` and raise if the value is ``None``.

        Parameters
        ----------
        config : Mapping
            Configuration mapping to query.
        key : str
            Key to look up.
        config_name : str, optional
            Human-readable name of the configuration, used in the error
            message. Defaults to ``"AdvCP config"``.

        Returns
        -------
        Any
            The value associated with ``key``.

        Raises
        ------
        ValueError
            If the key is missing or its value is ``None``.
        """
        value = config.get(key)
        if value is None:
            raise ValueError(f"Unexpected None in {config_name} for '{key}'.")
        return value

    @staticmethod
    def resolve_ego_agent_id(scenario_data: AdvCPScenarioData) -> AgentId:
        """
        Find the ego agent id inside a scenario-memory snapshot.

        The ego agent is the one with a truthy ``"ego"`` flag in its
        per-agent record.

        Parameters
        ----------
        scenario_data : AdvCPScenarioData
            Scenario memory snapshot for a single tick.

        Returns
        -------
        AgentId
            The agent id of the ego CAV.

        Raises
        ------
        ValueError
            If no ego agent is present.
        """
        ego_agent_id = next((agent_id for agent_id, agent_data in scenario_data.items() if agent_data.get("ego")), None)
        if ego_agent_id is None:
            raise ValueError("Unable to resolve ego agent for AdvCP attack.")
        return ego_agent_id

    @staticmethod
    def load_agent_state(scenario_data: AdvCPScenarioData, agent_id: AgentId) -> AdvCPAgentState:
        """
        Assemble an ``AdvCPAgentState`` from a scenario-memory snapshot.

        Resolves the agent's params either from an in-memory mapping or
        by loading the referenced YAML file, then extracts the lidar
        and ego poses.

        Parameters
        ----------
        scenario_data : AdvCPScenarioData
            Scenario memory snapshot for a single tick.
        agent_id : AgentId
            Id of the agent to look up.

        Returns
        -------
        AdvCPAgentState
            Resolved agent state.

        Raises
        ------
        ValueError
            If the agent record provides neither ``params`` nor ``yaml``.
        """
        agent_data = scenario_data[agent_id]
        timestamp = next(key for key in agent_data.keys() if key != "ego")
        snapshot = cast(Mapping[str, Any], agent_data[timestamp])
        yaml_path = cast(str | None, snapshot.get("yaml"))
        params = cast(Mapping[str, Any] | None, snapshot.get("params"))
        if params is None:
            if yaml_path is None:
                raise ValueError(f"AdvCP agent state for '{agent_id}' does not define either 'params' or 'yaml'.")

            params = cast(Mapping[str, Any], load_yaml(yaml_path))

        lidar_pose = params["lidar_pose"]
        return {
            "agent_id": agent_id,
            "timestamp": timestamp,
            "yaml_path": yaml_path,
            "params": params,
            "lidar_pose": lidar_pose,
            "ego_pose": params.get("true_ego_pos", lidar_pose),
        }

    @classmethod
    def resolve_configured_attacker_ids(cls, advcp_config: AdvCPConfig) -> list[AttackerId]:
        """
        Normalise the ``attacker_ids`` config entry.

        Strips whitespace, drops empty / ``None`` entries, deduplicates
        while preserving order.

        Parameters
        ----------
        advcp_config : AdvCPConfig
            Resolved AdvCP config.

        Returns
        -------
        list of AttackerId
            Normalised attacker ids in their original order.

        Raises
        ------
        ValueError
            If ``attacker_ids`` is missing or not a list.
        """
        attacker_ids_raw = cls.require_config_value(advcp_config, "attacker_ids")
        if not isinstance(attacker_ids_raw, list):
            raise ValueError("AdvCP config key 'attacker_ids' must be a sequence of agent ids.")

        ordered_ids: list[AttackerId] = []
        for attacker_id in attacker_ids_raw:
            if attacker_id is None:
                continue
            normalized_attacker_id = str(attacker_id).strip()
            if not normalized_attacker_id:
                continue
            if normalized_attacker_id in ordered_ids:
                continue
            ordered_ids.append(normalized_attacker_id)
        return ordered_ids

    @staticmethod
    def resolve_present_and_missing_attackers(
        configured_attacker_ids: Sequence[AttackerId],
        available_agent_ids: Iterable[Any],
    ) -> tuple[list[AttackerId], list[AttackerId]]:
        """
        Split configured attackers into those present and absent in a
        set of available agents.

        Parameters
        ----------
        configured_attacker_ids : Sequence of AttackerId
            Attackers requested by the AdvCP config.
        available_agent_ids : Iterable
            Agent ids that are actually available (e.g. inside the
            current batch or in the scenario memory).

        Returns
        -------
        tuple
            ``(present, missing)`` lists, each preserving the order of
            ``configured_attacker_ids``.
        """
        available_agent_id_set = {str(agent_id) for agent_id in available_agent_ids}
        present_attacker_ids: list[AttackerId] = []
        missing_attacker_ids: list[AttackerId] = []
        for attacker_id in configured_attacker_ids:
            if attacker_id in available_agent_id_set:
                present_attacker_ids.append(attacker_id)
            else:
                missing_attacker_ids.append(attacker_id)
        return present_attacker_ids, missing_attacker_ids

    @classmethod
    def resolve_attack_scope(
        cls,
        advcp_config: AdvCPConfig,
        memory_data: AdvCPMemoryData,
    ) -> tuple[AdvCPScenarioData, list[AttackerId], list[AttackerId], list[AttackerId]]:
        """
        Bundle scenario data and attacker resolution for a single tick.

        Convenience wrapper that returns everything an attack runner
        typically needs at the top of its loop.

        Parameters
        ----------
        advcp_config : AdvCPConfig
            Resolved AdvCP config.
        memory_data : AdvCPMemoryData
            Per-tick memory data (single batch element).

        Returns
        -------
        tuple
            ``(scenario_data, configured, present, missing)``.
        """
        scenario_data = next(iter(memory_data.values()))
        configured_attacker_ids = cls.resolve_configured_attacker_ids(advcp_config)
        present_attacker_ids, missing_attacker_ids = cls.resolve_present_and_missing_attackers(
            configured_attacker_ids,
            scenario_data.keys(),
        )
        return scenario_data, configured_attacker_ids, present_attacker_ids, missing_attacker_ids

    @classmethod
    def build_lidar_pose_map(cls, scenario_data: AdvCPScenarioData) -> dict[AgentId, npt.NDArray]:
        """
        Build an ``agent_id -> lidar_pose`` map for an entire scenario.

        Parameters
        ----------
        scenario_data : AdvCPScenarioData
            Scenario memory snapshot.

        Returns
        -------
        dict
            Mapping from agent id to a length-6 ``np.float32`` lidar
            pose ``[x, y, z, roll, yaw, pitch]``.
        """
        return {agent_id: np.asarray(cls.load_agent_state(scenario_data, agent_id)["lidar_pose"], dtype=np.float32) for agent_id in scenario_data}

    @staticmethod
    def resolve_agent_snapshot(scenario_data: AdvCPScenarioData, agent_id: AgentId) -> LiveMemorySnapshot:
        """
        Look up a single agent's ``LiveMemorySnapshot`` for the active
        timestamp.

        Parameters
        ----------
        scenario_data : AdvCPScenarioData
            Scenario memory snapshot.
        agent_id : AgentId
            Agent to look up.

        Returns
        -------
        LiveMemorySnapshot
            The per-tick OpenCOOD snapshot for ``agent_id``.
        """
        agent_data = scenario_data[agent_id]
        timestamp = next(key for key in agent_data.keys() if key != "ego")
        return cast(LiveMemorySnapshot, agent_data[timestamp])

    @staticmethod
    def require_agent_lidar(agent_snapshot: LiveMemorySnapshot, agent_id: AgentId, context: str) -> npt.NDArray:
        """
        Return ``agent_snapshot["lidar_np"]`` or raise a descriptive error.

        Parameters
        ----------
        agent_snapshot : LiveMemorySnapshot
            Snapshot to query.
        agent_id : AgentId
            Used for the error message only.
        context : str
            Caller name (for example ``"AdvCP early attack"``) used in
            the error message.

        Returns
        -------
        npt.NDArray
            ``(N, 4)`` lidar point cloud as ``float32``.

        Raises
        ------
        ValueError
            If the snapshot does not contain ``lidar_np``.
        """
        if (lidar := agent_snapshot.get("lidar_np")) is None:
            raise ValueError(f"{context} requires in-memory lidar_np for attacker '{agent_id}'.")
        return np.asarray(lidar, dtype=np.float32)

    @classmethod
    def resolve_density(cls, density_value: Any, context: str = "early attack") -> int:
        """
        Map a user-facing density string to its integer code.

        Parameters
        ----------
        density_value : Any
            One of the ``DENSITY_ALIASES`` keys (case-insensitive).
        context : str, optional
            Caller name used in the error message.

        Returns
        -------
        int
            Integer code consumed by the early-fusion attack.

        Raises
        ------
        ValueError
            If ``density_value`` is not a recognised alias.
        """
        normalized_value = density_value
        if isinstance(density_value, str):
            normalized_value = density_value.strip().lower()
        if normalized_value not in cls.DENSITY_ALIASES:
            supported_values = ", ".join(f"'{density}'" for density in cls.DENSITY_ALIASES)
            raise ValueError(f"Unsupported AdvCP {context} density '{density_value}'. Supported values are {supported_values}.")
        return cls.DENSITY_ALIASES[normalized_value]

    @staticmethod
    def build_batch_from_memory(dataset: Any, device: torch.device, memory_data: AdvCPMemoryData) -> Mapping[str, Any]:
        """
        Re-collate ``memory_data`` through the OpenCOOD dataset and move
        it to ``device``.

        Side effect: calls ``dataset.update_database(memory_data=...)``,
        which mutates the dataset in place. Callers that need to
        restore the original database must re-call this helper with
        the original memory afterwards.

        Parameters
        ----------
        dataset : Any
            OpenCOOD dataset instance.
        device : torch.device
            Target device for the collated batch.
        memory_data : AdvCPMemoryData
            Memory data to collate.

        Returns
        -------
        Mapping
            Collated batch on ``device``.
        """
        from opencood.tools import train_utils

        dataset.update_database(memory_data=memory_data)
        batch = dataset.collate_batch_test([dataset[0]])
        return train_utils.to_device(batch, device)

    @staticmethod
    def raise_no_configured_attackers(fusion_name: str) -> NoReturn:
        """
        Raise ``ValueError`` indicating the AdvCP config has no attackers.

        Parameters
        ----------
        fusion_name : str
            Name of the fusion strategy ("early", "intermediate",
            "late") used in the error message.
        """
        raise ValueError(f"AdvCP {fusion_name} attack cannot be applied because no attackers are configured.")

    @staticmethod
    def report_missing_attackers_from_current_batch(
        attacker_ids: Sequence[AttackerId],
        available_agent_ids: Iterable[Any],
        *,
        fusion_name: str | None = None,
    ) -> None:
        """
        Emit a structured warning when attackers are absent from the
        batch.

        Used to make it visible to the user that AdvCP is being skipped
        on a tick because the attackers happen not to be in
        communication range or otherwise not present.

        Parameters
        ----------
        attacker_ids : Sequence of AttackerId
            Attackers that are configured but missing.
        available_agent_ids : Iterable
            Agent ids that are actually present in the batch.
        fusion_name : str, optional
            Used to disambiguate the warning ("early", "intermediate",
            "late").
        """
        if not attacker_ids:
            return
        attack_prefix = "AdvCP attack" if fusion_name is None else f"AdvCP {fusion_name} attack"
        logger.warning(
            "%s will not be applied on this tick because none of the configured attackers are present in the current batch. "
            "Configured attackers: %s. Batch agents: %s. Continuing with normal cooperative perception inference.",
            attack_prefix,
            ", ".join(attacker_ids),
            ", ".join(str(agent_id) for agent_id in available_agent_ids),
        )

    @staticmethod
    def resolve_batch_agent_ids(batch_data: Mapping[str, Any], *, fallback_to_top_level: bool = True) -> list[AgentId]:
        """
        Extract the list of agent ids from a collated batch.

        Prefers ``batch_data["ego"]["origin_lidar_agent_ids"]`` (the
        ordered list OpenCOOD attaches when collating). Falls back to
        the top-level batch keys when the ordered list is unavailable
        and ``fallback_to_top_level`` is ``True``.

        Parameters
        ----------
        batch_data : Mapping
            Collated batch.
        fallback_to_top_level : bool, optional
            Whether to fall back to top-level keys when the ordered
            list is missing. Defaults to ``True``.

        Returns
        -------
        list of AgentId
            Agent ids present in the batch, possibly empty.
        """
        ego_entry = batch_data.get("ego")
        if isinstance(ego_entry, Mapping):
            agent_ids = ego_entry.get("origin_lidar_agent_ids")
            if isinstance(agent_ids, Sequence) and not isinstance(agent_ids, (str, bytes)):
                return [str(agent_id) for agent_id in agent_ids]

        if fallback_to_top_level:
            return [str(agent_id) for agent_id in batch_data.keys()]

        return []

    @classmethod
    def resolve_spoof_boxes_by_attacker(
        cls,
        advcp_config: AdvCPConfig,
        memory_data: AdvCPMemoryData | None,
    ) -> tuple[list[AttackerId], dict[BatchAttackerId, list[BoxLwhBottomCenter]]]:
        """
        Resolve target boxes per attacker, indexed by batch-side id.

        Used by the late-fusion attack to know which attacker produces
        which set of target boxes. The returned mapping uses
        ``BatchAttackerId`` keys: real agent ids for non-ego attackers,
        and the literal ``"ego"`` for an attacker that is itself the
        ego CAV (because OpenCOOD keys the ego entry under ``"ego"``).

        Parameters
        ----------
        advcp_config : AdvCPConfig
            Resolved AdvCP config.
        memory_data : Optional[AdvCPMemoryData]
            Per-tick memory data.

        Returns
        -------
        tuple
            ``(resolved_attacker_ids, boxes_by_batch_attacker)``.

        Raises
        ------
        ValueError
            If ``memory_data`` is ``None``.
        NotImplementedError
            If ``mode`` is not one of ``"spoofing"`` or ``"removal"``.
        """
        if memory_data is None:
            raise ValueError("AdvCP late attack requires current memory data.")

        mode = cls.require_config_value(advcp_config, "mode")
        match mode:
            case "removal" | "spoofing":
                pass
            case _:
                raise NotImplementedError(f"AdvCP mode '{mode}' is not available yet.")

        scenario_data, _, present_attacker_ids, _ = cls.resolve_attack_scope(advcp_config, memory_data)
        ego_agent_id = cls.resolve_ego_agent_id(scenario_data)
        resolved_attacker_ids: list[AttackerId] = []
        attack_boxes_by_batch_attacker: dict[BatchAttackerId, list[BoxLwhBottomCenter]] = {}

        for attacker_id in present_attacker_ids:
            _, _, _, attack_boxes = cls.resolve_spoof_boxes_for_agent(scenario_data, advcp_config, attacker_id)
            batch_attacker_id = "ego" if attacker_id == ego_agent_id else attacker_id
            attack_boxes_by_batch_attacker.setdefault(batch_attacker_id, []).extend(attack_boxes)
            resolved_attacker_ids.append(attacker_id)

        return resolved_attacker_ids, attack_boxes_by_batch_attacker

    @classmethod
    def resolve_spoof_boxes_for_agent(
        cls,
        scenario_data: AdvCPScenarioData,
        advcp_config: AdvCPConfig,
        attacker_id: AttackerId,
    ) -> tuple[AgentId, AdvCPAgentState, AdvCPAgentState, list[BoxLwhBottomCenter]]:
        """
        Build target boxes in the **attacker's** lidar frame.

        Each ``boxes`` entry is interpreted (relative to ego or
        absolute), composed with the attacker's lidar pose, and
        returned as a length-7 ``[x, y, z, l, w, h, yaw]`` array.

        Parameters
        ----------
        scenario_data : AdvCPScenarioData
            Scenario memory snapshot.
        advcp_config : AdvCPConfig
            Resolved AdvCP config.
        attacker_id : AttackerId
            Attacker whose lidar frame is the target frame.

        Returns
        -------
        tuple
            ``(ego_agent_id, ego_state, attacker_state, attack_boxes)``.

        Raises
        ------
        ValueError
            If the ``boxes`` config is missing or empty.
        """
        ego_agent_id = cls.resolve_ego_agent_id(scenario_data)
        ego_state = cls.load_agent_state(scenario_data, ego_agent_id)
        attacker_state = cls.load_agent_state(scenario_data, attacker_id)

        box_specs = cls.require_config_value(advcp_config, "boxes")
        if not isinstance(box_specs, list) or len(box_specs) == 0:
            raise ValueError("AdvCP config must define a non-empty boxes list.")

        attack_boxes = [
            cls.resolve_box_spec_for_sensor_pose(
                spec,
                index,
                advcp_config,
                ego_state,
                attacker_state["lidar_pose"],
            )
            for index, spec in enumerate(box_specs)
        ]
        return ego_agent_id, ego_state, attacker_state, attack_boxes

    @classmethod
    def resolve_spoof_boxes_for_ego(
        cls,
        scenario_data: AdvCPScenarioData,
        advcp_config: AdvCPConfig,
        attacker_id: AttackerId,
    ) -> tuple[AgentId, AdvCPAgentState, AdvCPAgentState, list[BoxLwhBottomCenter]]:
        """
        Build target boxes in the **ego** lidar frame.

        Same as :meth:`resolve_spoof_boxes_for_agent` but the resulting
        boxes are expressed in the ego lidar frame rather than the
        attacker's. Used by intermediate fusion to compute target
        feature-map indices that match the cooperatively-fused output
        space.

        Parameters
        ----------
        scenario_data : AdvCPScenarioData
            Scenario memory snapshot.
        advcp_config : AdvCPConfig
            Resolved AdvCP config.
        attacker_id : AttackerId
            Used only to resolve attacker state for the return tuple;
            the boxes themselves are placed in the ego lidar frame.

        Returns
        -------
        tuple
            ``(ego_agent_id, ego_state, attacker_state, attack_boxes)``.
        """
        ego_agent_id = cls.resolve_ego_agent_id(scenario_data)
        ego_state = cls.load_agent_state(scenario_data, ego_agent_id)
        attacker_state = cls.load_agent_state(scenario_data, attacker_id)

        box_specs = cls.require_config_value(advcp_config, "boxes")
        if not isinstance(box_specs, list) or len(box_specs) == 0:
            raise ValueError("AdvCP config must define a non-empty boxes list.")

        attack_boxes = [
            cls.resolve_box_spec_for_sensor_pose(
                spec,
                index,
                advcp_config,
                ego_state,
                ego_state["lidar_pose"],
            )
            for index, spec in enumerate(box_specs)
        ]
        return ego_agent_id, ego_state, attacker_state, attack_boxes

    @classmethod
    def resolve_box_spec_for_sensor_pose(
        cls,
        spec: AdvCPBoxSpec,
        index: int,
        advcp_config: AdvCPConfig,
        ego_state: AdvCPAgentState,
        sensor_pose: Sequence[float],
    ) -> BoxLwhBottomCenter:
        """
        Resolve a single ``AdvCPBoxSpec`` into a target box in a chosen
        sensor frame.

        Validates the spec, picks ``size`` (from the spec or the global
        default), composes a world-frame pose (when ``relative``), and
        finally projects into ``sensor_pose``.

        Parameters
        ----------
        spec : AdvCPBoxSpec
            Single box spec from the AdvCP YAML.
        index : int
            Position of the spec in the ``boxes`` list (used for error
            messages only).
        advcp_config : AdvCPConfig
            Resolved AdvCP config (provides ``default_size``).
        ego_state : AdvCPAgentState
            Ego state (used to interpret ``relative`` poses).
        sensor_pose : Sequence of float
            Target sensor pose ``[x, y, z, roll, yaw, pitch]``.

        Returns
        -------
        BoxLwhBottomCenter
            ``[x, y, z, length, width, height, yaw]`` in the target
            sensor frame.

        Raises
        ------
        ValueError
            If the spec is malformed.
        """
        if not isinstance(spec, dict):
            raise ValueError(f"AdvCP box entry #{index} must be a mapping.")

        has_relative = "relative" in spec
        has_absolute = "absolute" in spec
        if has_relative == has_absolute:
            raise ValueError(f"AdvCP box entry #{index} must define exactly one of 'relative' or 'absolute'.")

        pose = np.asarray(spec["relative"] if has_relative else spec["absolute"], dtype=np.float32)
        if pose.shape != (6,):
            raise ValueError(f"boxes[{index}] must contain 6 values: [x, y, z, roll, yaw, pitch].")

        size = np.asarray(
            spec.get("size", cls.require_config_value(advcp_config, "default_size")),
            dtype=np.float32,
        )
        if size.shape != (3,):
            raise ValueError(f"boxes[{index}].size must contain 3 values: [length, width, height].")

        if has_relative:
            world_pose = cls.compose_relative_pose(ego_state["ego_pose"], pose)
        else:
            world_pose = pose

        return cls.world_box_to_sensor_box(world_pose, size, sensor_pose)

    @staticmethod
    def compose_relative_pose(reference_pose: npt.NDArray | Sequence[float], relative_pose: npt.NDArray) -> npt.NDArray:
        """
        Convert a pose expressed relative to ``reference_pose`` into a
        world-frame pose.

        The translation is propagated through the homogeneous transform
        of ``reference_pose``; rotations are added componentwise (in
        degrees).

        Parameters
        ----------
        reference_pose : Sequence of float or npt.NDArray
            Frame in world coordinates: ``[x, y, z, roll, yaw, pitch]``.
        relative_pose : npt.NDArray
            Pose relative to the reference frame, same layout.

        Returns
        -------
        npt.NDArray
            Composed pose in world coordinates, same layout.
        """
        reference_pose_array = np.asarray(reference_pose, dtype=np.float32)
        reference_matrix = x_to_world(reference_pose_array.tolist())
        # Lift the relative XYZ into a homogeneous point and apply the
        # reference frame's world transform; Euler angles are summed
        # componentwise because the OpenCOOD pose representation is
        # additive in [roll, yaw, pitch].
        relative_point = np.array([relative_pose[0], relative_pose[1], relative_pose[2], 1.0], dtype=np.float32)
        world_point = reference_matrix @ relative_point

        world_pose = np.zeros(6, dtype=np.float32)
        world_pose[:3] = world_point[:3]
        world_pose[3:] = reference_pose_array[3:] + relative_pose[3:]
        return world_pose

    @staticmethod
    def world_box_to_sensor_box(world_pose: npt.NDArray, size: npt.NDArray, sensor_pose: Sequence[float]) -> BoxLwhBottomCenter:
        """
        Project a world-frame box centre into a sensor frame.

        Computes the inverse of ``sensor_pose`` and applies it to the
        world-frame translation; subtracts ``sensor_pose`` yaw from
        ``world_pose`` yaw and converts to radians.

        Parameters
        ----------
        world_pose : npt.NDArray
            ``[x, y, z, roll, yaw, pitch]`` in world coordinates.
        size : npt.NDArray
            ``[length, width, height]``.
        sensor_pose : Sequence of float
            Target sensor pose, same layout as ``world_pose``.

        Returns
        -------
        BoxLwhBottomCenter
            Length-7 box in the sensor frame:
            ``[x, y, z, length, width, height, yaw_rad]``.
        """
        sensor_matrix = x_to_world(list(sensor_pose))
        world_to_sensor = np.linalg.inv(sensor_matrix)
        world_point = np.array([world_pose[0], world_pose[1], world_pose[2], 1.0], dtype=np.float32)
        sensor_point = world_to_sensor @ world_point

        # OpenCOOD stores Euler angles in degrees. The model expects
        # yaw in radians, so we subtract first and convert once at the
        # end.
        yaw_sensor = np.radians(float(world_pose[4] - sensor_pose[4]))

        return np.array(
            [
                sensor_point[0],
                sensor_point[1],
                sensor_point[2],
                size[0],
                size[1],
                size[2],
                yaw_sensor,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def convert_box_for_model(box_lwh_bottom_center: BoxLwhBottomCenter, dataset: Any) -> BoxModelTensor:
        """
        Convert a sensor-frame box into the dataset's model input format.

        OpenCOOD post-processors expect either ``"hwl"`` or ``"lwh"``
        ordering. The internal AdvCP representation stores
        ``[x, y, z, l, w, h, yaw]`` with ``z`` at the bottom-centre;
        this helper reorders the axes if required and shifts ``z`` to
        the box centre.

        Parameters
        ----------
        box_lwh_bottom_center : BoxLwhBottomCenter
            Internal AdvCP box, length 7.
        dataset : Any
            Dataset whose ``post_processor.params['order']`` selects
            ``"hwl"`` (default) or ``"lwh"``.

        Returns
        -------
        BoxModelTensor
            Length-7 tensor in the model's expected ordering, with
            ``z`` raised to the centre of the box.

        Raises
        ------
        NotImplementedError
            If the order is not one of ``"hwl"`` or ``"lwh"``.
        """
        model_box = np.copy(box_lwh_bottom_center)
        order = dataset.post_processor.params.get("order", "hwl")

        if order == "hwl":
            # Internal layout is l, w, h; model wants h, w, l.
            model_box[3:6] = model_box[[5, 4, 3]]
            # After reordering, model_box[3] is height; raise z by half-height.
            model_box[2] += 0.5 * model_box[3]
        elif order == "lwh":
            # Same axis order; raise z by half-height (model_box[5]).
            model_box[2] += 0.5 * model_box[5]
        else:
            raise NotImplementedError(f"Unsupported box order for AdvCP spoofing: {order}")

        return torch.from_numpy(model_box).type(torch.float32)

    @staticmethod
    def model_boxes_to_lwh(boxes: torch.Tensor, dataset: Any) -> torch.Tensor:
        """
        Reorder model-format boxes into the canonical ``lwh`` order.

        OpenCOOD models can be configured with either ``"lwh"`` or
        ``"hwl"`` axis ordering for box dimensions. AdvCP geometry
        utilities expect ``"lwh"``. This helper rearranges columns when
        needed and returns the original tensor when already correct.

        Parameters
        ----------
        boxes : torch.Tensor
            ``(N, 7)`` boxes in the dataset/model order.
        dataset : Any
            Dataset whose ``post_processor.params["order"]`` selects
            the model box order.

        Returns
        -------
        torch.Tensor
            ``(N, 7)`` boxes in ``[x, y, z, l, w, h, yaw]`` order.

        Raises
        ------
        NotImplementedError
            If the dataset's order is not one of ``"hwl"`` or ``"lwh"``.
        """
        if boxes.numel() == 0:
            return boxes
        order = dataset.post_processor.params.get("order", "hwl")
        if order == "hwl":
            return boxes[:, [0, 1, 2, 5, 4, 3, 6]]
        if order == "lwh":
            return boxes
        raise NotImplementedError(f"Unsupported box order for AdvCP geometry: {order}")

    @staticmethod
    def compute_iou_weights(proposals_lwh: torch.Tensor, target_box_lwh: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D oriented IoU between every proposal and one target.

        Both proposals and target are interpreted as ``lwh`` boxes and
        projected to BEV/XY oriented rectangles. The returned IoU values
        are used by intermediate-fusion losses and late-fusion removal
        filtering.

        Parameters
        ----------
        proposals_lwh : torch.Tensor
            ``(N, 7)`` proposal boxes in ``lwh`` order.
        target_box_lwh : torch.Tensor
            ``(7,)`` target box in ``lwh`` order.

        Returns
        -------
        torch.Tensor
            ``(N,)`` IoU values in ``[0, 1]``.
        """
        if proposals_lwh.shape[0] == 0:
            return torch.zeros((0,), dtype=target_box_lwh.dtype, device=target_box_lwh.device)

        repeated_target_boxes = target_box_lwh.unsqueeze(0).expand(proposals_lwh.shape[0], -1)
        proposal_corners = box_utils.boxes_to_corners2d(proposals_lwh, order="lwh")[:, :, :2]
        target_corners = box_utils.boxes_to_corners2d(repeated_target_boxes, order="lwh")[:, :, :2]
        intersection_area = AdvCPAttackHelper._oriented_box_intersection_2d(
            proposal_corners,
            target_corners,
        )
        proposal_area = proposals_lwh[:, 3] * proposals_lwh[:, 4]
        target_area = repeated_target_boxes[:, 3] * repeated_target_boxes[:, 4]
        union_area = torch.clamp(proposal_area + target_area - intersection_area, min=1e-6)
        return torch.clamp(intersection_area / union_area, min=0.0, max=1.0)

    @classmethod
    def _oriented_box_intersection_2d(
        cls,
        corners1: torch.Tensor,
        corners2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable 2D oriented-box intersection area, batched.

        Builds the intersection polygon from contained corners and
        edge-edge intersections, then computes its area with the
        shoelace formula.
        """
        intersections, intersection_mask = cls._box_intersection_th(corners1, corners2)
        corners1_in_corners2, corners2_in_corners1 = cls._box_in_box_th(corners1, corners2)
        vertices = torch.cat(
            [
                corners1,
                corners2,
                intersections.reshape(corners1.shape[0], -1, 2),
            ],
            dim=1,
        )
        vertex_mask = torch.cat(
            [
                corners1_in_corners2,
                corners2_in_corners1,
                intersection_mask.reshape(corners1.shape[0], -1),
            ],
            dim=1,
        )
        return cls._calculate_polygon_area(vertices, vertex_mask)

    @staticmethod
    def _box_intersection_th(
        corners1: torch.Tensor,
        corners2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute every edge-edge intersection between two batched boxes.
        """
        epsilon = 1e-8
        lines1 = torch.cat([corners1, corners1[:, [1, 2, 3, 0], :]], dim=2)
        lines2 = torch.cat([corners2, corners2[:, [1, 2, 3, 0], :]], dim=2)

        lines1_expanded = lines1.unsqueeze(2).repeat(1, 1, 4, 1)
        lines2_expanded = lines2.unsqueeze(1).repeat(1, 4, 1, 1)
        x1 = lines1_expanded[..., 0]
        y1 = lines1_expanded[..., 1]
        x2 = lines1_expanded[..., 2]
        y2 = lines1_expanded[..., 3]
        x3 = lines2_expanded[..., 0]
        y3 = lines2_expanded[..., 1]
        x4 = lines2_expanded[..., 2]
        y4 = lines2_expanded[..., 3]

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        denominator_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        denominator_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)

        t = denominator_t / torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        u = -denominator_u / torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        t = torch.where(denominator == 0, torch.full_like(t, -1.0), t)
        u = torch.where(denominator == 0, torch.full_like(u, -1.0), u)

        mask_t = (t > 0) & (t < 1)
        mask_u = (u > 0) & (u < 1)
        intersection_mask = mask_t & mask_u

        stable_t = denominator_t / (denominator + epsilon)
        intersections = torch.stack(
            [
                x1 + stable_t * (x2 - x1),
                y1 + stable_t * (y2 - y1),
            ],
            dim=-1,
        )
        intersections = intersections * intersection_mask.unsqueeze(-1).to(intersections.dtype)
        return intersections, intersection_mask

    @classmethod
    def _box_in_box_th(
        cls,
        corners1: torch.Tensor,
        corners2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Test corner containment in both directions.
        """
        return cls._box1_in_box2(corners1, corners2), cls._box1_in_box2(corners2, corners1)

    @staticmethod
    def _box1_in_box2(corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
        """
        Boolean mask: which corners of box 1 lie inside box 2.
        """
        corner_a = corners2[:, 0:1, :]
        corner_b = corners2[:, 1:2, :]
        corner_d = corners2[:, 3:4, :]
        vector_ab = corner_b - corner_a
        vector_am = corners1 - corner_a
        vector_ad = corner_d - corner_a
        projection_ab = torch.sum(vector_ab * vector_am, dim=-1)
        norm_ab = torch.sum(vector_ab * vector_ab, dim=-1)
        projection_ad = torch.sum(vector_ad * vector_am, dim=-1)
        norm_ad = torch.sum(vector_ad * vector_ad, dim=-1)

        condition_ab = (projection_ab / norm_ab > -1e-6) & (projection_ab / norm_ab < 1.0 + 1e-6)
        condition_ad = (projection_ad / norm_ad > -1e-6) & (projection_ad / norm_ad < 1.0 + 1e-6)
        return condition_ab & condition_ad

    @staticmethod
    def _calculate_polygon_area(vertices: torch.Tensor, vertex_mask: torch.Tensor) -> torch.Tensor:
        """
        Shoelace area of a batched polygon with masked vertices.
        """
        num_valid = vertex_mask.to(torch.int32).sum(dim=1)
        safe_num_valid = torch.clamp(num_valid, min=1).to(vertices.dtype)
        centroid = (vertices * vertex_mask.unsqueeze(-1).to(vertices.dtype)).sum(dim=1) / safe_num_valid.unsqueeze(-1)
        normalized_vertices = vertices - centroid.unsqueeze(1)
        angles = torch.atan2(normalized_vertices[..., 1], normalized_vertices[..., 0])
        invalid_angle = torch.full_like(angles, float("inf"))
        sorted_indices = torch.argsort(torch.where(vertex_mask, angles, invalid_angle), dim=1)

        gather_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, 2)
        sorted_vertices = torch.gather(vertices, 1, gather_indices)
        max_vertices = sorted_vertices.shape[1]
        vertex_indices = torch.arange(max_vertices, device=vertices.device).unsqueeze(0).expand(sorted_vertices.shape[0], -1)
        next_indices = torch.where(
            vertex_indices + 1 < num_valid.unsqueeze(1),
            vertex_indices + 1,
            torch.zeros_like(vertex_indices),
        )
        valid_edge_mask = vertex_indices < num_valid.unsqueeze(1)
        next_gather_indices = next_indices.unsqueeze(-1).expand(-1, -1, 2)
        next_vertices = torch.gather(sorted_vertices, 1, next_gather_indices)

        cross_products = sorted_vertices[..., 0] * next_vertices[..., 1] - sorted_vertices[..., 1] * next_vertices[..., 0]
        polygon_area = torch.abs((cross_products * valid_edge_mask.to(sorted_vertices.dtype)).sum(dim=1)) / 2.0
        return torch.where(num_valid >= 3, polygon_area, torch.zeros_like(polygon_area))


class AdvCPCarMeshHelper:
    """
    3D mesh and ray-tracing utilities used by the early-fusion attack.

    The early-fusion attack injects or suppresses lidar points by
    ray-tracing a synthetic mesh placed at the configured target box.
    This class builds the meshes (either a real bundled car mesh or a
    fallback bounding-box shell) and performs ray-mesh intersection
    via Open3D.

    Class state
    -----------
    CAR_MESH_3D_EXAMPLES : dict
        Reference bounding boxes ``[x, y, z, l, w, h, yaw]`` for the
        bundled car meshes; used to compute scale factors when fitting
        the mesh to a configured target box.
    _CAR_MESH_DIVIDE_CACHE : dict
        Cache of mesh-divide pickles loaded from disk, keyed by path.
    _REAL_MESH_WARNING_EMITTED : bool
        One-shot guard to avoid spamming a fallback warning when the
        bundled car mesh assets are missing.
    """

    CAR_MESH_3D_EXAMPLES = {
        "car_000000": np.array([0.0, 0.0, 0.0, 5.00, 2.00, 1.75, 0.0], dtype=np.float32),
        "car_mesh_0200": np.array([0.0, 0.0, 0.0, 4.30, 1.91, 1.26, 0.0], dtype=np.float32),
    }
    _CAR_MESH_DIVIDE_CACHE: dict[Path, Any] = {}
    _REAL_MESH_WARNING_EMITTED = False

    @staticmethod
    def build_spoof_mesh_pieces(spoof_box: BoxLwhBottomCenter) -> list[Any]:
        """
        Build a four-wall bounding-box shell for ray tracing.

        Used as a fallback when the bundled car mesh is unavailable.
        Each wall is a thin box along one face of the target box;
        together they approximate the silhouette of a car.

        Parameters
        ----------
        spoof_box : BoxLwhBottomCenter
            Target box ``[x, y, z, l, w, h, yaw]``.

        Returns
        -------
        list of Open3D meshes
            Four wall meshes, one per long/short side.
        """
        thickness = 0.05
        length = float(spoof_box[3])
        width = float(spoof_box[4])
        height = float(spoof_box[5])

        # Two walls along the long axis (front/back), two along the
        # short axis (left/right). Each wall is offset to the
        # corresponding face and given a 5 cm thickness so it casts a
        # solid silhouette during ray tracing.
        pieces = [
            ((thickness, width, height), (length / 2.0 - thickness / 2.0, 0.0, height / 2.0)),
            ((thickness, width, height), (-length / 2.0 + thickness / 2.0, 0.0, height / 2.0)),
            ((length, thickness, height), (0.0, width / 2.0 - thickness / 2.0, height / 2.0)),
            ((length, thickness, height), (0.0, -width / 2.0 + thickness / 2.0, height / 2.0)),
        ]
        return [AdvCPCarMeshHelper.build_box_piece_mesh(spoof_box, extents, center) for extents, center in pieces]

    @classmethod
    def build_spoof_meshes(cls, spoof_box: BoxLwhBottomCenter, advcp_config: AdvCPConfig) -> list[Any]:
        """
        Build the spoofing meshes for a target box.

        Prefers the bundled real car mesh (split into pieces via
        ``car_mesh_divide.pkl``) and falls back to the bounding-box
        shell when the assets are missing.

        Parameters
        ----------
        spoof_box : BoxLwhBottomCenter
            Target box ``[x, y, z, l, w, h, yaw]``.
        advcp_config : AdvCPConfig
            Resolved AdvCP config (provides mesh asset paths).

        Returns
        -------
        list of Open3D meshes
            Real car mesh pieces or the four-wall fallback.
        """
        car_mesh_pieces = cls.build_real_car_mesh_pieces(spoof_box, advcp_config)
        if car_mesh_pieces is not None:
            return car_mesh_pieces
        if not cls._REAL_MESH_WARNING_EMITTED:
            logger.warning("AdvCP early spoofing 3D model assets were not found. Falling back to bbox-shell ray tracing instead of real car mesh.")
            cls._REAL_MESH_WARNING_EMITTED = True
        return cls.build_spoof_mesh_pieces(spoof_box)

    @classmethod
    def build_collision_mesh(cls, spoof_box: BoxLwhBottomCenter, advcp_config: AdvCPConfig) -> Any:
        """
        Build a single solid mesh used for occlusion testing.

        Used by the dense-spoofing path to determine which existing
        lidar points are occluded by the spoofed object (and should
        therefore be replaced with the surface intersection of the
        spoof mesh).

        Parameters
        ----------
        spoof_box : BoxLwhBottomCenter
            Target box.
        advcp_config : AdvCPConfig
            Resolved AdvCP config.

        Returns
        -------
        Open3D mesh
            Either the merged car mesh, the single car mesh piece, or
            a solid box matching the target's dimensions.
        """
        car_mesh_pieces = cls.build_real_car_mesh_pieces(spoof_box, advcp_config)
        if car_mesh_pieces is not None:
            return car_mesh_pieces[0] if len(car_mesh_pieces) == 1 else cls.merge_meshes(car_mesh_pieces)
        return cls.build_box_piece_mesh(
            spoof_box,
            (float(spoof_box[3]), float(spoof_box[4]), float(spoof_box[5])),
            (0.0, 0.0, float(spoof_box[5]) / 2.0),
        )

    @classmethod
    def build_real_car_mesh_pieces(cls, spoof_box: BoxLwhBottomCenter, advcp_config: AdvCPConfig) -> list[Any] | None:
        """
        Load and post-process the bundled car mesh for a target box.

        Reads the triangle mesh from ``car_mesh_path`` and, when
        available, the per-region vertex index list from
        ``car_mesh_divide_path`` (cached on first load). Each piece is
        then scaled, rotated, and translated to fit the target box.

        Parameters
        ----------
        spoof_box : BoxLwhBottomCenter
            Target box.
        advcp_config : AdvCPConfig
            Resolved AdvCP config (provides asset paths).

        Returns
        -------
        Optional[list of Open3D meshes]
            ``None`` when the asset is missing or empty; otherwise the
            list of post-processed mesh pieces.
        """
        import open3d as o3d

        # TODO: Replace bundled car_mesh/car_mesh_divide asset loading with on-the-fly asset generation
        # once AdvCP issue #8 is resolved: https://github.com/zqzqz/AdvCollaborativePerception/issues/8
        car_mesh_path, car_mesh_divide_path = cls.resolve_car_mesh_paths(advcp_config)
        if car_mesh_path is None or not car_mesh_path.exists():
            return None

        car_mesh = o3d.io.read_triangle_mesh(str(car_mesh_path))
        if car_mesh.is_empty():
            return None

        car_mesh_name = cls.car_mesh_name_from_path(car_mesh_path)
        if car_mesh_divide_path is not None and car_mesh_divide_path.exists():
            if car_mesh_divide_path not in cls._CAR_MESH_DIVIDE_CACHE:
                with open(car_mesh_divide_path, "rb") as handle:
                    cls._CAR_MESH_DIVIDE_CACHE[car_mesh_divide_path] = pickle.load(handle)
            car_mesh_pieces = [car_mesh.select_by_index(vertex_indices) for vertex_indices in cls._CAR_MESH_DIVIDE_CACHE[car_mesh_divide_path]]
        else:
            car_mesh_pieces = [car_mesh]
        return cls.post_process_car_meshes(car_mesh_pieces, spoof_box, car_mesh_name)

    @classmethod
    def post_process_car_meshes(cls, car_mesh_pieces: list[Any], spoof_box: BoxLwhBottomCenter, car_mesh_name: str) -> list[Any]:
        """
        Fit each mesh piece into the configured target box.

        Applies a uniform scale (preserving the mesh proportions) and a
        yaw rotation, then translates to the box's bottom centre.

        Parameters
        ----------
        car_mesh_pieces : list of Open3D meshes
            Mesh pieces produced by :meth:`build_real_car_mesh_pieces`.
        spoof_box : BoxLwhBottomCenter
            Target box.
        car_mesh_name : str
            Stem of the mesh file; used to look up the reference
            bounding box in ``CAR_MESH_3D_EXAMPLES``.

        Returns
        -------
        list of Open3D meshes
            Transformed mesh pieces ready for ray tracing.
        """
        processed_meshes = []
        car_mesh_bbox = cls.CAR_MESH_3D_EXAMPLES.get(car_mesh_name, cls.CAR_MESH_3D_EXAMPLES["car_mesh_0200"])
        # Use the smallest per-axis scale ratio to preserve aspect.
        scale = float(np.min(spoof_box[3:6] / car_mesh_bbox[3:6]))
        rotation = np.array(
            [
                [np.cos(spoof_box[6]), -np.sin(spoof_box[6]), 0.0],
                [np.sin(spoof_box[6]), np.cos(spoof_box[6]), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        for car_mesh_piece in car_mesh_pieces:
            processed_mesh = copy.deepcopy(car_mesh_piece)
            processed_mesh.scale(scale, np.zeros(3, dtype=np.float64))
            processed_mesh.rotate(rotation, np.zeros(3, dtype=np.float64))
            processed_mesh.translate(np.asarray(spoof_box[:3], dtype=np.float64))
            processed_meshes.append(processed_mesh)

        return processed_meshes

    @staticmethod
    def car_mesh_name_from_path(car_mesh_path: Path | None) -> str:
        """
        Extract a mesh name (file stem) from a path, with a default.

        Parameters
        ----------
        car_mesh_path : Optional[Path]

        Returns
        -------
        str
            ``car_mesh_path.stem`` if available, else ``"car_mesh_0200"``.
        """
        if car_mesh_path is None:
            return "car_mesh_0200"
        return car_mesh_path.stem

    @classmethod
    def resolve_car_mesh_paths(cls, advcp_config: AdvCPConfig) -> tuple[Path, Path]:
        """
        Resolve the bundled car-mesh asset paths to absolute filesystem
        paths.

        Parameters
        ----------
        advcp_config : AdvCPConfig
            Resolved AdvCP config.

        Returns
        -------
        tuple of Path
            ``(car_mesh_path, car_mesh_divide_path)``.
        """
        car_mesh_path = Path(str(AdvCPAttackHelper.require_config_value(advcp_config, "car_mesh_path"))).expanduser()

        if not car_mesh_path.is_absolute():
            car_mesh_path = (Path.cwd() / car_mesh_path).resolve()

        car_mesh_divide_path = Path(str(AdvCPAttackHelper.require_config_value(advcp_config, "car_mesh_divide_path"))).expanduser()
        if not car_mesh_divide_path.is_absolute():
            car_mesh_divide_path = (Path.cwd() / car_mesh_divide_path).resolve()

        return car_mesh_path, car_mesh_divide_path

    @staticmethod
    def build_box_piece_mesh(
        spoof_box: BoxLwhBottomCenter,
        extents: tuple[float, float, float],
        center_local: tuple[float, float, float],
    ) -> Any:
        """
        Build a single rotated, translated 3D box mesh.

        Constructs an axis-aligned box with the given ``extents``,
        offsets it so its centre sits at ``center_local`` (relative to
        the box origin), then rotates by ``spoof_box`` yaw and
        translates to ``spoof_box[:3]``.

        Parameters
        ----------
        spoof_box : BoxLwhBottomCenter
            Provides the world-frame translation and yaw.
        extents : tuple of float
            ``(extent_x, extent_y, extent_z)`` of the piece.
        center_local : tuple of float
            ``(x, y, z)`` centre of the piece in the local box frame
            (before the ``spoof_box`` rotation and translation).

        Returns
        -------
        Open3D mesh
            The transformed piece.
        """
        import open3d as o3d

        x, y, z, _, _, _, yaw = spoof_box.tolist()
        extent_x, extent_y, extent_z = extents
        mesh = o3d.geometry.TriangleMesh.create_box(width=extent_x, height=extent_y, depth=extent_z)
        # Centre the axis-aligned box at the origin in XY (Open3D
        # creates it in the positive octant), then offset to the
        # requested local centre with a half-height correction.
        mesh.translate(np.array([-extent_x / 2.0, -extent_y / 2.0, 0.0], dtype=np.float64))
        mesh.translate(np.array([center_local[0], center_local[1], center_local[2] - extent_z / 2.0], dtype=np.float64))
        rotation = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        mesh.rotate(rotation, np.zeros(3, dtype=np.float64))
        mesh.translate(np.array([x, y, z], dtype=np.float64))
        return mesh

    @staticmethod
    def merge_meshes(meshes: list[Any]) -> Any:
        """
        Merge a non-empty list of Open3D meshes into a single mesh.

        Parameters
        ----------
        meshes : list of Open3D meshes

        Returns
        -------
        Open3D mesh
            The merged result.
        """
        merged_mesh = copy.deepcopy(meshes[0])
        for mesh in meshes[1:]:
            merged_mesh += mesh
        return merged_mesh

    @staticmethod
    def ray_intersection(meshes: list[Any], rays: npt.NDArray) -> npt.NDArray:
        """
        Intersect a batch of rays with a set of meshes.

        Builds an Open3D ``RaycastingScene`` from ``meshes``, casts
        each ray, and returns the world-space intersection point per
        ray (or ``inf`` when the ray misses).

        Parameters
        ----------
        meshes : list of Open3D meshes
            Geometry to intersect against.
        rays : npt.NDArray
            ``(N, 6)`` float32 array. Each row is
            ``[ox, oy, oz, dx, dy, dz]`` (origin + unit direction).

        Returns
        -------
        npt.NDArray
            ``(N, 3)`` float32 array of intersection points; rows
            corresponding to misses are filled with ``inf``.
        """
        import open3d as o3d

        scene = o3d.t.geometry.RaycastingScene()
        mesh_id_map = {}
        for mesh in meshes:
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            mesh_id = scene.add_triangles(mesh_t)
            mesh_id_map[mesh_id] = mesh

        ray_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans_raw = scene.cast_rays(ray_tensor)
        ans = {key: value.numpy() for key, value in ans_raw.items()}

        intersections = np.full((rays.shape[0], 3), np.inf, dtype=np.float32)
        for ray_index in range(rays.shape[0]):
            # t_hit > 10000 indicates a miss (Open3D returns a very
            # large finite value rather than infinity for missed rays).
            if ans["t_hit"][ray_index] > 10000:
                continue
            # Recover the world-space intersection point from the
            # triangle vertices and barycentric (uv) coordinates.
            mesh = mesh_id_map[ans["geometry_ids"][ray_index]]
            triangle_vertices = np.asarray(mesh.triangles)[ans["primitive_ids"][ray_index]]
            vertices = np.asarray(mesh.vertices)[triangle_vertices]
            uv = ans["primitive_uvs"][ray_index]
            intersections[ray_index] = (1.0 - np.sum(uv)) * vertices[0] + uv[0] * vertices[1] + uv[1] * vertices[2]

        return intersections
