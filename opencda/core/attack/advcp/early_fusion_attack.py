"""
AdvCP early-fusion attack.

Early fusion shares raw lidar point clouds between CAVs before any
neural processing. The cooperative perception model receives a single
merged point cloud (transformed into the ego frame). To attack at this
layer, AdvCP rewrites the attacker's lidar point cloud in shared
memory so that the merged cloud already contains injected or removed
points by the time it reaches the model.

Spoofing
--------
For each configured target box, build a synthetic 3D mesh (a real car
mesh when the bundled assets are available, otherwise a four-wall
bounding-box shell). Cast each lidar ray and replace the points whose
ray hits the mesh with the surface intersection. The resulting cloud
contains a realistic-looking obstacle at the target location.

Removal
-------
Each target box becomes a "removal zone". Existing lidar returns
inside that zone are replaced with farther-away points (sampled from a
wall mesh, an adversarial-shape mesh, or the local ground plane) so
that the perception model no longer sees the object.

Density modes
-------------
The integer density code (resolved from the user's ``density``
config string) controls how many synthetic points per target box are
written:

- 0 / ``"replace"``: replace only the points already inside the
  target zone.
- 1 / ``"dense_a"``: also synthesise one additional auxiliary ray
  origin offset toward the target.
- 2 / ``"dense_all"``: additionally synthesise one auxiliary ray
  origin per neighbouring CAV pose.
- 3 / ``"sampled"``: weight per-mesh sampling by visibility solid
  angle from each non-attacker viewpoint and sample replacement points
  accordingly. This is the default.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch
from opencood.tools import inference_utils
from opencood.utils.transformation_utils import x_to_world

from opencda.core.attack.advcp.attack_helper import AdvCPAttackHelper, AdvCPCarMeshHelper
from opencda.core.attack.advcp.types import (
    AdvCPAttackResult,
    AdvCPConfig,
    AdvCPMemoryData,
    AdvCPVisualizationContext,
    AgentId,
    AttackerId,
    BoxLwhBottomCenter,
)

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.early_fusion_attack")


class AdvCoperceptionEarlyFusionAttack:
    """
    Early-fusion AdvCP attack runner.

    Stateless except for a one-shot warning flag for missing
    adversarial-shape assets. All public entry points are class /
    static methods.

    Attributes
    ----------
    _REMOVE_ADV_SHAPE_WARNING_EMITTED : bool
        Class-level guard preventing repeated fallback warnings when
        the optional adversarial-shape mesh assets are missing.
    """

    _REMOVE_ADV_SHAPE_WARNING_EMITTED = False

    @classmethod
    def run(
        cls,
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: AdvCPConfig,
        memory_data: AdvCPMemoryData | None = None,
    ) -> AdvCPAttackResult:
        """
        Execute the early-fusion attack for a single tick.

        Dispatches to the spoofing or removal handler based on
        ``mode`` and validates that ``memory_data`` is available
        (required to access the attackers' lidar clouds).

        Parameters
        ----------
        batch_data : Any
            Collated batch.
        model : Any
            Cooperative perception model.
        dataset : Any
            OpenCOOD dataset (provides post-processing, range filters,
            and the database mutation API).
        device : torch.device
            Device the model lives on.
        advcp_config : AdvCPConfig
            Resolved AdvCP config.
        memory_data : Optional[AdvCPMemoryData]
            Per-tick memory data; required for any non-fallback path.

        Returns
        -------
        AdvCPAttackResult

        Raises
        ------
        NotImplementedError
            If ``mode`` is not one of ``"spoofing"`` or ``"removal"``.
        ValueError
            If ``memory_data`` is ``None`` for a supported mode.
        """
        mode = AdvCPAttackHelper.require_config_value(advcp_config, "mode")
        handler_by_mode = {
            "spoofing": cls._run_spoof,
            "removal": cls._run_removal,
        }

        def fail(*_args: Any, **_kwargs: Any) -> AdvCPAttackResult:
            raise NotImplementedError(f"AdvCP mode '{mode}' is not available for early fusion.")

        handler = handler_by_mode.get(mode, fail)
        if handler is not fail and memory_data is None:
            raise ValueError(f"AdvCP early {mode} requires current memory data.")

        return handler(
            batch_data,
            model,
            dataset,
            device,
            advcp_config,
            memory_data,
        )

    @classmethod
    def _run_spoof(
        cls,
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: AdvCPConfig,
        memory_data: AdvCPMemoryData,
    ) -> AdvCPAttackResult:
        """
        Spoofing-mode handler.

        Iterates over every present attacker, ray-traces the
        configured target meshes into the attacker's lidar cloud, and
        records the modified clouds back in shared memory. After all
        attackers have written their modifications, runs cooperative
        inference on the modified memory and returns the result.
        """
        advcp_context = AdvCPVisualizationContext(mode="spoofing")
        scenario_data, configured_attacker_ids, present_attacker_ids, _ = AdvCPAttackHelper.resolve_attack_scope(
            advcp_config,
            memory_data,
        )
        if not configured_attacker_ids:
            AdvCPAttackHelper.raise_no_configured_attackers("early")

        density = AdvCPAttackHelper.resolve_density(AdvCPAttackHelper.require_config_value(advcp_config, "density"))
        attacked_memory = copy.deepcopy(memory_data)
        attacked_scenario_data = next(iter(attacked_memory.values()))
        spoof_boxes_ego: list[npt.NDArray] = []
        lidar_poses = AdvCPAttackHelper.build_lidar_pose_map(scenario_data)

        for attacker_id in present_attacker_ids:
            _, _, _, attack_boxes = AdvCPAttackHelper.resolve_spoof_boxes_for_agent(scenario_data, advcp_config, attacker_id)
            if not spoof_boxes_ego:
                _, _, _, spoof_boxes_ego = AdvCPAttackHelper.resolve_spoof_boxes_for_ego(
                    scenario_data,
                    advcp_config,
                    attacker_id,
                )

            attacked_snapshot = AdvCPAttackHelper.resolve_agent_snapshot(attacked_scenario_data, attacker_id)
            spoofed_lidar = AdvCPAttackHelper.require_agent_lidar(attacked_snapshot, attacker_id, "AdvCP early attack")
            spoofing_mask = np.zeros((spoofed_lidar.shape[0],), dtype=np.bool_)
            for attack_box in attack_boxes:
                spoofed_lidar, box_spoofing_mask = cls._apply_sampled_ray_traced_spoof(
                    spoofed_lidar,
                    spoofing_mask,
                    attack_box,
                    lidar_poses,
                    attacker_id,
                    advcp_config,
                    density,
                )
                spoofing_mask = box_spoofing_mask
            attacked_snapshot.update(
                {
                    "lidar_np": spoofed_lidar,
                    "spoofing_mask": spoofing_mask,
                }
            )
            advcp_context.attacker_ids.append(attacker_id)

        if not advcp_context.attacker_ids:
            AdvCPAttackHelper.report_missing_attackers_from_current_batch(
                configured_attacker_ids,
                scenario_data.keys(),
                fusion_name="early",
            )
            return cls._run_fallback_inference(batch_data, model, dataset, advcp_context)

        fake_box_tensor = cls._build_removed_box_tensor(spoof_boxes_ego, dataset, device)
        if fake_box_tensor is not None:
            advcp_context.fake_box_tensor = fake_box_tensor

        return (
            *cls._run_inference_with_attacked_memory(
                batch_data,
                model,
                dataset,
                device,
                memory_data,
                attacked_memory,
                advcp_context.attacker_ids,
                advcp_context,
            ),
            advcp_context,
        )

    @classmethod
    def _run_removal(
        cls,
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: AdvCPConfig,
        memory_data: AdvCPMemoryData,
    ) -> AdvCPAttackResult:
        """
        Removal-mode handler.

        Iterates over every present attacker, replaces lidar points
        falling inside each target box with farther-away points
        (sampled from a wall, adversarial-shape, or ground-plane mesh),
        and records the modified clouds back in shared memory. The
        cooperative model is then run on the modified memory.
        """
        advcp_context = AdvCPVisualizationContext(mode="removal")
        scenario_data, configured_attacker_ids, present_attacker_ids, _ = AdvCPAttackHelper.resolve_attack_scope(
            advcp_config,
            memory_data,
        )
        if not configured_attacker_ids:
            AdvCPAttackHelper.raise_no_configured_attackers("early")

        density = AdvCPAttackHelper.resolve_density(AdvCPAttackHelper.require_config_value(advcp_config, "density"))
        advshape_enabled = cls._resolve_advshape_enabled(advcp_config)
        attacked_memory = copy.deepcopy(memory_data)
        attacked_scenario_data = next(iter(attacked_memory.values()))
        removal_boxes_ego: list[npt.NDArray] = []
        lidar_poses = AdvCPAttackHelper.build_lidar_pose_map(scenario_data)

        for attacker_id in present_attacker_ids:
            _, _, _, removal_boxes = AdvCPAttackHelper.resolve_spoof_boxes_for_agent(scenario_data, advcp_config, attacker_id)
            if not removal_boxes_ego:
                _, _, _, removal_boxes_ego = AdvCPAttackHelper.resolve_spoof_boxes_for_ego(
                    scenario_data,
                    advcp_config,
                    attacker_id,
                )

            attacked_snapshot = AdvCPAttackHelper.resolve_agent_snapshot(attacked_scenario_data, attacker_id)
            removed_lidar = AdvCPAttackHelper.require_agent_lidar(attacked_snapshot, attacker_id, "AdvCP early attack")
            for removal_box in removal_boxes:
                removed_lidar = cls._apply_sampled_ray_traced_remove(
                    removed_lidar,
                    removal_box,
                    lidar_poses,
                    attacker_id,
                    advcp_config,
                    density,
                    advshape_enabled,
                )
            attacked_snapshot["lidar_np"] = removed_lidar
            attacked_snapshot.pop("spoofing_mask", None)
            advcp_context.attacker_ids.append(attacker_id)

        if not advcp_context.attacker_ids:
            AdvCPAttackHelper.report_missing_attackers_from_current_batch(
                configured_attacker_ids,
                scenario_data.keys(),
                fusion_name="early",
            )
            return cls._run_fallback_inference(batch_data, model, dataset, advcp_context)

        removed_box_tensor = cls._build_removed_box_tensor(removal_boxes_ego, dataset, device)
        if removed_box_tensor is not None:
            advcp_context.removed_box_tensor = removed_box_tensor

        return (
            *cls._run_inference_with_attacked_memory(
                batch_data,
                model,
                dataset,
                device,
                memory_data,
                attacked_memory,
                advcp_context.attacker_ids,
                advcp_context,
            ),
            advcp_context,
        )

    @staticmethod
    def _run_fallback_inference(
        batch_data: Any,
        model: Any,
        dataset: Any,
        advcp_context: AdvCPVisualizationContext,
    ) -> AdvCPAttackResult:
        """
        No-attacker fallback: run vanilla early-fusion inference and
        attach the (empty) AdvCP visualization context.
        """
        return (*inference_utils.inference_early_fusion(batch_data, model, dataset), advcp_context)

    @staticmethod
    def _run_inference_with_attacked_memory(
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        memory_data: AdvCPMemoryData,
        attacked_memory: AdvCPMemoryData,
        attacked_attacker_ids: Sequence[AttackerId],
        advcp_context: AdvCPVisualizationContext,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Run cooperative inference with the attacker-modified memory.

        Builds a fresh batch from ``attacked_memory`` (which mutates
        the dataset in place), confirms that the attackers actually
        appear in the resulting batch, and runs early-fusion inference.
        Always restores the original memory in the dataset before
        returning so subsequent calls operate on the unmodified data.

        Parameters
        ----------
        batch_data : Any
            Original collated batch; reused by the fallback path. The
            batch is also rewritten in place when the attack runs.
        model : Any
            Cooperative perception model.
        dataset : Any
            OpenCOOD dataset.
        device : torch.device
            Device for the rebuilt batch.
        memory_data : AdvCPMemoryData
            Original memory (used to restore the dataset on exit).
        attacked_memory : AdvCPMemoryData
            Memory containing the attacker-rewritten lidar clouds.
        attacked_attacker_ids : Sequence of AttackerId
            Attackers expected to appear in the attacked batch.
        advcp_context : AdvCPVisualizationContext
            Context object; cleared if the attackers turned out to be
            absent from the batch (fallback path).

        Returns
        -------
        tuple
            ``(pred_box_tensor, pred_score, gt_box_tensor)``.
        """
        restored_original_memory = False
        try:
            attacked_batch = AdvCPAttackHelper.build_batch_from_memory(dataset, device, attacked_memory)
            attacked_batch_agent_ids = AdvCPAttackHelper.resolve_batch_agent_ids(attacked_batch, fallback_to_top_level=False)
            if attacked_batch_agent_ids and not any(attacker_id in attacked_batch_agent_ids for attacker_id in attacked_attacker_ids):
                AdvCPAttackHelper.report_missing_attackers_from_current_batch(
                    attacked_attacker_ids,
                    attacked_batch_agent_ids,
                    fusion_name="early",
                )
                advcp_context.attacker_ids = []
                advcp_context.fake_box_tensor = None  # noqa: DC05
                advcp_context.removed_box_tensor = None
                dataset.update_database(memory_data=memory_data)
                restored_original_memory = True
                return inference_utils.inference_early_fusion(batch_data, model, dataset)
            batch_data.clear()
            batch_data.update(attacked_batch)
            return inference_utils.inference_early_fusion(batch_data, model, dataset)
        finally:
            if not restored_original_memory:
                dataset.update_database(memory_data=memory_data)

    @staticmethod
    def _resolve_advshape_enabled(advcp_config: AdvCPConfig) -> bool:
        """
        Read the ``advshape`` config flag, validating its type.

        Raises
        ------
        ValueError
            If ``advshape`` is set to a non-bool value.
        """
        advshape_value = advcp_config.get("advshape", False)
        if not isinstance(advshape_value, bool):
            raise ValueError("AdvCP config key 'advshape' must be bool.")
        return advshape_value

    @staticmethod
    def _build_removed_box_tensor(
        removal_boxes_ego: list[BoxLwhBottomCenter],
        dataset: Any,
        device: torch.device,
    ) -> torch.Tensor | None:
        """
        Convert a list of ego-frame removal boxes into a corner tensor.

        Used to populate the AdvCP visualization context with the boxes
        that were targeted on this tick.

        Parameters
        ----------
        removal_boxes_ego : list of BoxLwhBottomCenter
            Removal boxes in the ego lidar frame.
        dataset : Any
            OpenCOOD dataset (provides the ``order`` parameter).
        device : torch.device
            Device for the returned tensor.

        Returns
        -------
        Optional[torch.Tensor]
            ``(N, 8, 3)`` corner tensor, or ``None`` when the list is
            empty.
        """
        if not removal_boxes_ego:
            return None

        # TODO: Move this up when https://github.com/CAVISE/OpenCDA/pull/65 is merged
        from opencood.utils import box_utils

        removed_box_tensors = [AdvCPAttackHelper.convert_box_for_model(removal_box, dataset).to(device) for removal_box in removal_boxes_ego]
        if len(removed_box_tensors) == 0:
            return None

        stacked_removed_boxes = torch.stack(removed_box_tensors, dim=0)
        return box_utils.boxes_to_corners_3d(
            stacked_removed_boxes,
            order=dataset.post_processor.params.get("order", "hwl"),
        )

    @staticmethod
    def _build_lidar_rays(lidar: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray] | None:
        """
        Reconstruct outgoing rays from a lidar point cloud.

        For each point, the corresponding ray originates at the sensor
        origin (assumed to be ``(0, 0, 0)`` in the lidar frame) and
        travels in the direction of the point. Points at the sensor
        origin are dropped. The output ``rays`` array follows the
        Open3D ``RaycastingScene`` convention:
        ``[ox, oy, oz, dx, dy, dz]`` per row.

        Parameters
        ----------
        lidar : npt.NDArray
            ``(N, 4+)`` lidar cloud (xyz + reflectance + ...).

        Returns
        -------
        Optional[tuple of npt.NDArray]
            ``(points_xyz, point_distance, rays)`` or ``None`` when
            the cloud is empty or all points are at the origin.
        """
        if lidar.size == 0:
            return None

        points_xyz = np.asarray(lidar[:, :3], dtype=np.float32)
        point_distance = np.linalg.norm(points_xyz, axis=1)
        valid_mask = point_distance > 1e-6
        if not np.any(valid_mask):
            return None

        # Direction = normalised point vector. The dummy zeros for
        # invalid (origin) points are never used downstream.
        direction = np.zeros_like(points_xyz, dtype=np.float32)
        direction[valid_mask] = points_xyz[valid_mask] / point_distance[valid_mask, None]
        rays = np.hstack([np.zeros((direction.shape[0], 3), dtype=np.float32), direction])
        return points_xyz, point_distance, rays

    @classmethod
    def _calculate_mesh_sampling_weights(
        cls,
        meshes: list[Any],
        lidar_poses: Mapping[AgentId, npt.NDArray],
        attacker_id: AttackerId,
    ) -> npt.NDArray:
        """
        Compute per-mesh sampling weights based on visibility solid
        angle from neighbouring CAV poses.

        Each non-attacker CAV "sees" the meshes from its own pose. A
        mesh that subtends a larger horizontal and vertical angular
        extent from a CAV's viewpoint receives a higher weight,
        reflecting how many lidar returns that CAV would naturally
        register on it.

        The angular extents are computed from the mesh vertices in the
        attacker's lidar frame (so the attack only cares about points
        the attacker itself could plausibly emit).

        Parameters
        ----------
        meshes : list of Open3D meshes
        lidar_poses : Mapping
            ``agent_id -> world-frame lidar pose``.
        attacker_id : AttackerId
            Attacker whose lidar-frame coordinate system the weights
            are expressed in.

        Returns
        -------
        npt.NDArray
            ``(len(meshes),)`` non-negative weights. Falls back to
            uniform weights when no neighbouring CAV provides a
            non-zero angular contribution.
        """
        mesh_weight = np.zeros(len(meshes), dtype=np.float64)
        attacker_pose = lidar_poses[attacker_id]
        for vehicle_id, lidar_pose in lidar_poses.items():
            if vehicle_id == attacker_id:
                continue
            # Project the neighbour's lidar origin into the attacker's
            # frame so we can evaluate angles consistently.
            lidar_offset = cls._world_points_to_sensor(
                np.asarray(lidar_pose[:3], dtype=np.float32)[np.newaxis, :],
                attacker_pose,
            )[0]
            for mesh_index, mesh in enumerate(meshes):
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                if vertices.size == 0:
                    continue
                # Horizontal angular extent (azimuth range across the
                # mesh) and vertical angular extent (elevation range,
                # approximated by tangent for small angles).
                h_angle = np.arctan2(vertices[:, 1] - lidar_offset[1], vertices[:, 0] - lidar_offset[0])
                planar_distance = np.linalg.norm(vertices[:, :2] - lidar_offset[:2], axis=1)
                planar_distance = np.maximum(planar_distance, 1e-6)
                v_angle = (vertices[:, 2] - lidar_offset[2]) / planar_distance
                # 0.005 rad / 0.01 rad approximate the angular
                # resolution of the simulated lidar; the product
                # estimates the number of beams hitting this mesh.
                mesh_weight[mesh_index] += ((h_angle.max() - h_angle.min()) / 0.005) * ((v_angle.max() - v_angle.min()) / 0.01)

        if not np.any(mesh_weight > 0):
            mesh_weight[:] = 1.0
        return mesh_weight

    @staticmethod
    def _sample_intersection_replacements(
        replace_mask_list: list[npt.NDArray],
        replace_data_list: list[npt.NDArray],
        mesh_weight: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Pick a replacement point for each lidar return that intersects
        any mesh.

        For each ray, the candidate meshes are those whose
        ``replace_mask`` has it as True. The chosen mesh is sampled
        from the candidates weighted by ``mesh_weight`` (visibility
        solid angle). Each ray's replacement point is then taken from
        the corresponding mesh's intersection table.

        Parameters
        ----------
        replace_mask_list : list of npt.NDArray
            One boolean array per mesh, ``True`` where that mesh's
            ray-trace returned a finite intersection.
        replace_data_list : list of npt.NDArray
            One ``(N, 3)`` array per mesh of intersection points.
        mesh_weight : npt.NDArray
            ``(num_meshes,)`` non-negative weights.

        Returns
        -------
        tuple of npt.NDArray
            ``(replace_indices, replace_data)`` for use by
            ``_apply_ray_tracing``.
        """
        point_sampling_weight = np.vstack(replace_mask_list).T.astype(np.float64) * mesh_weight
        replace_indices = np.argwhere(np.logical_or.reduce(replace_mask_list)).reshape(-1).astype(np.int32)
        replace_data = []
        stacked_replace_masks = np.vstack(replace_mask_list)
        for point_index in replace_indices:
            weights = point_sampling_weight[point_index]
            total = np.sum(weights)
            if total <= 0:
                # Fallback when all candidates have zero weight: just
                # pick the first mesh that registers an intersection.
                mesh_index = int(np.argmax(stacked_replace_masks[:, point_index]))
            else:
                mesh_index = int(np.random.choice(mesh_weight.shape[0], p=weights / total))
            replace_data.append(replace_data_list[mesh_index][point_index])
        return replace_indices, np.asarray(replace_data, dtype=np.float32)

    @classmethod
    def _build_extra_rays(
        cls,
        rays: npt.NDArray,
        target_box: BoxLwhBottomCenter,
        lidar_poses: Mapping[AgentId, npt.NDArray],
        attacker_id: AttackerId,
        density: int,
        dense_distance: float,
    ) -> list[npt.NDArray]:
        """
        Build auxiliary ray batches that originate near the target box
        rather than at the attacker's sensor.

        Used by the dense modes to ensure the attack still places
        synthetic points even on rays the attacker itself cannot reach.

        - Always emits one ray batch shifted toward the target along
          the line from the attacker to the target.
        - For ``density == 2`` (``"dense_all"``) additionally emits
          one ray batch per neighbouring CAV, shifted toward the
          target along the line from that CAV to the target.

        Parameters
        ----------
        rays : npt.NDArray
            ``(N, 6)`` original ray batch (used as a template for
            directions).
        target_box : BoxLwhBottomCenter
            Target box.
        lidar_poses : Mapping
            ``agent_id -> world-frame lidar pose``.
        attacker_id : AttackerId
        density : int
            Density code (2 enables per-CAV auxiliary rays).
        dense_distance : float
            Distance offset toward the target.

        Returns
        -------
        list of npt.NDArray
            Auxiliary ray batches, possibly empty.
        """
        extra_rays_list: list[npt.NDArray] = []
        target_offset = target_box[:2]
        target_distance = float(np.linalg.norm(target_offset))
        if target_distance > 1e-6:
            lidar_offset = target_offset / target_distance * max(target_distance - dense_distance, 0.0)
            extra_rays = np.array(rays, copy=True)
            extra_rays[:, :2] = lidar_offset
            extra_rays_list.append(extra_rays)

        if density != 2:
            return extra_rays_list

        attacker_pose = lidar_poses[attacker_id]
        for vehicle_id, lidar_pose in lidar_poses.items():
            if vehicle_id == attacker_id:
                continue
            lidar_offset_3d = cls._world_points_to_sensor(
                np.asarray(lidar_pose[:3], dtype=np.float32)[np.newaxis, :],
                attacker_pose,
            )[0]
            lidar_offset = lidar_offset_3d[:2]
            offset_distance = float(np.linalg.norm(target_offset - lidar_offset))
            if offset_distance <= 1e-6:
                continue
            shifted_offset = target_offset + (lidar_offset - target_offset) / offset_distance * dense_distance
            extra_rays = np.array(rays, copy=True)
            extra_rays[:, :2] = shifted_offset
            extra_rays_list.append(extra_rays)
        return extra_rays_list

    @staticmethod
    def _apply_sampled_ray_traced_remove(
        lidar: npt.NDArray,
        removal_box: BoxLwhBottomCenter,
        lidar_poses: Mapping[AgentId, npt.NDArray],
        attacker_id: AttackerId,
        advcp_config: AdvCPConfig,
        density: int,
        advshape_enabled: bool,
    ) -> npt.NDArray:
        """
        Remove lidar points inside the target box (sampled-density mode).

        For ``density != 3`` defers to
        :meth:`_apply_dense_ray_traced_remove`. Otherwise builds
        per-mesh removal candidates (wall meshes or adversarial-shape
        meshes), computes per-mesh visibility weights, and replaces
        each ray's hit with a sampled mesh's intersection point.

        Parameters
        ----------
        lidar : npt.NDArray
            Original ``(N, 4+)`` lidar cloud.
        removal_box : BoxLwhBottomCenter
            Target box.
        lidar_poses : Mapping
        attacker_id : AttackerId
        advcp_config : AdvCPConfig
        density : int
            Density code.
        advshape_enabled : bool
            Whether to prefer the adversarial-shape mesh over wall
            meshes.

        Returns
        -------
        npt.NDArray
            Modified lidar cloud.
        """
        if density != 3:
            return AdvCoperceptionEarlyFusionAttack._apply_dense_ray_traced_remove(
                lidar,
                removal_box,
                lidar_poses,
                attacker_id,
                advcp_config,
                density,
                advshape_enabled,
            )

        ray_data = AdvCoperceptionEarlyFusionAttack._build_lidar_rays(lidar)
        if ray_data is None:
            return np.asarray(lidar, dtype=np.float32)
        _, _, rays = ray_data

        meshes = AdvCoperceptionEarlyFusionAttack._build_sampled_removal_meshes(removal_box, advcp_config, advshape_enabled)
        if len(meshes) == 0:
            return AdvCoperceptionEarlyFusionAttack._apply_box_removal(lidar, removal_box)

        replace_mask_list: list[npt.NDArray] = []
        replace_data_list: list[npt.NDArray] = []
        for mesh in meshes:
            intersect_points = AdvCPCarMeshHelper.ray_intersection([mesh], rays)
            replace_mask_list.append(np.isfinite(intersect_points[:, 0]))
            replace_data_list.append(intersect_points)

        if not replace_mask_list or not np.logical_or.reduce(replace_mask_list).any():
            return np.asarray(lidar, dtype=np.float32)

        mesh_weight = AdvCoperceptionEarlyFusionAttack._calculate_mesh_sampling_weights(meshes, lidar_poses, attacker_id)
        replace_indices, replace_data = AdvCoperceptionEarlyFusionAttack._sample_intersection_replacements(
            replace_mask_list,
            replace_data_list,
            mesh_weight,
        )

        return AdvCoperceptionEarlyFusionAttack._apply_ray_tracing(
            np.asarray(lidar, dtype=np.float32),
            replace_indices=replace_indices,
            replace_data=replace_data,
        )

    @staticmethod
    def _apply_dense_ray_traced_remove(
        lidar: npt.NDArray,
        removal_box: BoxLwhBottomCenter,
        lidar_poses: Mapping[AgentId, npt.NDArray],
        attacker_id: AttackerId,
        advcp_config: AdvCPConfig,
        density: int,
        advshape_enabled: bool,
    ) -> npt.NDArray:
        """
        Remove lidar points inside the target box (dense modes).

        For ``density == 0`` (``"replace"``) replaces each in-box ray
        with the corresponding mesh intersection. For ``density != 0``
        additionally injects auxiliary rays from neighbouring viewpoints
        and appends the resulting points instead of replacing.

        Falls back to a plain box removal when no mesh is available.
        """
        ray_data = AdvCoperceptionEarlyFusionAttack._build_lidar_rays(lidar)
        if ray_data is None:
            return np.asarray(lidar, dtype=np.float32)
        points_xyz, _, rays = ray_data
        remove_indices = AdvCoperceptionEarlyFusionAttack._select_points_in_expanded_box(points_xyz, removal_box)
        if remove_indices.shape[0] == 0:
            return np.asarray(lidar, dtype=np.float32)
        selected_rays = rays[remove_indices]

        meshes = AdvCoperceptionEarlyFusionAttack._build_dense_removal_meshes(
            removal_box,
            lidar,
            advcp_config,
            advshape_enabled,
        )
        if len(meshes) == 0:
            return AdvCoperceptionEarlyFusionAttack._apply_box_removal(lidar, removal_box)

        intersect_points = AdvCPCarMeshHelper.ray_intersection(meshes, selected_rays)
        in_range_mask = np.isfinite(intersect_points[:, 0])

        if density == 0:
            replace_indices = remove_indices[in_range_mask]
            replace_data = intersect_points[in_range_mask].astype(np.float32)
            return AdvCoperceptionEarlyFusionAttack._apply_ray_tracing(
                np.asarray(lidar, dtype=np.float32),
                replace_indices=replace_indices,
                replace_data=replace_data,
            )

        dense_distance = float(AdvCPAttackHelper.require_config_value(advcp_config, "dense_distance"))
        extra_rays_list = AdvCoperceptionEarlyFusionAttack._build_extra_rays(
            rays,
            removal_box,
            lidar_poses,
            attacker_id,
            density,
            dense_distance,
        )
        extra_points_list: list[npt.NDArray] = []
        for extra_rays in extra_rays_list:
            extra_intersections = AdvCPCarMeshHelper.ray_intersection(meshes, extra_rays)
            extra_mask = np.isfinite(extra_intersections[:, 0])
            if not np.any(extra_mask):
                continue
            extra_intersections = extra_intersections[extra_mask].astype(np.float32)
            inside_target_mask = AdvCoperceptionEarlyFusionAttack._compute_points_inside_box_mask(
                np.asarray(extra_intersections[:, :3], dtype=np.float32),
                removal_box,
            )
            if np.any(inside_target_mask):
                extra_points_list.append(extra_intersections[inside_target_mask])

        ignore_indices = remove_indices
        append_data = np.vstack(extra_points_list) if extra_points_list else None
        return AdvCoperceptionEarlyFusionAttack._apply_ray_tracing(
            np.asarray(lidar, dtype=np.float32),
            ignore_indices=ignore_indices,
            append_data=append_data,
        )

    @staticmethod
    def _build_sampled_removal_meshes(
        removal_box: BoxLwhBottomCenter,
        advcp_config: AdvCPConfig,
        advshape_enabled: bool,
    ) -> list[Any]:
        """
        Pick the mesh set for the sampled-density removal mode.

        Returns the adversarial-shape mesh pieces when ``advshape``
        is enabled and the assets are available; otherwise four wall
        meshes around the target box.
        """
        if advshape_enabled:
            adv_shape_meshes = AdvCoperceptionEarlyFusionAttack._build_adv_shape_meshes(removal_box, advcp_config)
            if len(adv_shape_meshes) > 0:
                return adv_shape_meshes
        return AdvCoperceptionEarlyFusionAttack._build_removal_wall_meshes(removal_box)

    @staticmethod
    def _build_dense_removal_meshes(
        removal_box: BoxLwhBottomCenter,
        lidar: npt.NDArray,
        advcp_config: AdvCPConfig,
        advshape_enabled: bool,
    ) -> list[Any]:
        """
        Pick the mesh set for the dense-density removal mode.

        When ``advshape`` is enabled and assets are present, returns
        either the single adversarial-shape mesh or its merged form.
        Otherwise returns a single ground-plane mesh derived from the
        local lidar Z distribution (so removed points are pushed onto
        the road instead of an arbitrary wall).
        """
        if advshape_enabled:
            adv_shape_meshes = AdvCoperceptionEarlyFusionAttack._build_adv_shape_meshes(removal_box, advcp_config)
            if len(adv_shape_meshes) > 0:
                if len(adv_shape_meshes) == 1:
                    return adv_shape_meshes
                return [AdvCPCarMeshHelper.merge_meshes(adv_shape_meshes)]
        return [AdvCoperceptionEarlyFusionAttack._build_ground_plane_mesh(lidar)]

    @staticmethod
    def _build_removal_wall_meshes(removal_box: BoxLwhBottomCenter) -> list[Any]:
        """
        Build four thin wall meshes around the perimeter of a removal
        box.

        Each wall is a thin (1 cm) box offset 0.3 m past one face of
        the target box, so that lidar rays pointing into the target
        zone first encounter a wall rather than disappearing. The
        resulting cloud thus contains points on the wall surface
        instead of inside the target.

        The four walls (in the order returned) line up with the +y,
        +x, -y, -x faces of the target box in its local frame.
        """
        wall_boxes = []
        extend_distance = 0.3

        wall_box_1 = np.array(removal_box, copy=True)
        wall_box_1[0] -= np.sin(wall_box_1[6]) * (wall_box_1[4] / 2.0 + extend_distance)
        wall_box_1[1] += np.cos(wall_box_1[6]) * (wall_box_1[4] / 2.0 + extend_distance)
        wall_box_1[4] = 0.01
        wall_boxes.append(wall_box_1)

        wall_box_2 = np.array(removal_box, copy=True)
        wall_box_2[0] += np.cos(wall_box_2[6]) * (wall_box_2[3] / 2.0 + extend_distance)
        wall_box_2[1] += np.sin(wall_box_2[6]) * (wall_box_2[3] / 2.0 + extend_distance)
        wall_box_2[3] = 0.01
        wall_boxes.append(wall_box_2)

        wall_box_3 = np.array(removal_box, copy=True)
        wall_box_3[0] += np.sin(wall_box_3[6]) * (wall_box_3[4] / 2.0 + extend_distance)
        wall_box_3[1] -= np.cos(wall_box_3[6]) * (wall_box_3[4] / 2.0 + extend_distance)
        wall_box_3[4] = 0.01
        wall_boxes.append(wall_box_3)

        wall_box_4 = np.array(removal_box, copy=True)
        wall_box_4[0] -= np.cos(wall_box_4[6]) * (wall_box_4[3] / 2.0 + extend_distance)
        wall_box_4[1] -= np.sin(wall_box_4[6]) * (wall_box_4[3] / 2.0 + extend_distance)
        wall_box_4[3] = 0.01
        wall_boxes.append(wall_box_4)

        return [
            AdvCPCarMeshHelper.build_box_piece_mesh(
                wall_box,
                (float(wall_box[3]), float(wall_box[4]), float(wall_box[5])),
                (0.0, 0.0, float(wall_box[5]) / 2.0),
            )
            for wall_box in wall_boxes
        ]

    @staticmethod
    def _build_ground_plane_mesh(lidar: npt.NDArray) -> Any:
        """
        Build a large ground-plane mesh for dense-mode removal.

        The plane Z is set 20 cm below the 2nd percentile of the
        lidar's Z distribution, which is a robust estimate of the
        ground level. When the cloud is empty the plane is placed at
        ``z = -2.0`` (a sensible default for typical CARLA setups).
        """
        import open3d as o3d

        if lidar.size == 0:
            ground_z = -2.0
        else:
            ground_z = float(np.percentile(np.asarray(lidar[:, 2], dtype=np.float32), 2.0)) - 0.2
        mesh = o3d.geometry.TriangleMesh.create_box(width=200.0, height=200.0, depth=0.2)
        mesh.translate(np.array([-100.0, -100.0, ground_z], dtype=np.float64))
        return mesh

    @classmethod
    def _build_adv_shape_meshes(
        cls,
        removal_box: BoxLwhBottomCenter,
        advcp_config: AdvCPConfig,
    ) -> list[Any]:
        """
        Build adversarial-shape mesh pieces fitted to a removal box.

        Loads the canonical template (a 4.9 x 2.5 x 2.0 box, optionally
        deformed by a precomputed perturbation), splits it into
        regions, then scales / rotates / translates each piece to
        match ``removal_box``. Logs and returns ``[]`` on any IO or
        validation failure.
        """
        try:
            template_mesh = cls._create_adv_shape_template_mesh(advcp_config)
            mesh_divide = cls._resolve_remove_adv_shape_divide(template_mesh, advcp_config)
            mesh_pieces = [template_mesh.select_by_index(indices) for indices in mesh_divide] if mesh_divide else [template_mesh]
            return cls._apply_adv_shape_transform(mesh_pieces, removal_box)
        except Exception as exc:
            if not cls._REMOVE_ADV_SHAPE_WARNING_EMITTED:
                logger.warning(
                    "AdvCP early removal adv-shape assets are unavailable (%s). Falling back to wall meshes.",
                    exc,
                )
                cls._REMOVE_ADV_SHAPE_WARNING_EMITTED = True
            return []

    @staticmethod
    def _resolve_optional_advshape_path(path_value: Any) -> Path | None:
        if path_value is None:
            return None
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    @classmethod
    def _create_adv_shape_template_mesh(cls, advcp_config: AdvCPConfig) -> Any:
        import open3d as o3d

        # Matches AdvCP canonical adversarial shape template dimensions.
        template_box = np.array([0.0, 0.0, 0.0, 4.9, 2.5, 2.0, 0.0], dtype=np.float32)
        mesh = AdvCPCarMeshHelper.build_box_piece_mesh(
            template_box,
            (float(template_box[3]), float(template_box[4]), float(template_box[5])),
            (0.0, 0.0, float(template_box[5]) / 2.0),
        )
        mesh = mesh.subdivide_midpoint(2)

        perturb_path = cls._resolve_optional_advshape_path(advcp_config.get("remove_adv_shape_perturb_path"))
        if perturb_path is not None and perturb_path.exists():
            perturbation = np.load(perturb_path)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            expected_shape = vertices.shape
            if perturbation.shape == expected_shape:
                vertices = vertices + perturbation.astype(np.float64)
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
            else:
                raise ValueError(f"Unexpected adv-shape perturbation shape {perturbation.shape}; expected {expected_shape}.")
        return mesh

    @classmethod
    def _resolve_remove_adv_shape_divide(
        cls,
        template_mesh: Any,
        advcp_config: AdvCPConfig,
    ) -> list[npt.NDArray]:
        divide_path = cls._resolve_optional_advshape_path(advcp_config.get("remove_adv_shape_divide_path"))
        if divide_path is not None and divide_path.exists():
            with open(divide_path, "rb") as handle:
                raw_divide = pickle.load(handle)
            return [np.asarray(indices, dtype=np.int32) for indices in raw_divide]
        return cls._default_adv_shape_divide(template_mesh)

    @staticmethod
    def _default_adv_shape_divide(template_mesh: Any) -> list[npt.NDArray]:
        vertices = np.asarray(template_mesh.vertices, dtype=np.float64)
        bbox = np.array([4.9, 2.5, 2.0], dtype=np.float64)
        return [
            np.argwhere(vertices[:, 0] > bbox[0] / 2.0 - 0.01).reshape(-1),
            np.argwhere(vertices[:, 0] < -bbox[0] / 2.0 + 0.01).reshape(-1),
            np.argwhere(np.logical_and(vertices[:, 0] >= 0.0, vertices[:, 1] > bbox[1] / 2.0 - 0.01)).reshape(-1),
            np.argwhere(np.logical_and(vertices[:, 0] <= 0.0, vertices[:, 1] > bbox[1] / 2.0 - 0.01)).reshape(-1),
            np.argwhere(np.logical_and(vertices[:, 0] >= 0.0, vertices[:, 1] < -bbox[1] / 2.0 + 0.01)).reshape(-1),
            np.argwhere(np.logical_and(vertices[:, 0] <= 0.0, vertices[:, 1] < -bbox[1] / 2.0 + 0.01)).reshape(-1),
            np.argwhere(np.logical_and(vertices[:, 0] >= 0.0, vertices[:, 2] > bbox[2] - 0.01)).reshape(-1),
            np.argwhere(np.logical_and(vertices[:, 0] <= 0.0, vertices[:, 2] > bbox[2] - 0.01)).reshape(-1),
            np.argwhere(np.logical_and(vertices[:, 0] >= 0.0, vertices[:, 2] < 0.01)).reshape(-1),
            np.argwhere(np.logical_and(vertices[:, 0] <= 0.0, vertices[:, 2] < 0.01)).reshape(-1),
        ]

    @staticmethod
    def _apply_adv_shape_transform(mesh_pieces: list[Any], removal_box: npt.NDArray) -> list[Any]:
        transformed_meshes = []
        reference_size = np.array([4.9, 2.5, 2.0], dtype=np.float64)
        scale = float(np.max((np.asarray(removal_box[3:6], dtype=np.float64) + 0.6) / reference_size))
        rotation = np.array(
            [
                [np.cos(removal_box[6]), -np.sin(removal_box[6]), 0.0],
                [np.sin(removal_box[6]), np.cos(removal_box[6]), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        translation = np.asarray(removal_box[:3], dtype=np.float64)

        for mesh_piece in mesh_pieces:
            transformed_mesh = copy.deepcopy(mesh_piece)
            transformed_mesh.scale(scale, np.zeros(3, dtype=np.float64))
            transformed_mesh.rotate(rotation, np.zeros(3, dtype=np.float64))
            transformed_mesh.translate(translation)
            transformed_meshes.append(transformed_mesh)
        return transformed_meshes

    @staticmethod
    def _apply_ray_tracing(
        lidar: npt.NDArray,
        replace_indices: npt.NDArray | None = None,
        replace_data: npt.NDArray | None = None,
        ignore_indices: npt.NDArray | None = None,
        append_data: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """
        Apply a sequence of replace / delete / append edits to a lidar
        cloud.

        Operations are applied in order. ``replace_data`` only
        overwrites the XYZ columns; the reflectance column is
        preserved. ``append_data`` is given a synthetic reflectance
        of 1.0 per appended point.
        """
        attacked_lidar = np.array(lidar, copy=True)
        if replace_indices is not None and replace_indices.shape[0] > 0 and replace_data is not None:
            attacked_lidar[replace_indices, :3] = replace_data
        if ignore_indices is not None and ignore_indices.shape[0] > 0:
            attacked_lidar = np.delete(attacked_lidar, ignore_indices, axis=0)
        if append_data is not None and append_data.shape[0] > 0:
            attacked_lidar = np.vstack([attacked_lidar, AdvCoperceptionEarlyFusionAttack._append_reflectance_column(append_data)])
        return attacked_lidar.astype(np.float32)

    @staticmethod
    def _apply_box_removal(
        lidar: npt.NDArray,
        removal_box: BoxLwhBottomCenter,
    ) -> npt.NDArray:
        """
        Drop every lidar point falling inside a box.

        Used as a fallback when no removal mesh is available. Returns
        the original cloud unchanged when nothing is inside the box.
        """
        if lidar.size == 0:
            return np.asarray(lidar, dtype=np.float32)

        points_xyz = np.asarray(lidar[:, :3], dtype=np.float32)
        inside_mask = AdvCoperceptionEarlyFusionAttack._compute_points_inside_box_mask(points_xyz, removal_box)
        if not np.any(inside_mask):
            return np.asarray(lidar, dtype=np.float32)
        return np.asarray(lidar[~inside_mask], dtype=np.float32)

    @staticmethod
    def _select_points_in_expanded_box(
        points_xyz: npt.NDArray,
        removal_box: BoxLwhBottomCenter,
    ) -> npt.NDArray:
        """
        Indices of points falling inside a slightly enlarged target box.

        The box is widened by 0.6 m in each horizontal direction and
        extended down by 5 m and up to a 10 m height. This catches
        points near the box surface (which would otherwise be missed
        due to lidar noise) and points just above ground (for ground-
        plane removal).
        """
        expanded_box = np.array(removal_box, copy=True)
        expanded_box[2] -= 5.0
        expanded_box[3] += 0.6
        expanded_box[4] += 0.6
        expanded_box[5] = 10.0
        in_expanded_mask = AdvCoperceptionEarlyFusionAttack._compute_points_inside_box_mask(points_xyz, expanded_box)
        return np.argwhere(in_expanded_mask).reshape(-1).astype(np.int32)

    @staticmethod
    def _compute_points_inside_box_mask(
        points_xyz: npt.NDArray,
        box_lwh_bottom_center: BoxLwhBottomCenter,
    ) -> npt.NDArray:
        """
        Boolean mask for points lying inside a yawed bounding box.

        Performs an inverse rotation around the box yaw, then checks
        the local-frame coordinates against half-extents. The Z check
        treats the box centre as the bottom face (per the
        ``BoxLwhBottomCenter`` convention).

        Parameters
        ----------
        points_xyz : npt.NDArray
            ``(N, 3)`` points.
        box_lwh_bottom_center : BoxLwhBottomCenter
            ``[x, y, z, l, w, h, yaw]``.

        Returns
        -------
        npt.NDArray
            ``(N,)`` boolean mask, ``True`` for points inside the box.
        """
        center_x, center_y, center_z = [float(value) for value in box_lwh_bottom_center[:3]]
        length, width, height = [float(value) for value in box_lwh_bottom_center[3:6]]
        yaw = float(box_lwh_bottom_center[6])

        translated_x = points_xyz[:, 0] - center_x
        translated_y = points_xyz[:, 1] - center_y
        translated_z = points_xyz[:, 2] - center_z

        # Rotate the translated points by -yaw to get coordinates in
        # the box's own local frame, where the inside check reduces to
        # comparing absolute values against half-extents.
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        local_x = cos_yaw * translated_x + sin_yaw * translated_y
        local_y = -sin_yaw * translated_x + cos_yaw * translated_y

        half_length = length / 2.0
        half_width = width / 2.0
        epsilon = 1e-4
        return (
            (np.abs(local_x) <= (half_length + epsilon))
            & (np.abs(local_y) <= (half_width + epsilon))
            & (translated_z >= -epsilon)
            & (translated_z <= (height + epsilon))
        )

    @staticmethod
    def _apply_sampled_ray_traced_spoof(
        lidar: npt.NDArray,
        spoofing_mask: npt.NDArray,
        spoof_box: BoxLwhBottomCenter,
        lidar_poses: Mapping[AgentId, npt.NDArray],
        attacker_id: AttackerId,
        advcp_config: AdvCPConfig,
        density: int,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Inject points onto a spoof box (sampled-density mode).

        For ``density != 3`` defers to
        :meth:`_apply_dense_ray_traced_spoof`. Otherwise builds the
        spoof mesh pieces, ray-traces every existing lidar ray, and
        replaces hit points with the surface intersection sampled per
        viewpoint visibility weights. Marks the replaced points as
        spoofed in ``spoofing_mask``.

        Returns
        -------
        tuple of npt.NDArray
            ``(modified_lidar, updated_spoofing_mask)``.
        """
        if density != 3:
            return AdvCoperceptionEarlyFusionAttack._apply_dense_ray_traced_spoof(
                lidar,
                spoofing_mask,
                spoof_box,
                lidar_poses,
                attacker_id,
                advcp_config,
                density,
            )

        ray_data = AdvCoperceptionEarlyFusionAttack._build_lidar_rays(lidar)
        if ray_data is None:
            return lidar, spoofing_mask
        _, _, rays = ray_data

        meshes = AdvCPCarMeshHelper.build_spoof_meshes(spoof_box, advcp_config)
        replace_mask_list: list[npt.NDArray] = []
        replace_data_list: list[npt.NDArray] = []
        for mesh in meshes:
            intersect_points = AdvCPCarMeshHelper.ray_intersection([mesh], rays)
            replace_mask_list.append(np.isfinite(intersect_points[:, 0]))
            replace_data_list.append(intersect_points)

        if not replace_mask_list or not np.logical_or.reduce(replace_mask_list).any():
            return lidar, spoofing_mask

        mesh_weight = AdvCoperceptionEarlyFusionAttack._calculate_mesh_sampling_weights(meshes, lidar_poses, attacker_id)
        replace_indices, replace_data = AdvCoperceptionEarlyFusionAttack._sample_intersection_replacements(
            replace_mask_list,
            replace_data_list,
            mesh_weight,
        )
        return AdvCoperceptionEarlyFusionAttack._apply_ray_tracing_with_mask(
            np.asarray(lidar, dtype=np.float32),
            spoofing_mask,
            replace_indices=replace_indices,
            replace_data=replace_data,
        )

    @staticmethod
    def _apply_dense_ray_traced_spoof(
        lidar: npt.NDArray,
        spoofing_mask: npt.NDArray,
        spoof_box: BoxLwhBottomCenter,
        lidar_poses: Mapping[AgentId, npt.NDArray],
        attacker_id: AttackerId,
        advcp_config: AdvCPConfig,
        density: int,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Inject points onto a spoof box (dense modes).

        Uses an occlusion test against a single solid collision mesh:
        a ray's hit replaces an existing point only if the existing
        point lies behind the spoof object. For ``density != 0``
        additionally injects auxiliary rays from neighbouring
        viewpoints, taking only the hits that fall inside the target
        box (so injected points are confined to the spoofed object's
        envelope).
        """
        dense_distance = float(AdvCPAttackHelper.require_config_value(advcp_config, "dense_distance"))
        ray_data = AdvCoperceptionEarlyFusionAttack._build_lidar_rays(lidar)
        if ray_data is None:
            return lidar, spoofing_mask
        _, point_distance, rays = ray_data

        collision_mesh = AdvCPCarMeshHelper.build_collision_mesh(spoof_box, advcp_config)
        intersect_points = AdvCPCarMeshHelper.ray_intersection([collision_mesh], rays)
        in_range_mask = np.isfinite(intersect_points[:, 0])
        intersect_distance = np.linalg.norm(intersect_points, axis=1)
        occlusion_mask = point_distance > intersect_distance
        replace_mask = np.logical_and(in_range_mask, occlusion_mask)

        if density == 0:
            replace_indices = np.argwhere(replace_mask).reshape(-1).astype(np.int32)
            replace_data = intersect_points[replace_indices].astype(np.float32)
            return AdvCoperceptionEarlyFusionAttack._apply_ray_tracing_with_mask(
                np.asarray(lidar, dtype=np.float32),
                spoofing_mask,
                replace_indices=replace_indices,
                replace_data=replace_data,
            )

        extra_rays_list = AdvCoperceptionEarlyFusionAttack._build_extra_rays(
            rays,
            spoof_box,
            lidar_poses,
            attacker_id,
            density,
            dense_distance,
        )
        extra_points_list = []
        for extra_rays in extra_rays_list:
            extra_intersections = AdvCPCarMeshHelper.ray_intersection([collision_mesh], extra_rays)
            extra_mask = np.isfinite(extra_intersections[:, 0])
            if np.any(extra_mask):
                extra_points_list.append(extra_intersections[extra_mask].astype(np.float32))

        ignore_indices = np.argwhere(replace_mask).reshape(-1).astype(np.int32)
        append_data = np.vstack(extra_points_list) if extra_points_list else None
        return AdvCoperceptionEarlyFusionAttack._apply_ray_tracing_with_mask(
            np.asarray(lidar, dtype=np.float32),
            spoofing_mask,
            ignore_indices=ignore_indices,
            append_data=append_data,
        )

    @staticmethod
    def _apply_ray_tracing_with_mask(
        lidar: npt.NDArray,
        spoofing_mask: npt.NDArray,
        replace_indices: npt.NDArray | None = None,
        replace_data: npt.NDArray | None = None,
        ignore_indices: npt.NDArray | None = None,
        append_data: npt.NDArray | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Same edit pipeline as :meth:`_apply_ray_tracing` but with a
        parallel boolean mask flagging spoofed points.

        Replaced and appended points are flagged ``True``; deletions
        keep ``mask`` and ``lidar`` length-aligned by deleting the
        corresponding mask rows.

        Returns
        -------
        tuple of npt.NDArray
            ``(spoofed_lidar, updated_mask)``.
        """
        spoofed_lidar = np.array(lidar, copy=True)
        updated_mask = np.array(spoofing_mask, copy=True)
        if replace_indices is not None and replace_indices.shape[0] > 0 and replace_data is not None:
            spoofed_lidar[replace_indices, :3] = replace_data
            updated_mask[replace_indices] = True
        if ignore_indices is not None and ignore_indices.shape[0] > 0:
            spoofed_lidar = np.delete(spoofed_lidar, ignore_indices, axis=0)
            updated_mask = np.delete(updated_mask, ignore_indices, axis=0)
        if append_data is not None and append_data.shape[0] > 0:
            spoofed_lidar = np.vstack([spoofed_lidar, AdvCoperceptionEarlyFusionAttack._append_reflectance_column(append_data)])
            updated_mask = np.hstack([updated_mask, np.ones((append_data.shape[0],), dtype=np.bool_)])
        return spoofed_lidar.astype(np.float32), updated_mask.astype(np.bool_)

    @staticmethod
    def _append_reflectance_column(points_xyz: npt.NDArray) -> npt.NDArray:
        """
        Append a constant reflectance of 1.0 to a XYZ point array.

        OpenCOOD lidar clouds use ``(x, y, z, reflectance)`` rows;
        synthetic injected points must include the fourth column to
        stay shape-compatible.
        """
        reflectance = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
        return np.hstack([points_xyz.astype(np.float32), reflectance])

    @staticmethod
    def _world_points_to_sensor(points_world: npt.NDArray, sensor_pose: npt.NDArray) -> npt.NDArray:
        """
        Project world-frame points into a sensor frame.

        Inverse of ``x_to_world(sensor_pose)`` applied to the points
        in homogeneous coordinates.

        Parameters
        ----------
        points_world : npt.NDArray
            ``(N, 3)`` world-frame points.
        sensor_pose : npt.NDArray
            Sensor pose ``[x, y, z, roll, yaw, pitch]``.

        Returns
        -------
        npt.NDArray
            ``(N, 3)`` sensor-frame points.
        """
        sensor_matrix = x_to_world(sensor_pose.tolist())
        world_to_sensor = np.linalg.inv(sensor_matrix)
        homogeneous_points = np.hstack([points_world.astype(np.float32), np.ones((points_world.shape[0], 1), dtype=np.float32)])
        sensor_points = (world_to_sensor @ homogeneous_points.T).T
        return sensor_points[:, :3].astype(np.float32)
