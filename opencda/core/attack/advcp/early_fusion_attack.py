from __future__ import annotations

import copy
import logging
from typing import Any, Mapping

import numpy as np
import torch
from opencood.tools import inference_utils, train_utils
from opencood.utils.transformation_utils import x_to_world

from opencda.core.attack.advcp.attack_helper import AdvCPAttackHelper, AdvCPCarMeshHelper
from opencda.core.attack.advcp.types import AdvCPAttackResult, AdvCPVisualizationContext

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.early_fusion_attack")


class AdvCoperceptionEarlyFusionAttack:
    DENSE_DISTANCE = 10.0
    DENSITY_ALIASES = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        "replace": 0,
        "dense_a": 1,
        "denseall": 2,
        "dense_all": 2,
        "sampled": 3,
    }

    @staticmethod
    def run(
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: dict[str, Any],
        memory_data: dict[Any, Any] | None = None,
    ) -> AdvCPAttackResult:
        mode = advcp_config.get("mode", "spoof")
        advcp_context: AdvCPVisualizationContext = {
            "attacker_ids": [],
            "fake_box_tensor": None,
            "mode": mode,
        }

        if mode == "remove":
            raise NotImplementedError("AdvCP early-fusion removal is not available yet.")
        if mode != "spoof":
            raise NotImplementedError(f"AdvCP mode '{mode}' is not available for early fusion.")
        if memory_data is None:
            raise ValueError("AdvCP early spoofing requires current memory data.")

        scenario_data = next(iter(memory_data.values()))
        attacker_id = advcp_config.get("attacker_id")
        if not attacker_id:
            logger.warning("AdvCP early attack will not be applied on this tick because no valid attacker_id is configured.")
            return (*inference_utils.inference_early_fusion(batch_data, model, dataset), advcp_context)
        if attacker_id not in scenario_data:
            logger.warning(
                "AdvCP early attack will not be applied on this tick because attacker '%s' is not present in the current scenario data. "
                "Continuing with normal cooperative perception inference.",
                attacker_id,
            )
            return (*inference_utils.inference_early_fusion(batch_data, model, dataset), advcp_context)

        _, _, _, attack_boxes = AdvCPAttackHelper.resolve_spoof_boxes_for_agent(scenario_data, advcp_config, attacker_id)
        density = AdvCoperceptionEarlyFusionAttack._resolve_density(advcp_config.get("density", 3))
        advcp_context["attacker_ids"] = [attacker_id]

        attacked_memory = copy.deepcopy(memory_data)
        attacked_scenario_data = next(iter(attacked_memory.values()))
        attacked_agent_data = attacked_scenario_data[attacker_id]
        attacked_timestamp = next(key for key in attacked_agent_data.keys() if key != "ego")
        attacked_snapshot = attacked_agent_data[attacked_timestamp]
        attacker_lidar = attacked_snapshot.get("lidar_np")
        if attacker_lidar is None:
            raise ValueError(f"AdvCP early attack requires in-memory lidar_np for attacker '{attacker_id}'.")

        lidar_poses = {
            agent_id: np.asarray(AdvCPAttackHelper.load_agent_state(scenario_data, agent_id)["lidar_pose"], dtype=np.float32)
            for agent_id in scenario_data
        }

        spoofed_lidar = np.asarray(attacker_lidar, dtype=np.float32)
        spoofing_mask = np.zeros((spoofed_lidar.shape[0],), dtype=np.bool_)
        for attack_box in attack_boxes:
            spoofed_lidar, box_spoofing_mask = AdvCoperceptionEarlyFusionAttack._apply_sampled_ray_traced_spoof(
                spoofed_lidar,
                spoofing_mask,
                attack_box,
                lidar_poses,
                attacker_id,
                advcp_config,
                density,
            )
            spoofing_mask = box_spoofing_mask
        attacked_snapshot["lidar_np"] = spoofed_lidar
        attacked_snapshot["spoofing_mask"] = spoofing_mask

        dataset.update_database(memory_data=attacked_memory)
        try:
            attacked_batch = dataset.collate_batch_test([dataset[0]])
            attacked_batch = train_utils.to_device(attacked_batch, device)
            batch_data.clear()
            batch_data.update(attacked_batch)
            pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(batch_data, model, dataset)
        finally:
            dataset.update_database(memory_data=memory_data)

        return pred_box_tensor, pred_score, gt_box_tensor, advcp_context

    @staticmethod
    def _apply_sampled_ray_traced_spoof(
        lidar: np.ndarray,
        spoofing_mask: np.ndarray,
        spoof_box: np.ndarray,
        lidar_poses: Mapping[str, np.ndarray],
        attacker_id: str,
        advcp_config: Mapping[str, Any],
        density: int,
    ) -> tuple[np.ndarray, np.ndarray]:
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

        if lidar.size == 0:
            return lidar, spoofing_mask

        point_xyz = np.asarray(lidar[:, :3], dtype=np.float32)
        distance = np.linalg.norm(point_xyz, axis=1)
        valid_mask = distance > 1e-6
        if not np.any(valid_mask):
            return lidar, spoofing_mask

        direction = np.zeros_like(point_xyz, dtype=np.float32)
        direction[valid_mask] = point_xyz[valid_mask] / distance[valid_mask, None]
        rays = np.hstack([np.zeros((direction.shape[0], 3), dtype=np.float32), direction])

        meshes = AdvCPCarMeshHelper.build_spoof_meshes(spoof_box, advcp_config)
        replace_mask_list: list[np.ndarray] = []
        replace_data_list: list[np.ndarray] = []
        for mesh in meshes:
            intersect_points = AdvCPCarMeshHelper.ray_intersection([mesh], rays)
            replace_mask_list.append(np.isfinite(intersect_points[:, 0]))
            replace_data_list.append(intersect_points)

        if not replace_mask_list or not np.logical_or.reduce(replace_mask_list).any():
            return lidar, spoofing_mask

        mesh_weight = np.zeros(len(meshes), dtype=np.float64)
        attacker_pose = lidar_poses[attacker_id]
        for vehicle_id, lidar_pose in lidar_poses.items():
            if vehicle_id == attacker_id:
                continue
            lidar_offset = AdvCoperceptionEarlyFusionAttack._world_points_to_sensor(
                np.asarray(lidar_pose[:3], dtype=np.float32)[np.newaxis, :],
                attacker_pose,
            )[0]
            for mesh_index, mesh in enumerate(meshes):
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                if vertices.size == 0:
                    continue
                h_angle = np.arctan2(vertices[:, 1] - lidar_offset[1], vertices[:, 0] - lidar_offset[0])
                planar_distance = np.linalg.norm(vertices[:, :2] - lidar_offset[:2], axis=1)
                planar_distance = np.maximum(planar_distance, 1e-6)
                v_angle = (vertices[:, 2] - lidar_offset[2]) / planar_distance
                mesh_weight[mesh_index] += ((h_angle.max() - h_angle.min()) / 0.005) * ((v_angle.max() - v_angle.min()) / 0.01)

        if not np.any(mesh_weight > 0):
            mesh_weight[:] = 1.0

        point_sampling_weight = np.vstack(replace_mask_list).T.astype(np.float64) * mesh_weight
        replace_indices = np.argwhere(np.logical_or.reduce(replace_mask_list)).reshape(-1).astype(np.int32)

        np.random.seed(0)
        replace_data = []
        for point_index in replace_indices:
            weights = point_sampling_weight[point_index]
            total = np.sum(weights)
            if total <= 0:
                mesh_index = int(np.argmax(np.vstack(replace_mask_list)[:, point_index]))
            else:
                mesh_index = int(np.random.choice(mesh_weight.shape[0], p=weights / total))
            replace_data.append(replace_data_list[mesh_index][point_index])

        replace_data_array = np.asarray(replace_data, dtype=np.float32)
        return AdvCoperceptionEarlyFusionAttack._apply_ray_tracing_with_mask(
            np.asarray(lidar, dtype=np.float32),
            spoofing_mask,
            replace_indices=replace_indices,
            replace_data=replace_data_array,
        )

    @staticmethod
    def _apply_dense_ray_traced_spoof(
        lidar: np.ndarray,
        spoofing_mask: np.ndarray,
        spoof_box: np.ndarray,
        lidar_poses: Mapping[str, np.ndarray],
        attacker_id: str,
        advcp_config: Mapping[str, Any],
        density: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if lidar.size == 0:
            return lidar, spoofing_mask

        point_xyz = np.asarray(lidar[:, :3], dtype=np.float32)
        point_distance = np.linalg.norm(point_xyz, axis=1)
        valid_mask = point_distance > 1e-6
        if not np.any(valid_mask):
            return lidar, spoofing_mask

        direction = np.zeros_like(point_xyz, dtype=np.float32)
        direction[valid_mask] = point_xyz[valid_mask] / point_distance[valid_mask, None]
        rays = np.hstack([np.zeros((direction.shape[0], 3), dtype=np.float32), direction])

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

        extra_rays_list = []
        target_offset = spoof_box[:2]
        target_distance = float(np.linalg.norm(target_offset))
        if target_distance > 1e-6:
            lidar_offset = target_offset / target_distance * max(target_distance - AdvCoperceptionEarlyFusionAttack.DENSE_DISTANCE, 0.0)
            extra_rays = np.array(rays, copy=True)
            extra_rays[:, :2] = lidar_offset
            extra_rays_list.append(extra_rays)

        if density == 2:
            attacker_pose = lidar_poses[attacker_id]
            for vehicle_id, lidar_pose in lidar_poses.items():
                if vehicle_id == attacker_id:
                    continue
                lidar_offset_3d = AdvCoperceptionEarlyFusionAttack._world_points_to_sensor(
                    np.asarray(lidar_pose[:3], dtype=np.float32)[np.newaxis, :],
                    attacker_pose,
                )[0]
                lidar_offset = lidar_offset_3d[:2]
                offset_distance = float(np.linalg.norm(target_offset - lidar_offset))
                if offset_distance <= 1e-6:
                    continue
                shifted_offset = target_offset + (lidar_offset - target_offset) / offset_distance * AdvCoperceptionEarlyFusionAttack.DENSE_DISTANCE
                extra_rays = np.array(rays, copy=True)
                extra_rays[:, :2] = shifted_offset
                extra_rays_list.append(extra_rays)

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
        lidar: np.ndarray,
        spoofing_mask: np.ndarray,
        replace_indices: np.ndarray | None = None,
        replace_data: np.ndarray | None = None,
        ignore_indices: np.ndarray | None = None,
        append_data: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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
    def _append_reflectance_column(points_xyz: np.ndarray) -> np.ndarray:
        reflectance = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
        return np.hstack([points_xyz.astype(np.float32), reflectance])

    @classmethod
    def _resolve_density(cls, density_value: Any) -> int:
        normalized_value = density_value
        if isinstance(density_value, str):
            normalized_value = density_value.strip().lower()
        if normalized_value not in cls.DENSITY_ALIASES:
            raise ValueError(
                f"Unsupported AdvCP early spoofing density '{density_value}'. Supported values are 0, 1, 2, 3, "
                "'replace', 'dense_a', 'dense_all', and 'sampled'."
            )
        return cls.DENSITY_ALIASES[normalized_value]

    @staticmethod
    def _world_points_to_sensor(points_world: np.ndarray, sensor_pose: np.ndarray) -> np.ndarray:
        sensor_matrix = x_to_world(sensor_pose.tolist())
        world_to_sensor = np.linalg.inv(sensor_matrix)
        homogeneous_points = np.hstack([points_world.astype(np.float32), np.ones((points_world.shape[0], 1), dtype=np.float32)])
        sensor_points = (world_to_sensor @ homogeneous_points.T).T
        return sensor_points[:, :3].astype(np.float32)
