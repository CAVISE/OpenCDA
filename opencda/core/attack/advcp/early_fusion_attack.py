from __future__ import annotations

import copy
import logging
from pathlib import Path
import pickle
from typing import Any, Mapping, TypedDict

import numpy as np
import torch

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.early_fusion_attack")


class EarlyAdvCPVisualizationContext(TypedDict):
    attacker_ids: list[str]
    fake_box_tensor: Any | None  # noqa: DC01
    mode: str | None


class AdvCoperceptionEarlyFusionAttack:
    DEFAULT_MESH_FILENAME = "car_0200.ply"
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
    MODEL_3D_EXAMPLES = {
        "car_000000": np.array([0.0, 0.0, 0.0, 5.00, 2.00, 1.75, 0.0], dtype=np.float32),
        "car_0200": np.array([0.0, 0.0, 0.0, 4.30, 1.91, 1.26, 0.0], dtype=np.float32),
    }
    _MESH_DIVIDE_CACHE: dict[Path, Any] = {}
    _REAL_MESH_WARNING_EMITTED = False

    @staticmethod
    def run(
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: dict[str, Any],
        memory_data: dict[Any, Any] | None = None,
    ) -> tuple[Any, Any, Any, EarlyAdvCPVisualizationContext]:
        from opencood.tools import inference_utils, train_utils
        from opencda.core.attack.advcp.adv_coperception_model_manager import AdvCoperceptionModelManager

        mode = advcp_config.get("mode", "spoof")
        advcp_context: EarlyAdvCPVisualizationContext = {
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

        ego_agent_id = next((agent_id for agent_id, agent_data in scenario_data.items() if agent_data.get("ego")), None)
        if ego_agent_id is None:
            raise ValueError("Unable to resolve ego agent for AdvCP early attack.")

        ego_state = AdvCoperceptionModelManager._load_agent_state(scenario_data, ego_agent_id)
        attacker_state = AdvCoperceptionModelManager._load_agent_state(scenario_data, attacker_id)

        box_specs = advcp_config.get("boxes", [])
        if not isinstance(box_specs, list) or len(box_specs) == 0:
            raise ValueError("AdvCP config must define a non-empty boxes list.")

        attack_boxes = [
            AdvCoperceptionModelManager._resolve_box_spec(spec, index, advcp_config, ego_state, attacker_state)
            for index, spec in enumerate(box_specs)
        ]
        density = AdvCoperceptionEarlyFusionAttack._resolve_density(advcp_config.get("density", 3))
        advcp_context["attacker_ids"] = [attacker_id]

        attacked_memory = copy.deepcopy(memory_data)
        attacked_snapshot = AdvCoperceptionEarlyFusionAttack._get_agent_snapshot(attacked_memory, attacker_id)
        attacker_lidar = attacked_snapshot.get("lidar_np")
        if attacker_lidar is None:
            raise ValueError(f"AdvCP early attack requires in-memory lidar_np for attacker '{attacker_id}'.")

        lidar_poses = {
            agent_id: np.asarray(AdvCoperceptionModelManager._load_agent_state(scenario_data, agent_id)["lidar_pose"], dtype=np.float32)
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
    def _get_agent_snapshot(memory_data: dict[Any, Any], agent_id: str) -> dict[str, Any]:
        scenario_data = next(iter(memory_data.values()))
        agent_data = scenario_data[agent_id]
        timestamp = next(key for key in agent_data.keys() if key != "ego")
        return agent_data[timestamp]

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

        meshes = AdvCoperceptionEarlyFusionAttack._build_spoof_meshes(spoof_box, advcp_config)
        replace_mask_list: list[np.ndarray] = []
        replace_data_list: list[np.ndarray] = []
        for mesh in meshes:
            intersect_points = AdvCoperceptionEarlyFusionAttack._ray_intersection([mesh], rays)
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

        collision_mesh = AdvCoperceptionEarlyFusionAttack._build_collision_mesh(spoof_box, advcp_config)
        intersect_points = AdvCoperceptionEarlyFusionAttack._ray_intersection([collision_mesh], rays)
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
            extra_intersections = AdvCoperceptionEarlyFusionAttack._ray_intersection([collision_mesh], extra_rays)
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

    @staticmethod
    def _build_spoof_mesh_pieces(spoof_box: np.ndarray) -> list[Any]:
        thickness = 0.05
        length = float(spoof_box[3])
        width = float(spoof_box[4])
        height = float(spoof_box[5])

        pieces = [
            ((thickness, width, height), (length / 2.0 - thickness / 2.0, 0.0, height / 2.0)),
            ((thickness, width, height), (-length / 2.0 + thickness / 2.0, 0.0, height / 2.0)),
            ((length, thickness, height), (0.0, width / 2.0 - thickness / 2.0, height / 2.0)),
            ((length, thickness, height), (0.0, -width / 2.0 + thickness / 2.0, height / 2.0)),
        ]
        return [AdvCoperceptionEarlyFusionAttack._build_box_piece_mesh(spoof_box, extents, center) for extents, center in pieces]

    @classmethod
    def _build_spoof_meshes(cls, spoof_box: np.ndarray, advcp_config: Mapping[str, Any]) -> list[Any]:
        model_meshes = cls._build_real_car_mesh_pieces(spoof_box, advcp_config)
        if model_meshes is not None:
            return model_meshes
        if not cls._REAL_MESH_WARNING_EMITTED:
            logger.warning("AdvCP early spoofing 3D model assets were not found. Falling back to bbox-shell ray tracing instead of real car mesh.")
            cls._REAL_MESH_WARNING_EMITTED = True
        return cls._build_spoof_mesh_pieces(spoof_box)

    @classmethod
    def _build_collision_mesh(cls, spoof_box: np.ndarray, advcp_config: Mapping[str, Any]):
        model_meshes = cls._build_real_car_mesh_pieces(spoof_box, advcp_config)
        if model_meshes is not None:
            return model_meshes[0] if len(model_meshes) == 1 else cls._merge_meshes(model_meshes)
        return cls._build_box_piece_mesh(
            spoof_box,
            (float(spoof_box[3]), float(spoof_box[4]), float(spoof_box[5])),
            (0.0, 0.0, float(spoof_box[5]) / 2.0),
        )

    @classmethod
    def _build_real_car_mesh_pieces(cls, spoof_box: np.ndarray, advcp_config: Mapping[str, Any]) -> list[Any] | None:
        import open3d as o3d

        mesh_path, mesh_divide_path = cls._resolve_mesh_paths(advcp_config)
        if mesh_path is None or not mesh_path.exists():
            return None

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh.is_empty():
            return None

        mesh_name = cls._mesh_name_from_path(mesh_path)
        if mesh_divide_path is not None and mesh_divide_path.exists():
            if mesh_divide_path not in cls._MESH_DIVIDE_CACHE:
                with open(mesh_divide_path, "rb") as handle:
                    cls._MESH_DIVIDE_CACHE[mesh_divide_path] = pickle.load(handle)
            meshes = [mesh.select_by_index(vertex_indices) for vertex_indices in cls._MESH_DIVIDE_CACHE[mesh_divide_path]]
        else:
            meshes = [mesh]
        return cls._post_process_model_meshes(meshes, spoof_box, mesh_name)

    @classmethod
    def _post_process_model_meshes(cls, meshes: list[Any], spoof_box: np.ndarray, mesh_name: str) -> list[Any]:
        processed_meshes = []
        model_bbox = cls.MODEL_3D_EXAMPLES.get(mesh_name, cls.MODEL_3D_EXAMPLES["car_0200"])
        scale = float(np.min(spoof_box[3:6] / model_bbox[3:6]))
        rotation = np.array(
            [
                [np.cos(spoof_box[6]), -np.sin(spoof_box[6]), 0.0],
                [np.sin(spoof_box[6]), np.cos(spoof_box[6]), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        for mesh in meshes:
            processed_mesh = copy.deepcopy(mesh)
            processed_mesh.scale(scale, np.zeros(3, dtype=np.float64))
            processed_mesh.rotate(rotation, np.zeros(3, dtype=np.float64))
            processed_mesh.translate(np.asarray(spoof_box[:3], dtype=np.float64))
            processed_meshes.append(processed_mesh)

        return processed_meshes

    @staticmethod
    def _mesh_name_from_path(mesh_path: Path | None) -> str:
        if mesh_path is None:
            return "car_0200"
        return mesh_path.stem

    @classmethod
    def _resolve_mesh_paths(cls, advcp_config: Mapping[str, Any]) -> tuple[Path | None, Path | None]:
        local_model_root = Path(__file__).resolve().parent / "3d_models"
        mesh_path_value = advcp_config.get("model_path")
        if mesh_path_value:
            mesh_path = Path(str(mesh_path_value)).expanduser()
        else:
            mesh_path = local_model_root / cls.DEFAULT_MESH_FILENAME

        if not mesh_path.is_absolute():
            mesh_path = (Path.cwd() / mesh_path).resolve()

        mesh_divide_path_value = advcp_config.get("mesh_divide_path")
        if mesh_divide_path_value:
            mesh_divide_path = Path(str(mesh_divide_path_value)).expanduser()
            if not mesh_divide_path.is_absolute():
                mesh_divide_path = (Path.cwd() / mesh_divide_path).resolve()
        else:
            mesh_divide_path = local_model_root / "spoof" / "mesh_divide.pkl"
            if not mesh_divide_path.exists():
                mesh_divide_path = mesh_path.parent / "spoof" / "mesh_divide.pkl"

        return mesh_path, mesh_divide_path

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
    def _build_box_piece_mesh(spoof_box: np.ndarray, extents: tuple[float, float, float], center_local: tuple[float, float, float]):
        import open3d as o3d

        x, y, z, _, _, _, yaw = spoof_box.tolist()
        extent_x, extent_y, extent_z = extents
        mesh = o3d.geometry.TriangleMesh.create_box(width=extent_x, height=extent_y, depth=extent_z)
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
    def _merge_meshes(meshes: list[Any]):
        merged_mesh = copy.deepcopy(meshes[0])
        for mesh in meshes[1:]:
            merged_mesh += mesh
        return merged_mesh

    @staticmethod
    def _ray_intersection(meshes: list[Any], rays: np.ndarray) -> np.ndarray:
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
            if ans["t_hit"][ray_index] > 10000:
                continue
            mesh = mesh_id_map[ans["geometry_ids"][ray_index]]
            triangle_vertices = np.asarray(mesh.triangles)[ans["primitive_ids"][ray_index]]
            vertices = np.asarray(mesh.vertices)[triangle_vertices]
            uv = ans["primitive_uvs"][ray_index]
            intersections[ray_index] = (1.0 - np.sum(uv)) * vertices[0] + uv[0] * vertices[1] + uv[1] * vertices[2]

        return intersections

    @staticmethod
    def _world_points_to_sensor(points_world: np.ndarray, sensor_pose: np.ndarray) -> np.ndarray:
        from opencood.utils.transformation_utils import x_to_world

        sensor_matrix = x_to_world(sensor_pose.tolist())
        world_to_sensor = np.linalg.inv(sensor_matrix)
        homogeneous_points = np.hstack([points_world.astype(np.float32), np.ones((points_world.shape[0], 1), dtype=np.float32)])
        sensor_points = (world_to_sensor @ homogeneous_points.T).T
        return sensor_points[:, :3].astype(np.float32)
