from __future__ import annotations

import copy
from collections import OrderedDict
import logging
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.transformation_utils import x_to_world

from opencda.core.attack.advcp.types import AdvCPAgentState, AdvCPBoxSpec, AdvCPConfig
from opencda.core.common.coperception_data_processor import LiveMemorySnapshot

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.advcp_manager")


class AdvCPAttackHelper:
    @staticmethod
    def require_config_value(config: Mapping[str, Any], key: str, config_name: str = "AdvCP config") -> Any:
        value = config.get(key)
        if value is None:
            raise ValueError(f"Unexpected None in {config_name} for '{key}'.")
        return value

    @staticmethod
    def resolve_ego_agent_id(scenario_data: Mapping[str, Any]) -> str:
        ego_agent_id = next((agent_id for agent_id, agent_data in scenario_data.items() if agent_data.get("ego")), None)
        if ego_agent_id is None:
            raise ValueError("Unable to resolve ego agent for AdvCP attack.")
        return ego_agent_id

    @staticmethod
    def load_agent_state(scenario_data: Mapping[str, Any], agent_id: str) -> AdvCPAgentState:
        agent_data = scenario_data[agent_id]
        timestamp = next(key for key in agent_data.keys() if key != "ego")
        snapshot = agent_data[timestamp]
        yaml_path = snapshot.get("yaml")
        if (params := snapshot.get("params")) is None:
            if yaml_path is None:
                raise ValueError(f"AdvCP agent state for '{agent_id}' does not define either 'params' or 'yaml'.")

            params = load_yaml(yaml_path)

        lidar_pose = params["lidar_pose"]
        return {
            "agent_id": agent_id,
            "timestamp": timestamp,
            "yaml_path": yaml_path,
            "params": params,
            "lidar_pose": lidar_pose,
            "ego_pose": params.get("true_ego_pos", lidar_pose),
        }

    @staticmethod
    def _deduplicate_attacker_ids(attacker_ids: Sequence[str]) -> list[str]:
        ordered_ids: list[str] = []
        seen_ids: set[str] = set()
        for attacker_id in attacker_ids:
            normalized_attacker_id = str(attacker_id).strip()
            if not normalized_attacker_id or normalized_attacker_id in seen_ids:
                continue
            seen_ids.add(normalized_attacker_id)
            ordered_ids.append(normalized_attacker_id)
        return ordered_ids

    @classmethod
    def resolve_configured_attacker_ids(cls, advcp_config: AdvCPConfig) -> list[str]:
        attacker_ids_raw = cls.require_config_value(advcp_config, "attacker_ids")
        if not isinstance(attacker_ids_raw, Sequence) or isinstance(attacker_ids_raw, (str, bytes)):
            raise ValueError("AdvCP config key 'attacker_ids' must be a sequence of agent ids.")
        return cls._deduplicate_attacker_ids([str(attacker_id) for attacker_id in attacker_ids_raw if attacker_id is not None])

    @classmethod
    def resolve_spoof_boxes_by_attacker(
        cls,
        advcp_config: AdvCPConfig,
        memory_data: OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]] | None,
    ) -> tuple[list[str], dict[str, list[np.ndarray]]]:
        if memory_data is None:
            raise ValueError("AdvCP late spoofing requires current memory data.")

        mode = cls.require_config_value(advcp_config, "mode")
        match mode:
            case "spoof":
                pass
            case _:
                raise NotImplementedError(f"AdvCP mode '{mode}' is not available yet.")

        scenario_data = next(iter(memory_data.values()))
        ego_agent_id = cls.resolve_ego_agent_id(scenario_data)

        configured_attacker_ids = cls.resolve_configured_attacker_ids(advcp_config)
        resolved_attacker_ids: list[str] = []
        attack_boxes_by_batch_attacker: dict[str, list[np.ndarray]] = {}

        for attacker_id in configured_attacker_ids:
            if attacker_id not in scenario_data:
                logger.warning(
                    "AdvCP attack will not be applied on this tick because attacker '%s' is not present in the current scenario data. "
                    "Continuing with normal cooperative perception inference.",
                    attacker_id,
                )
                continue

            _, _, _, attack_boxes = cls.resolve_spoof_boxes_for_agent(scenario_data, advcp_config, attacker_id)
            batch_attacker_id = "ego" if attacker_id == ego_agent_id else attacker_id
            attack_boxes_by_batch_attacker.setdefault(batch_attacker_id, []).extend(attack_boxes)
            resolved_attacker_ids.append(attacker_id)

        return resolved_attacker_ids, attack_boxes_by_batch_attacker

    @classmethod
    def resolve_spoof_boxes(
        cls,
        advcp_config: AdvCPConfig,
        memory_data: OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]] | None,
    ) -> tuple[str | None, list[np.ndarray]]:
        _, attack_boxes_by_batch_attacker = cls.resolve_spoof_boxes_by_attacker(advcp_config, memory_data)
        if not attack_boxes_by_batch_attacker:
            return None, []
        attacker_id, attack_boxes = next(iter(attack_boxes_by_batch_attacker.items()))
        return attacker_id, attack_boxes

    @classmethod
    def resolve_spoof_boxes_for_agent(
        cls,
        scenario_data: Mapping[str, Any],
        advcp_config: AdvCPConfig,
        attacker_id: str,
    ) -> tuple[str, AdvCPAgentState, AdvCPAgentState, list[np.ndarray]]:
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
        scenario_data: Mapping[str, Any],
        advcp_config: AdvCPConfig,
        attacker_id: str,
    ) -> tuple[str, AdvCPAgentState, AdvCPAgentState, list[np.ndarray]]:
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
    ) -> np.ndarray:
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
    def compose_relative_pose(reference_pose: np.ndarray | Sequence[float], relative_pose: np.ndarray) -> np.ndarray:
        reference_pose_array = np.asarray(reference_pose, dtype=np.float32)
        reference_matrix = x_to_world(reference_pose_array.tolist())
        relative_point = np.array([relative_pose[0], relative_pose[1], relative_pose[2], 1.0], dtype=np.float32)
        world_point = reference_matrix @ relative_point

        world_pose = np.zeros(6, dtype=np.float32)
        world_pose[:3] = world_point[:3]
        world_pose[3:] = reference_pose_array[3:] + relative_pose[3:]
        return world_pose

    @staticmethod
    def world_box_to_sensor_box(world_pose: np.ndarray, size: np.ndarray, sensor_pose: Sequence[float]) -> np.ndarray:
        sensor_matrix = x_to_world(list(sensor_pose))
        world_to_sensor = np.linalg.inv(sensor_matrix)
        world_point = np.array([world_pose[0], world_pose[1], world_pose[2], 1.0], dtype=np.float32)
        sensor_point = world_to_sensor @ world_point

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
    def convert_box_for_model(box_lwh_bottom_center: np.ndarray, dataset: Any) -> torch.Tensor:
        model_box = np.copy(box_lwh_bottom_center)
        order = dataset.post_processor.params.get("order", "hwl")

        if order == "hwl":
            model_box[3:6] = model_box[[5, 4, 3]]
            model_box[2] += 0.5 * model_box[3]
        elif order == "lwh":
            model_box[2] += 0.5 * model_box[5]
        else:
            raise NotImplementedError(f"Unsupported box order for AdvCP spoofing: {order}")

        return torch.from_numpy(model_box).type(torch.float32)


class AdvCPCarMeshHelper:
    CAR_MESH_3D_EXAMPLES = {
        "car_000000": np.array([0.0, 0.0, 0.0, 5.00, 2.00, 1.75, 0.0], dtype=np.float32),
        "car_mesh_0200": np.array([0.0, 0.0, 0.0, 4.30, 1.91, 1.26, 0.0], dtype=np.float32),
    }
    _CAR_MESH_DIVIDE_CACHE: dict[Path, Any] = {}
    _REAL_MESH_WARNING_EMITTED = False

    @staticmethod
    def build_spoof_mesh_pieces(spoof_box: np.ndarray) -> list[Any]:
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
        return [AdvCPCarMeshHelper.build_box_piece_mesh(spoof_box, extents, center) for extents, center in pieces]

    @classmethod
    def build_spoof_meshes(cls, spoof_box: np.ndarray, advcp_config: AdvCPConfig) -> list[Any]:
        car_mesh_pieces = cls.build_real_car_mesh_pieces(spoof_box, advcp_config)
        if car_mesh_pieces is not None:
            return car_mesh_pieces
        if not cls._REAL_MESH_WARNING_EMITTED:
            logger.warning("AdvCP early spoofing 3D model assets were not found. Falling back to bbox-shell ray tracing instead of real car mesh.")
            cls._REAL_MESH_WARNING_EMITTED = True
        return cls.build_spoof_mesh_pieces(spoof_box)

    @classmethod
    def build_collision_mesh(cls, spoof_box: np.ndarray, advcp_config: AdvCPConfig) -> Any:
        car_mesh_pieces = cls.build_real_car_mesh_pieces(spoof_box, advcp_config)
        if car_mesh_pieces is not None:
            return car_mesh_pieces[0] if len(car_mesh_pieces) == 1 else cls.merge_meshes(car_mesh_pieces)
        return cls.build_box_piece_mesh(
            spoof_box,
            (float(spoof_box[3]), float(spoof_box[4]), float(spoof_box[5])),
            (0.0, 0.0, float(spoof_box[5]) / 2.0),
        )

    @classmethod
    def build_real_car_mesh_pieces(cls, spoof_box: np.ndarray, advcp_config: AdvCPConfig) -> list[Any] | None:
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
    def post_process_car_meshes(cls, car_mesh_pieces: list[Any], spoof_box: np.ndarray, car_mesh_name: str) -> list[Any]:
        processed_meshes = []
        car_mesh_bbox = cls.CAR_MESH_3D_EXAMPLES.get(car_mesh_name, cls.CAR_MESH_3D_EXAMPLES["car_mesh_0200"])
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
        if car_mesh_path is None:
            return "car_mesh_0200"
        return car_mesh_path.stem

    @classmethod
    def resolve_car_mesh_paths(cls, advcp_config: AdvCPConfig) -> tuple[Path, Path]:
        car_mesh_path = Path(str(AdvCPAttackHelper.require_config_value(advcp_config, "car_mesh_path"))).expanduser()

        if not car_mesh_path.is_absolute():
            car_mesh_path = (Path.cwd() / car_mesh_path).resolve()

        car_mesh_divide_path = Path(str(AdvCPAttackHelper.require_config_value(advcp_config, "car_mesh_divide_path"))).expanduser()
        if not car_mesh_divide_path.is_absolute():
            car_mesh_divide_path = (Path.cwd() / car_mesh_divide_path).resolve()

        return car_mesh_path, car_mesh_divide_path

    @staticmethod
    def build_box_piece_mesh(
        spoof_box: np.ndarray,
        extents: tuple[float, float, float],
        center_local: tuple[float, float, float],
    ) -> Any:
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
    def merge_meshes(meshes: list[Any]) -> Any:
        merged_mesh = copy.deepcopy(meshes[0])
        for mesh in meshes[1:]:
            merged_mesh += mesh
        return merged_mesh

    @staticmethod
    def ray_intersection(meshes: list[Any], rays: np.ndarray) -> np.ndarray:
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
