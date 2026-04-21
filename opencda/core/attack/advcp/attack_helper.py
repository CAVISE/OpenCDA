from __future__ import annotations

import copy
import logging
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.transformation_utils import x_to_world

from opencda.core.attack.advcp.types import AdvCPAgentState

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.advcp_manager")


class AdvCPAttackHelper:
    DEFAULT_BOX_SIZE = [4.5, 2.0, 1.6]

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

    @classmethod
    def resolve_spoof_boxes(
        cls,
        advcp_config: Mapping[str, Any],
        memory_data: Mapping[Any, Any] | None,
    ) -> tuple[str | None, list[np.ndarray]]:
        if memory_data is None:
            raise ValueError("AdvCP late spoofing requires current memory data.")

        if advcp_config.get("mode", "spoof") != "spoof":
            raise NotImplementedError(f"AdvCP mode '{advcp_config.get('mode')}' is not available yet.")

        scenario_data = next(iter(memory_data.values()))
        ego_agent_id = cls.resolve_ego_agent_id(scenario_data)

        attacker_id = advcp_config.get("attacker_id")
        if not attacker_id:
            logger.warning("AdvCP attack will not be applied on this tick because no valid attacker_id is configured.")
            return None, []
        if attacker_id not in scenario_data:
            logger.warning(
                "AdvCP attack will not be applied on this tick because attacker '%s' is not present in the current scenario data. "
                "Continuing with normal cooperative perception inference.",
                attacker_id,
            )
            return None, []

        _, _, _, attack_boxes = cls.resolve_spoof_boxes_for_agent(scenario_data, advcp_config, attacker_id)
        batch_attacker_id = "ego" if attacker_id == ego_agent_id else attacker_id
        return batch_attacker_id, attack_boxes

    @classmethod
    def resolve_spoof_boxes_for_agent(
        cls,
        scenario_data: Mapping[str, Any],
        advcp_config: Mapping[str, Any],
        attacker_id: str,
    ) -> tuple[str, AdvCPAgentState, AdvCPAgentState, list[np.ndarray]]:
        ego_agent_id = cls.resolve_ego_agent_id(scenario_data)
        ego_state = cls.load_agent_state(scenario_data, ego_agent_id)
        attacker_state = cls.load_agent_state(scenario_data, attacker_id)

        box_specs = advcp_config.get("boxes", [])
        if not isinstance(box_specs, list) or len(box_specs) == 0:
            raise ValueError("AdvCP config must define a non-empty boxes list.")

        attack_boxes = [cls.resolve_box_spec(spec, index, advcp_config, ego_state, attacker_state) for index, spec in enumerate(box_specs)]
        return ego_agent_id, ego_state, attacker_state, attack_boxes

    @classmethod
    def resolve_box_spec(
        cls,
        spec: dict[str, Any],
        index: int,
        advcp_config: Mapping[str, Any],
        ego_state: AdvCPAgentState,
        attacker_state: AdvCPAgentState,
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
            spec.get("size", advcp_config.get("default_size", cls.DEFAULT_BOX_SIZE)),
            dtype=np.float32,
        )
        if size.shape != (3,):
            raise ValueError(f"boxes[{index}].size must contain 3 values: [length, width, height].")

        if has_relative:
            world_pose = cls.compose_relative_pose(ego_state["ego_pose"], pose)
        else:
            world_pose = pose

        return cls.world_box_to_sensor_box(world_pose, size, attacker_state["lidar_pose"])

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
    DEFAULT_CAR_MESH_FILENAME = "car_mesh_0200.ply"
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
    def build_spoof_meshes(cls, spoof_box: np.ndarray, advcp_config: Mapping[str, Any]) -> list[Any]:
        car_mesh_pieces = cls.build_real_car_mesh_pieces(spoof_box, advcp_config)
        if car_mesh_pieces is not None:
            return car_mesh_pieces
        if not cls._REAL_MESH_WARNING_EMITTED:
            logger.warning("AdvCP early spoofing 3D model assets were not found. Falling back to bbox-shell ray tracing instead of real car mesh.")
            cls._REAL_MESH_WARNING_EMITTED = True
        return cls.build_spoof_mesh_pieces(spoof_box)

    @classmethod
    def build_collision_mesh(cls, spoof_box: np.ndarray, advcp_config: Mapping[str, Any]) -> Any:
        car_mesh_pieces = cls.build_real_car_mesh_pieces(spoof_box, advcp_config)
        if car_mesh_pieces is not None:
            return car_mesh_pieces[0] if len(car_mesh_pieces) == 1 else cls.merge_meshes(car_mesh_pieces)
        return cls.build_box_piece_mesh(
            spoof_box,
            (float(spoof_box[3]), float(spoof_box[4]), float(spoof_box[5])),
            (0.0, 0.0, float(spoof_box[5]) / 2.0),
        )

    @classmethod
    def build_real_car_mesh_pieces(cls, spoof_box: np.ndarray, advcp_config: Mapping[str, Any]) -> list[Any] | None:
        import open3d as o3d

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
    def resolve_car_mesh_paths(cls, advcp_config: Mapping[str, Any]) -> tuple[Path | None, Path | None]:
        local_model_root = Path(__file__).resolve().parent / "3d_models"
        car_mesh_path_value = advcp_config.get("car_mesh_path", advcp_config.get("model_path"))
        if car_mesh_path_value:
            car_mesh_path = Path(str(car_mesh_path_value)).expanduser()
        else:
            car_mesh_path = local_model_root / cls.DEFAULT_CAR_MESH_FILENAME

        if not car_mesh_path.is_absolute():
            car_mesh_path = (Path.cwd() / car_mesh_path).resolve()

        car_mesh_divide_path_value = advcp_config.get("car_mesh_divide_path", advcp_config.get("mesh_divide_path"))
        if car_mesh_divide_path_value:
            car_mesh_divide_path = Path(str(car_mesh_divide_path_value)).expanduser()
            if not car_mesh_divide_path.is_absolute():
                car_mesh_divide_path = (Path.cwd() / car_mesh_divide_path).resolve()
        else:
            car_mesh_divide_path = local_model_root / "spoof" / "car_mesh_divide.pkl"
            if not car_mesh_divide_path.exists():
                car_mesh_divide_path = car_mesh_path.parent / "spoof" / "car_mesh_divide.pkl"

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
