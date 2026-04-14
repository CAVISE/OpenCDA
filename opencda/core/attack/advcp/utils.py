import logging
from typing import Any

import numpy as np
import yaml

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.transformation_utils import x_to_world

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.utils")

DEFAULT_BOX_SIZE = [4.5, 2.0, 1.6]


def load_advcp_config(config_path: str | None) -> dict[str, Any] | None:
    if not config_path:
        return None

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("AdvCP config must be a mapping.")

    config.setdefault("enabled", True)
    config.setdefault("mode", "spoof")
    config.setdefault("default_size", list(DEFAULT_BOX_SIZE))
    config.setdefault("boxes", [])
    return config


def resolve_attacker_id(attacker_id: Any, valid_agent_ids: list[str]) -> str:
    if attacker_id is None:
        raise ValueError("AdvCP config must define attacker_id.")

    attacker_str = str(attacker_id)
    if attacker_str in valid_agent_ids:
        return attacker_str

    if attacker_str.isdigit():
        matches = [agent_id for agent_id in valid_agent_ids if agent_id.endswith(f"-{attacker_str}")]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"AdvCP attacker_id '{attacker_id}' is ambiguous. Matching agents: {', '.join(matches)}. "
                "Use the full runtime id such as 'cav-1' or 'rsu-1'."
            )

    raise ValueError(f"AdvCP attacker_id '{attacker_id}' does not exist. Known agents: {', '.join(valid_agent_ids)}")


def resolve_late_spoof_boxes(advcp_config: dict[str, Any] | None, memory_data: dict[str, Any] | None) -> tuple[str | None, list[np.ndarray]]:
    if not advcp_config or not advcp_config.get("enabled", True):
        return None, []

    if memory_data is None:
        raise ValueError("AdvCP late spoofing requires current memory data.")

    if advcp_config.get("mode", "spoof") != "spoof":
        raise NotImplementedError(f"AdvCP mode '{advcp_config.get('mode')}' is not available yet.")

    scenario_data = next(iter(memory_data.values()))
    ego_agent_id = _find_ego_agent_id(scenario_data)
    if ego_agent_id is None:
        raise ValueError("Unable to resolve ego agent for AdvCP attack.")

    attacker_id = advcp_config.get("attacker_id")
    if not attacker_id:
        raise ValueError("AdvCP config must define attacker_id.")
    if attacker_id not in scenario_data:
        raise ValueError(f"Attacker '{attacker_id}' is not present in the current tick.")

    ego_state = _load_agent_state(scenario_data, ego_agent_id)
    attacker_state = _load_agent_state(scenario_data, attacker_id)

    box_specs = advcp_config.get("boxes", [])
    if not isinstance(box_specs, list) or len(box_specs) == 0:
        raise ValueError("AdvCP config must define a non-empty boxes list.")

    spoof_boxes = []
    for index, spec in enumerate(box_specs):
        spoof_boxes.append(_resolve_box_spec(spec, index, advcp_config, ego_state, attacker_state))

    batch_attacker_id = "ego" if attacker_id == ego_agent_id else attacker_id
    return batch_attacker_id, spoof_boxes


def _find_ego_agent_id(scenario_data: dict[str, Any]) -> str | None:
    for agent_id, agent_data in scenario_data.items():
        if agent_data.get("ego"):
            return agent_id
    return None


def _load_agent_state(scenario_data: dict[str, Any], agent_id: str) -> dict[str, Any]:
    agent_data = scenario_data[agent_id]
    timestamp = next(key for key in agent_data.keys() if key != "ego")
    yaml_path = agent_data[timestamp]["yaml"]
    params = load_yaml(yaml_path)
    return {
        "agent_id": agent_id,
        "timestamp": timestamp,
        "yaml_path": yaml_path,
        "params": params,
        "lidar_pose": params["lidar_pose"],
        "ego_pose": params.get("true_ego_pos", params["lidar_pose"]),
    }


def _resolve_box_spec(
    spec: dict[str, Any],
    index: int,
    advcp_config: dict[str, Any],
    ego_state: dict[str, Any],
    attacker_state: dict[str, Any],
) -> np.ndarray:
    if not isinstance(spec, dict):
        raise ValueError(f"AdvCP box entry #{index} must be a mapping.")

    has_relative = "relative" in spec
    has_absolute = "absolute" in spec
    if has_relative == has_absolute:
        raise ValueError(f"AdvCP box entry #{index} must define exactly one of 'relative' or 'absolute'.")

    pose = spec["relative"] if has_relative else spec["absolute"]
    pose = _normalize_pose(pose, f"boxes[{index}]")
    size = _normalize_size(spec.get("size", advcp_config.get("default_size", DEFAULT_BOX_SIZE)), f"boxes[{index}].size")

    if has_relative:
        world_pose = _compose_relative_pose(ego_state["ego_pose"], pose)
    else:
        world_pose = pose

    return _world_box_to_sensor_box(world_pose, size, attacker_state["lidar_pose"])


def _normalize_pose(raw_pose: Any, field_name: str) -> np.ndarray:
    pose = np.asarray(raw_pose, dtype=np.float32)
    if pose.shape != (6,):
        raise ValueError(f"{field_name} must contain 6 values: [x, y, z, roll, yaw, pitch].")
    return pose


def _normalize_size(raw_size: Any, field_name: str) -> np.ndarray:
    size = np.asarray(raw_size, dtype=np.float32)
    if size.shape != (3,):
        raise ValueError(f"{field_name} must contain 3 values: [length, width, height].")
    return size


def _compose_relative_pose(reference_pose: np.ndarray | list[float], relative_pose: np.ndarray) -> np.ndarray:
    reference_pose = np.asarray(reference_pose, dtype=np.float32)
    reference_matrix = x_to_world(reference_pose.tolist())
    relative_point = np.array([relative_pose[0], relative_pose[1], relative_pose[2], 1.0], dtype=np.float32)
    world_point = reference_matrix @ relative_point

    world_pose = np.zeros(6, dtype=np.float32)
    world_pose[:3] = world_point[:3]
    world_pose[3:] = reference_pose[3:] + relative_pose[3:]
    return world_pose


def _world_box_to_sensor_box(world_pose: np.ndarray, size: np.ndarray, sensor_pose: list[float]) -> np.ndarray:
    sensor_matrix = x_to_world(sensor_pose)
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
