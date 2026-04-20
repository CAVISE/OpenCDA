from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Any, Mapping, Optional, TypedDict

import numpy as np
import torch
import yaml  # type: ignore

from opencda.core.common.coperception_model_manager import (
    CoperceptionInferenceResult,
    CoperceptionModelManager,
    CoperceptionVisualizationConfig,
    CoperceptionVisualizer,
)

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.advcp_manager")


class AdvCPVisualizationContext(TypedDict):
    attacker_ids: list[str]
    fake_box_tensor: Any | None  # noqa: DC01
    mode: str | None


class AdvCoperceptionVisualizer(CoperceptionVisualizer):
    _DEFAULT_VISUALIZATION_CONFIG: CoperceptionVisualizationConfig = {
        "background": (0, 0, 0),
        "lidar_point_colors": {
            "other": (255, 255, 255),
            "ego": (80, 255, 80),
            "attackers": (255, 90, 90),
            "spoofing": (180, 0, 255),
        },
        "bbox_colors": {
            "gt": (0, 255, 0),
            "pred": (255, 0, 0),
            "fake": (180, 0, 255),
        },
    }

    @classmethod
    def _get_extra_box_tensors(cls, visualization_context: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        if not visualization_context:
            return {}
        return {"fake": visualization_context.get("fake_box_tensor")}

    @classmethod
    def _resolve_point_color(
        cls,
        config: Mapping[str, Any],
        agent_id: str,
        role: str,
        other_color: tuple,
        ego_color: tuple,
        visualization_context: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        lidar_point_colors = config["lidar_point_colors"]
        if agent_id is not None and agent_id in lidar_point_colors:
            return cls._as_uint8_color(lidar_point_colors[agent_id])
        attacker_ids = set((visualization_context or {}).get("attacker_ids", []))
        attacker_color = cls._as_uint8_color(config["lidar_point_colors"].get("attackers", other_color))
        if agent_id is not None and agent_id in attacker_ids:
            return attacker_color
        if role == "ego":
            return ego_color
        return other_color


class AdvCoperceptionModelManager(CoperceptionModelManager):
    DEFAULT_BOX_SIZE = [4.5, 2.0, 1.6]
    VISUALIZER_CLASS = AdvCoperceptionVisualizer
    SEQUENCE_BOX_GROUP_NAMES: tuple[str, ...] = ("pred", "gt", "fake")

    def __init__(
        self,
        opt: Any,
        current_time: str,
        payload_handler: Any = None,
        visualization_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.advcp_config = self.load_config(getattr(opt, "advcp_config", None))
        self.current_memory_data: Optional[dict[Any, Any]] = None
        super().__init__(opt, current_time, payload_handler=payload_handler, visualization_config=visualization_config)

    @staticmethod
    def load_config(config_path: str | None) -> dict[str, Any]:
        config: dict[str, Any] = {}

        if not config_path:
            logger.warning("AdvCP config path is not provided. Falling back to default AdvCP config.")
        else:
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    loaded_config = yaml.safe_load(handle) or {}
            except OSError as exc:
                logger.warning("Unable to load AdvCP config '%s': %s. Falling back to defaults.", config_path, exc)
            else:
                if isinstance(loaded_config, dict):
                    config = loaded_config
                else:
                    logger.warning("AdvCP config '%s' is not a mapping. Falling back to defaults.", config_path)

        config.setdefault("mode", "spoof")
        config.setdefault("default_size", list(AdvCoperceptionModelManager.DEFAULT_BOX_SIZE))
        config.setdefault("boxes", [{"relative": [5.0, 0.0, 0.0, 0.0, 90.0, 0.0]}])
        config.setdefault("attacker_id", "cav-1")
        return config

    def validate_advcp_agents(self, valid_agent_ids: list[str]) -> bool:
        mode = self.advcp_config.get("mode", "spoof")
        attacker_id = self.advcp_config.get("attacker_id")

        if attacker_id is None:
            logger.warning("AdvCP attack will not be applied because attacker_id is not defined in the AdvCP config.")
            self.advcp_config["attacker_id"] = None
        elif attacker_id in valid_agent_ids:
            self.advcp_config["attacker_id"] = attacker_id
        else:
            logger.warning(
                "AdvCP attacker_id '%s' does not exist. Known agents: %s. AdvCP attack will not be applied.",
                attacker_id,
                ", ".join(valid_agent_ids),
            )
            self.advcp_config["attacker_id"] = None

        attacker_ids = [self.advcp_config["attacker_id"]] if self.advcp_config.get("attacker_id") else []

        logger.info("AdvCP mode: %s", mode)
        if attacker_ids:
            logger.info("AdvCP attacks are enabled and will be applied during cooperative perception inference.")
            logger.info("AdvCP attackers: %s", ", ".join(attacker_ids))
            return True
        else:
            logger.warning("AdvCP is enabled, but no valid attackers were resolved. Attacks will not be applied.")
            return False

    def _run_late_inference(self, batch_data: Any) -> CoperceptionInferenceResult:  # noqa: DC04
        return self._build_inference_result(
            *self._inference_late_fusion_attack(
                batch_data,
                self.model,
                self.opencood_dataset,
                self.device,
                advcp_config=self.advcp_config,
                memory_data=self.current_memory_data,
            )
        )

    def _run_early_inference(self, batch_data: Any) -> CoperceptionInferenceResult:  # noqa: DC04
        return self._build_inference_result(
            *self._inference_early_fusion_attack(
                batch_data,
                self.model,
                self.opencood_dataset,
                self.device,
            )
        )

    def _run_intermediate_inference(self, batch_data: Any) -> CoperceptionInferenceResult:  # noqa: DC04
        return self._build_inference_result(
            *self._inference_intermediate_fusion_attack(
                batch_data,
                self.model,
                self.opencood_dataset,
                self.device,
            )
        )

    @staticmethod
    def _inference_late_fusion_attack(
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: dict[str, Any],
        memory_data: Optional[dict[Any, Any]] = None,
    ) -> tuple[Any, Any, Any, AdvCPVisualizationContext]:
        # TODO: Move this up when https://github.com/CAVISE/OpenCDA/pull/65 is merged
        from opencood.utils import box_utils

        output_dict: OrderedDict[str, Any] = OrderedDict()
        advcp_context: AdvCPVisualizationContext = {"attacker_ids": [], "fake_box_tensor": None, "mode": None}

        for cav_id, cav_content in batch_data.items():
            output_dict[cav_id] = model(cav_content)

        mode = advcp_config.get("mode", "spoof")
        advcp_context["mode"] = mode
        if mode == "remove":
            AdvCoperceptionModelManager._raise_late_removal_not_available()

        attacker_id, attack_boxes = AdvCoperceptionModelManager.resolve_late_spoof_boxes(advcp_config, memory_data)
        if attacker_id is not None:
            advcp_context["attacker_ids"] = [attacker_id]

        if not attack_boxes:
            return AdvCoperceptionModelManager._run_default_late_prediction(batch_data, output_dict, dataset, advcp_context)

        if attacker_id not in batch_data:
            logger.warning(
                "AdvCP attack will not be applied on this tick because attacker '%s' is not present in the current batch. "
                "Continuing with normal cooperative perception inference.",
                attacker_id,
            )
            advcp_context["attacker_ids"] = []
            return AdvCoperceptionModelManager._run_default_late_prediction(batch_data, output_dict, dataset, advcp_context)

        pred_box3d_list = []
        pred_box2d_list = []
        pred_fake_list = []

        for cav_id, cav_content in batch_data.items():
            transformation_matrix = cav_content["transformation_matrix"]
            anchor_box = cav_content["anchor_box"]
            prob = output_dict[cav_id]["psm"]
            prob = torch.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)
            reg = output_dict[cav_id]["rm"]

            batch_box3d = dataset.post_processor.delta_to_boxes3d(reg, anchor_box)
            mask = torch.gt(prob, dataset.post_processor.params["target_args"]["score_threshold"])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])
            is_fake = torch.zeros((scores.shape[0],), dtype=torch.bool, device=device)

            if cav_id == attacker_id:
                injected_box_tensors = [
                    AdvCoperceptionModelManager._convert_box_for_model(attack_box, dataset).to(device) for attack_box in attack_boxes
                ]
                stacked_injected_boxes = torch.stack(injected_box_tensors, dim=0)
                injected_scores = torch.ones((len(attack_boxes),), dtype=scores.dtype, device=device)
                injected_is_fake = torch.ones((len(attack_boxes),), dtype=torch.bool, device=device)
                boxes3d = torch.vstack([boxes3d, stacked_injected_boxes])
                scores = torch.hstack([scores, injected_scores])
                is_fake = torch.hstack([is_fake, injected_is_fake])

            if boxes3d.shape[0] == 0:
                continue

            boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=dataset.post_processor.params["order"])
            projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)
            projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
            boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

            pred_box2d_list.append(boxes2d_score)
            pred_box3d_list.append(projected_boxes3d)
            pred_fake_list.append(is_fake)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            raise RuntimeError("AdvCP late spoofing produced no detection result.")

        pred_box2d_tensor = torch.vstack(pred_box2d_list)
        scores = pred_box2d_tensor[:, -1]
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        pred_is_fake_tensor = torch.hstack(pred_fake_list)

        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        pred_is_fake_tensor = pred_is_fake_tensor[keep_index]

        keep_index = box_utils.nms_rotated(pred_box3d_tensor, scores, dataset.post_processor.params["nms_thresh"])
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        pred_is_fake_tensor = pred_is_fake_tensor[keep_index]

        mask = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        pred_score = scores[mask]
        pred_is_fake_tensor = pred_is_fake_tensor[mask]
        gt_box_tensor = dataset.post_processor.generate_gt_bbx(batch_data)

        if torch.any(pred_is_fake_tensor):
            advcp_context["fake_box_tensor"] = pred_box3d_tensor[pred_is_fake_tensor]

        return pred_box3d_tensor, pred_score, gt_box_tensor, advcp_context

    @staticmethod
    def _run_default_late_prediction(
        batch_data: Any,
        output_dict: OrderedDict[str, Any],
        dataset: Any,
        advcp_context: AdvCPVisualizationContext,
    ) -> tuple[Any, Any, Any, AdvCPVisualizationContext]:
        pred_box_tensor, pred_score = dataset.post_processor.post_process(batch_data, output_dict)
        gt_box_tensor = dataset.post_processor.generate_gt_bbx(batch_data)
        return pred_box_tensor, pred_score, gt_box_tensor, advcp_context

    @staticmethod
    def _inference_early_fusion_attack(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        raise NotImplementedError("AdvCP early fusion spoofing is not available yet.")

    @staticmethod
    def _inference_intermediate_fusion_attack(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        raise NotImplementedError("AdvCP intermediate fusion spoofing is not available yet.")

    @staticmethod
    def resolve_late_spoof_boxes(advcp_config: dict[str, Any], memory_data: dict[str, Any] | None) -> tuple[str | None, list[np.ndarray]]:
        if memory_data is None:
            raise ValueError("AdvCP late spoofing requires current memory data.")

        if advcp_config.get("mode", "spoof") != "spoof":
            raise NotImplementedError(f"AdvCP mode '{advcp_config.get('mode')}' is not available yet.")

        scenario_data = next(iter(memory_data.values()))
        ego_agent_id = None
        for agent_id, agent_data in scenario_data.items():
            if agent_data.get("ego"):
                ego_agent_id = agent_id
        if ego_agent_id is None:
            raise ValueError("Unable to resolve ego agent for AdvCP attack.")

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

        ego_state = AdvCoperceptionModelManager._load_agent_state(scenario_data, ego_agent_id)
        attacker_state = AdvCoperceptionModelManager._load_agent_state(scenario_data, attacker_id)

        box_specs = advcp_config.get("boxes", [])
        if not isinstance(box_specs, list) or len(box_specs) == 0:
            raise ValueError("AdvCP config must define a non-empty boxes list.")

        spoof_boxes = []
        for index, spec in enumerate(box_specs):
            spoof_boxes.append(AdvCoperceptionModelManager._resolve_box_spec(spec, index, advcp_config, ego_state, attacker_state))

        batch_attacker_id = "ego" if attacker_id == ego_agent_id else attacker_id
        return batch_attacker_id, spoof_boxes

    @staticmethod
    def _load_agent_state(scenario_data: dict[str, Any], agent_id: str) -> dict[str, Any]:
        agent_data = scenario_data[agent_id]
        timestamp = next(key for key in agent_data.keys() if key != "ego")
        snapshot = agent_data[timestamp]
        yaml_path = snapshot.get("yaml")
        params = snapshot.get("params")
        if params is None:
            if yaml_path is None:
                raise ValueError(f"AdvCP agent state for '{agent_id}' does not define either 'params' or 'yaml'.")
            from opencood.hypes_yaml.yaml_utils import load_yaml

            params = load_yaml(yaml_path)

        return {
            "agent_id": agent_id,
            "timestamp": timestamp,
            "yaml_path": yaml_path,
            "params": params,
            "lidar_pose": params["lidar_pose"],
            "ego_pose": params.get("true_ego_pos", params["lidar_pose"]),
        }

    @staticmethod
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

        pose = np.asarray(spec["relative"] if has_relative else spec["absolute"], dtype=np.float32)
        if pose.shape != (6,):
            raise ValueError(f"boxes[{index}] must contain 6 values: [x, y, z, roll, yaw, pitch].")

        size = np.asarray(
            spec.get("size", advcp_config.get("default_size", AdvCoperceptionModelManager.DEFAULT_BOX_SIZE)),
            dtype=np.float32,
        )
        if size.shape != (3,):
            raise ValueError(f"boxes[{index}].size must contain 3 values: [length, width, height].")

        if has_relative:
            world_pose = AdvCoperceptionModelManager._compose_relative_pose(ego_state["ego_pose"], pose)
        else:
            world_pose = pose

        return AdvCoperceptionModelManager._world_box_to_sensor_box(world_pose, size, attacker_state["lidar_pose"])

    @staticmethod
    def _compose_relative_pose(reference_pose: np.ndarray | list[float], relative_pose: np.ndarray) -> np.ndarray:
        from opencood.utils.transformation_utils import x_to_world

        reference_pose = np.asarray(reference_pose, dtype=np.float32)
        reference_matrix = x_to_world(reference_pose.tolist())
        relative_point = np.array([relative_pose[0], relative_pose[1], relative_pose[2], 1.0], dtype=np.float32)
        world_point = reference_matrix @ relative_point

        world_pose = np.zeros(6, dtype=np.float32)
        world_pose[:3] = world_point[:3]
        world_pose[3:] = reference_pose[3:] + relative_pose[3:]
        return world_pose

    @staticmethod
    def _world_box_to_sensor_box(world_pose: np.ndarray, size: np.ndarray, sensor_pose: list[float]) -> np.ndarray:
        from opencood.utils.transformation_utils import x_to_world

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

    @staticmethod
    def _convert_box_for_model(box_lwh_bottom_center: np.ndarray, dataset: Any) -> torch.Tensor:
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

    @staticmethod
    def _raise_late_removal_not_available() -> None:
        raise NotImplementedError("AdvCP late-fusion removal is not available yet.")
