from __future__ import annotations

import copy
from collections import OrderedDict
import logging
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
from opencood.tools import inference_utils, train_utils
from opencood.utils import box_utils

from opencda.core.attack.advcp.attack_helper import AdvCPAttackHelper
from opencda.core.attack.advcp.early_fusion_attack import AdvCoperceptionEarlyFusionAttack
from opencda.core.attack.advcp.types import AdvCPAttackResult, AdvCPConfig, AdvCPIntermediateAttackState, AdvCPVisualizationContext
from opencda.core.common.coperception_data_processor import LiveMemorySnapshot

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.intermediate_fusion_attack")


class AdvCoperceptionIntermediateFusionAttack:
    @staticmethod
    def run(
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: AdvCPConfig,
        memory_data: OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]] | None = None,
        attack_state: AdvCPIntermediateAttackState | None = None,
    ) -> AdvCPAttackResult:
        mode = AdvCPAttackHelper.require_config_value(advcp_config, "mode")
        advcp_context: AdvCPVisualizationContext = {
            "attacker_ids": [],
            "fake_box_tensor": None,
            "mode": mode,
        }

        match mode:
            case "remove":
                raise NotImplementedError("AdvCP intermediate-fusion removal is not available yet.")
            case "spoof":
                pass
            case _:
                raise NotImplementedError(f"AdvCP mode '{mode}' is not available for intermediate fusion.")
        if memory_data is None:
            raise ValueError("AdvCP intermediate spoofing requires current memory data.")

        intermediate_state: AdvCPIntermediateAttackState = attack_state if attack_state is not None else {}
        attacker_id = str(AdvCPAttackHelper.require_config_value(advcp_config, "attacker_id"))
        current_scenario_data = next(iter(memory_data.values()))
        if attacker_id not in current_scenario_data:
            logger.warning(
                "AdvCP intermediate attack will not be applied on this tick because attacker '%s' is not present in the current scenario data. "
                "Continuing with normal cooperative perception inference.",
                attacker_id,
            )
            AdvCoperceptionIntermediateFusionAttack._update_attack_state(
                intermediate_state,
                memory_data,
                init_perturbation=None,
                online=bool(AdvCPAttackHelper.require_config_value(advcp_config, "online")),
            )
            return (*inference_utils.inference_intermediate_fusion(batch_data, model, dataset), advcp_context)

        current_attacker_index = AdvCoperceptionIntermediateFusionAttack._resolve_attacker_index(batch_data, attacker_id)
        if current_attacker_index is None:
            logger.warning(
                "AdvCP intermediate attack will not be applied on this tick because attacker '%s' is not present in the current batch. "
                "Continuing with normal cooperative perception inference.",
                attacker_id,
            )
            AdvCoperceptionIntermediateFusionAttack._update_attack_state(
                intermediate_state,
                memory_data,
                init_perturbation=None,
                online=bool(AdvCPAttackHelper.require_config_value(advcp_config, "online")),
            )
            return (*inference_utils.inference_intermediate_fusion(batch_data, model, dataset), advcp_context)

        current_ego_boxes = AdvCoperceptionIntermediateFusionAttack._resolve_ego_attack_boxes(
            current_scenario_data,
            advcp_config,
            attacker_id,
        )
        current_target_boxes = torch.stack(
            [AdvCPAttackHelper.convert_box_for_model(attack_box, dataset).to(device) for attack_box in current_ego_boxes],
            dim=0,
        )
        advcp_context["attacker_ids"] = [attacker_id]
        advcp_context["fake_box_tensor"] = box_utils.boxes_to_corners_3d(
            current_target_boxes,
            order=dataset.post_processor.params.get("order", "hwl"),
        )

        sync_enabled = bool(AdvCPAttackHelper.require_config_value(advcp_config, "sync"))
        optimize_memory_data = memory_data
        optimize_ego_boxes = current_ego_boxes
        previous_memory_data = intermediate_state.get("previous_memory_data")
        if sync_enabled and previous_memory_data is not None:
            previous_scenario_data = next(iter(previous_memory_data.values()))
            if attacker_id in previous_scenario_data:
                optimize_memory_data = previous_memory_data
                optimize_ego_boxes = AdvCoperceptionIntermediateFusionAttack._resolve_ego_attack_boxes(
                    previous_scenario_data,
                    advcp_config,
                    attacker_id,
                )
            else:
                logger.warning(
                    "AdvCP intermediate previous-tick optimization skipped previous tick because attacker '%s' was not present. "
                    "Falling back to current-tick optimization.",
                    attacker_id,
                )

        try:
            pred_box_tensor, pred_score, gt_box_tensor, init_perturbation = AdvCoperceptionIntermediateFusionAttack._optimize_spoofing(
                model,
                dataset,
                device,
                advcp_config,
                attacker_id,
                optimize_memory_data,
                optimize_ego_boxes,
                memory_data if optimize_memory_data is not memory_data else None,
                current_ego_boxes if optimize_memory_data is not memory_data else None,
                intermediate_state.get("init_perturbation"),
            )
        finally:
            dataset.update_database(memory_data=memory_data)

        AdvCoperceptionIntermediateFusionAttack._update_attack_state(
            intermediate_state,
            memory_data,
            init_perturbation=init_perturbation,
            online=bool(AdvCPAttackHelper.require_config_value(advcp_config, "online")),
        )
        return pred_box_tensor, pred_score, gt_box_tensor, advcp_context

    @staticmethod
    def _update_attack_state(
        attack_state: AdvCPIntermediateAttackState,
        memory_data: OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]],
        init_perturbation: list[np.ndarray] | None,
        online: bool,
    ) -> None:
        attack_state["previous_memory_data"] = copy.deepcopy(memory_data)
        attack_state["init_perturbation"] = init_perturbation if online else None

    @staticmethod
    def _resolve_ego_attack_boxes(
        scenario_data: Mapping[str, Any],
        advcp_config: AdvCPConfig,
        attacker_id: str,
    ) -> list[np.ndarray]:
        _, _, _, attack_boxes = AdvCPAttackHelper.resolve_spoof_boxes_for_ego(
            scenario_data,
            advcp_config,
            attacker_id,
        )
        return attack_boxes

    @staticmethod
    def _resolve_attacker_index(batch_data: Mapping[str, Any], attacker_id: str) -> int | None:
        ego_entry = batch_data.get("ego")
        if not isinstance(ego_entry, Mapping):
            return None

        agent_ids = ego_entry.get("origin_lidar_agent_ids")
        if not isinstance(agent_ids, Sequence):
            return None

        agent_ids_list = [str(agent_id) for agent_id in agent_ids]
        if attacker_id not in agent_ids_list:
            return None
        return agent_ids_list.index(attacker_id)

    @classmethod
    def _build_batch_from_memory(
        cls,
        dataset: Any,
        device: torch.device,
        memory_data: OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]],
    ) -> Mapping[str, Any]:
        dataset.update_database(memory_data=memory_data)
        batch = dataset.collate_batch_test([dataset[0]])
        return train_utils.to_device(batch, device)

    @classmethod
    def _apply_init_spoof_to_memory(
        cls,
        memory_data: OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]],
        advcp_config: AdvCPConfig,
        attacker_id: str,
    ) -> OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]]:
        attacked_memory = copy.deepcopy(memory_data)
        attacked_scenario_data = next(iter(attacked_memory.values()))
        original_scenario_data = next(iter(memory_data.values()))

        _, _, _, attack_boxes = AdvCPAttackHelper.resolve_spoof_boxes_for_agent(original_scenario_data, advcp_config, attacker_id)
        attacked_agent_data = attacked_scenario_data[attacker_id]
        attacked_timestamp = next(key for key in attacked_agent_data.keys() if key != "ego")
        attacked_snapshot = cast(LiveMemorySnapshot, attacked_agent_data[attacked_timestamp])
        attacker_lidar = attacked_snapshot.get("lidar_np")
        if attacker_lidar is None:
            raise ValueError(f"AdvCP intermediate init requires in-memory lidar_np for attacker '{attacker_id}'.")

        lidar_poses = {
            agent_id: np.asarray(AdvCPAttackHelper.load_agent_state(original_scenario_data, agent_id)["lidar_pose"], dtype=np.float32)
            for agent_id in original_scenario_data
        }
        density = AdvCoperceptionEarlyFusionAttack._resolve_density(AdvCPAttackHelper.require_config_value(advcp_config, "density"))

        spoofed_lidar = np.asarray(attacker_lidar, dtype=np.float32)
        spoofing_mask = np.zeros((spoofed_lidar.shape[0],), dtype=np.bool_)
        for attack_box in attack_boxes:
            spoofed_lidar, spoofing_mask = AdvCoperceptionEarlyFusionAttack._apply_sampled_ray_traced_spoof(
                spoofed_lidar,
                spoofing_mask,
                attack_box,
                lidar_poses,
                attacker_id,
                advcp_config,
                density,
            )

        attacked_snapshot["lidar_np"] = spoofed_lidar
        attacked_snapshot["spoofing_mask"] = spoofing_mask
        return attacked_memory

    @classmethod
    def _optimize_spoofing(
        cls,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: AdvCPConfig,
        attacker_id: str,
        optimize_memory_data: OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]],
        optimize_attack_boxes: Sequence[np.ndarray],
        real_memory_data: OrderedDict[int, OrderedDict[str, OrderedDict[str, LiveMemorySnapshot | bool]]] | None,
        real_attack_boxes: Sequence[np.ndarray] | None,
        stored_init_perturbation: list[np.ndarray] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[np.ndarray] | None]:
        torch.manual_seed(1)
        np.random.seed(1)

        max_perturb = float(AdvCPAttackHelper.require_config_value(advcp_config, "max_perturb"))
        learning_rate = float(AdvCPAttackHelper.require_config_value(advcp_config, "lr"))
        optimization_steps = int(AdvCPAttackHelper.require_config_value(advcp_config, "step"))
        feature_size = int(AdvCPAttackHelper.require_config_value(advcp_config, "feature_size"))
        use_init = bool(AdvCPAttackHelper.require_config_value(advcp_config, "init"))

        original_optimize_batch = cls._build_batch_from_memory(dataset, device, optimize_memory_data)
        optimize_batch = (
            cls._build_batch_from_memory(dataset, device, cls._apply_init_spoof_to_memory(optimize_memory_data, advcp_config, attacker_id))
            if use_init
            else original_optimize_batch
        )

        optimize_attacker_index = cls._resolve_attacker_index(optimize_batch, attacker_id)
        if optimize_attacker_index is None:
            logger.warning(
                "AdvCP intermediate attack will not be applied on this tick because attacker '%s' is not present in the optimization batch. "
                "Continuing with normal cooperative perception inference.",
                attacker_id,
            )
            return (*inference_utils.inference_intermediate_fusion(original_optimize_batch, model, dataset), None)

        real_original_batch: Mapping[str, Any] | None = None
        real_batch: Mapping[str, Any] | None = None
        real_attacker_index: int | None = None
        if real_memory_data is not None and real_attack_boxes is not None:
            real_original_batch = cls._build_batch_from_memory(dataset, device, real_memory_data)
            real_batch = (
                cls._build_batch_from_memory(dataset, device, cls._apply_init_spoof_to_memory(real_memory_data, advcp_config, attacker_id))
                if use_init
                else real_original_batch
            )
            real_attacker_index = cls._resolve_attacker_index(real_batch, attacker_id)
            if real_attacker_index is None:
                logger.warning(
                    "AdvCP intermediate attack will not be applied on this tick because attacker '%s' is not present in the current batch. "
                    "Continuing with normal cooperative perception inference.",
                    attacker_id,
                )
                return (*inference_utils.inference_intermediate_fusion(real_original_batch, model, dataset), None)

        with torch.no_grad():
            _, optimize_spatial_features = cls._attack_forward(
                optimize_batch,
                model,
                optimize_attacker_index,
                perturbations=None,
                centers=None,
                max_perturb=max_perturb,
            )
            _, original_spatial_features = cls._attack_forward(
                original_optimize_batch,
                model,
                optimize_attacker_index,
                perturbations=None,
                centers=None,
                max_perturb=max_perturb,
            )

            real_spatial_features: torch.Tensor | None = None
            real_original_spatial_features: torch.Tensor | None = None
            if real_batch is not None and real_original_batch is not None and real_attacker_index is not None:
                _, real_spatial_features = cls._attack_forward(
                    real_batch,
                    model,
                    real_attacker_index,
                    perturbations=None,
                    centers=None,
                    max_perturb=max_perturb,
                )
                _, real_original_spatial_features = cls._attack_forward(
                    real_original_batch,
                    model,
                    real_attacker_index,
                    perturbations=None,
                    centers=None,
                    max_perturb=max_perturb,
                )

        feature_dim = int(optimize_spatial_features[optimize_attacker_index].shape[0])
        optimize_centers = [cls._point_to_feature_index(attack_box, dataset) for attack_box in optimize_attack_boxes]
        real_centers = [cls._point_to_feature_index(attack_box, dataset) for attack_box in (real_attack_boxes or optimize_attack_boxes)]
        base_perturbations = cls._extract_base_perturbations(
            optimize_spatial_features[optimize_attacker_index],
            original_spatial_features[optimize_attacker_index],
            optimize_centers,
            feature_size,
        )
        real_base_perturbations = (
            cls._extract_base_perturbations(
                real_spatial_features[real_attacker_index],
                real_original_spatial_features[real_attacker_index],
                real_centers,
                feature_size,
            )
            if real_spatial_features is not None and real_original_spatial_features is not None and real_attacker_index is not None
            else [torch.zeros((feature_dim, 2 * feature_size, 2 * feature_size), device=optimize_spatial_features.device) for _ in optimize_centers]
        )

        perturbations = cls._initialize_perturbations(
            feature_dim,
            feature_size,
            optimize_spatial_features.device,
            stored_init_perturbation,
            len(optimize_centers),
        )
        optimizer = torch.optim.Adam(perturbations, lr=learning_rate)

        optimize_target_boxes = torch.stack(
            [
                AdvCPAttackHelper.convert_box_for_model(attack_box, dataset).to(optimize_spatial_features.device)
                for attack_box in optimize_attack_boxes
            ],
            dim=0,
        )
        best_loss = float("inf")
        best_pred_box_tensor: torch.Tensor | None = None
        best_pred_score: torch.Tensor | None = None
        best_gt_box_tensor: torch.Tensor | None = None
        best_init_perturbation: list[np.ndarray] | None = None

        for _ in range(optimization_steps):
            output_dict, _ = cls._attack_forward(
                original_optimize_batch,
                model,
                optimize_attacker_index,
                perturbations=[base_perturbation + perturbation for base_perturbation, perturbation in zip(base_perturbations, perturbations)],
                centers=optimize_centers,
                max_perturb=max_perturb,
            )
            loss = cls._compute_spoof_loss(output_dict, original_optimize_batch, dataset, optimize_target_boxes)

            with torch.no_grad():
                if real_original_batch is not None and real_attacker_index is not None:
                    eval_output_dict, _ = cls._attack_forward(
                        real_original_batch,
                        model,
                        real_attacker_index,
                        perturbations=[
                            real_base_perturbation + perturbation
                            for real_base_perturbation, perturbation in zip(real_base_perturbations, perturbations)
                        ],
                        centers=real_centers,
                        max_perturb=max_perturb,
                    )
                    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(real_original_batch, eval_output_dict)
                else:
                    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(original_optimize_batch, output_dict)

            if loss.item() < best_loss:
                best_loss = float(loss.item())
                best_pred_box_tensor = pred_box_tensor
                best_pred_score = pred_score
                best_gt_box_tensor = gt_box_tensor
                best_init_perturbation = [
                    torch.clamp(perturbation.detach(), min=-max_perturb, max=max_perturb).cpu().numpy() / 2.0 for perturbation in perturbations
                ]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if best_gt_box_tensor is None:
            inference_batch = real_original_batch if real_original_batch is not None else original_optimize_batch
            return (*inference_utils.inference_intermediate_fusion(inference_batch, model, dataset), None)

        return best_pred_box_tensor, best_pred_score, best_gt_box_tensor, best_init_perturbation

    @staticmethod
    def _initialize_perturbations(
        feature_dim: int,
        feature_size: int,
        device: torch.device | str,
        stored_init_perturbation: list[np.ndarray] | None,
        expected_count: int,
    ) -> list[torch.Tensor]:
        perturbations: list[torch.Tensor] = []
        for index in range(expected_count):
            if stored_init_perturbation is not None and index < len(stored_init_perturbation):
                perturbation = torch.from_numpy(stored_init_perturbation[index]).to(device=device, dtype=torch.float32)
            else:
                perturbation = torch.zeros((feature_dim, 2 * feature_size, 2 * feature_size), device=device, dtype=torch.float32)
            perturbation.requires_grad = True
            perturbations.append(perturbation)
        return perturbations

    @staticmethod
    def _extract_base_perturbations(
        spoofed_features: torch.Tensor,
        original_features: torch.Tensor,
        centers: Sequence[np.ndarray],
        feature_size: int,
    ) -> list[torch.Tensor]:
        base_perturbations = []
        for center in centers:
            patch = AdvCoperceptionIntermediateFusionAttack._extract_feature_patch(
                spoofed_features - original_features,
                center,
                feature_size,
            )
            base_perturbations.append(patch.detach())
        return base_perturbations

    @staticmethod
    def _extract_feature_patch(features: torch.Tensor, center: np.ndarray, feature_size: int) -> torch.Tensor:
        _, height, width = features.shape
        center_x = int(center[0])
        center_y = int(center[1])
        patch_height = 2 * feature_size
        patch_width = 2 * feature_size
        patch = torch.zeros((features.shape[0], patch_height, patch_width), device=features.device, dtype=features.dtype)

        x_start = max(center_x - feature_size, 0)
        x_end = min(center_x + feature_size, width)
        y_start = max(center_y - feature_size, 0)
        y_end = min(center_y + feature_size, height)

        patch_x_start = x_start - (center_x - feature_size)
        patch_x_end = patch_x_start + (x_end - x_start)
        patch_y_start = y_start - (center_y - feature_size)
        patch_y_end = patch_y_start + (y_end - y_start)

        patch[:, patch_y_start:patch_y_end, patch_x_start:patch_x_end] = features[:, y_start:y_end, x_start:x_end]
        return patch

    @classmethod
    def _attack_forward(
        cls,
        batch_data: Mapping[str, Any],
        model: Any,
        attacker_index: int,
        perturbations: Sequence[torch.Tensor] | None,
        centers: Sequence[np.ndarray] | None,
        max_perturb: float,
    ) -> tuple[OrderedDict[str, dict[str, torch.Tensor]], torch.Tensor]:
        model_name = type(model).__name__
        if model_name == "VoxelNetIntermediate":
            return cls._attack_forward_voxelnet(batch_data, model, attacker_index, perturbations, centers, max_perturb)
        return cls._attack_forward_point_pillar(batch_data, model, attacker_index, perturbations, centers, max_perturb)

    @classmethod
    def _attack_forward_point_pillar(
        cls,
        batch_data: Mapping[str, Any],
        model: Any,
        attacker_index: int,
        perturbations: Sequence[torch.Tensor] | None,
        centers: Sequence[np.ndarray] | None,
        max_perturb: float,
    ) -> tuple[OrderedDict[str, dict[str, torch.Tensor]], torch.Tensor]:
        ego_entry = batch_data["ego"]
        processed_lidar = ego_entry["processed_lidar"]
        record_len = ego_entry["record_len"]
        pairwise_t_matrix = ego_entry.get("pairwise_t_matrix")

        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
            "record_len": record_len,
        }
        if "PointPillarintermediateV2VAM" in type(model).__name__:
            batch_dict["voxel_features"] = batch_dict["voxel_features"].float()

        batch_dict = model.pillar_vfe(batch_dict)
        batch_dict = model.scatter(batch_dict)
        spatial_features = cls._apply_perturbations_to_attacker_features(
            batch_dict["spatial_features"],
            attacker_index,
            perturbations,
            centers,
            max_perturb,
        )
        batch_dict["spatial_features"] = spatial_features
        output = cls._run_point_pillar_head(model, batch_dict, ego_entry, record_len, pairwise_t_matrix)
        return OrderedDict(ego=output), spatial_features

    @classmethod
    def _attack_forward_voxelnet(
        cls,
        batch_data: Mapping[str, Any],
        model: Any,
        attacker_index: int,
        perturbations: Sequence[torch.Tensor] | None,
        centers: Sequence[np.ndarray] | None,
        max_perturb: float,
    ) -> tuple[OrderedDict[str, dict[str, torch.Tensor]], torch.Tensor]:
        ego_entry = batch_data["ego"]
        processed_lidar = ego_entry["processed_lidar"]
        record_len = ego_entry["record_len"]
        voxel_coords = processed_lidar["voxel_coords"]

        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": voxel_coords,
            "voxel_num_points": processed_lidar["voxel_num_points"],
        }
        record_len_tmp = record_len.cpu() if getattr(voxel_coords, "is_cuda", False) else record_len
        model.N = int(sum(record_len_tmp.detach().cpu().numpy().tolist()))

        voxelwise_features = model.svfe(batch_dict)["pillar_features"]
        voxel_coords_np = voxel_coords.detach().cpu().numpy()
        voxelwise_features = model.voxel_indexing(voxelwise_features, voxel_coords_np)
        voxelwise_features = model.cml(voxelwise_features)
        spatial_features = voxelwise_features.view(model.N, -1, model.H, model.W)
        if getattr(model, "compression", False):
            spatial_features = model.compression_layer(spatial_features)

        spatial_features = cls._apply_perturbations_to_attacker_features(
            spatial_features,
            attacker_index,
            perturbations,
            centers,
            max_perturb,
        )
        fused_features = model.fusion_net(spatial_features, record_len)
        psm, rm = model.rpn(fused_features)
        return OrderedDict(ego={"psm": psm, "rm": rm}), spatial_features

    @staticmethod
    def _run_point_pillar_head(
        model: Any,
        batch_dict: dict[str, Any],
        ego_entry: Mapping[str, Any],
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        model_name = type(model).__name__
        batch_dict = model.backbone(batch_dict)
        spatial_features_2d = batch_dict["spatial_features_2d"]

        if getattr(model, "shrink_flag", False):
            spatial_features_2d = model.shrink_conv(spatial_features_2d)

        if model_name == "PointPillarWhere2comm":
            psm_single = model.cls_head(spatial_features_2d)
            if getattr(model, "compression", False):
                spatial_features_2d = model.naive_compressor(spatial_features_2d)
            if model.multi_scale:
                fused_feature, _ = model.fusion_net(batch_dict["spatial_features"], psm_single, record_len, pairwise_t_matrix, model.backbone)
                if getattr(model, "shrink_flag", False):
                    fused_feature = model.shrink_conv(fused_feature)
            else:
                fused_feature, _ = model.fusion_net(spatial_features_2d, psm_single, record_len, pairwise_t_matrix)
            return {"psm": model.cls_head(fused_feature), "rm": model.reg_head(fused_feature)}

        if getattr(model, "compression", False):
            spatial_features_2d = model.naive_compressor(spatial_features_2d)

        if model_name == "PointPillarV2VNet":
            fused_feature = model.fusion_net(spatial_features_2d, record_len, pairwise_t_matrix)
            return {"psm": model.cls_head(fused_feature), "rm": model.reg_head(fused_feature)}

        if model_name in {"PointPillarFCooper", "PointPillarintermediateV2VAM"}:
            fused_feature = model.fusion_net(spatial_features_2d, record_len)
            return {"psm": model.cls_head(fused_feature), "rm": model.reg_head(fused_feature)}

        if model_name in {"PointPillarIntermediate", "PointPillarIntermediateV2", "PointPillarCoAlign"}:
            return {"psm": model.cls_head(spatial_features_2d), "rm": model.reg_head(spatial_features_2d)}

        raise NotImplementedError(
            f"AdvCP intermediate spoofing is not implemented for model '{model_name}'. "
            "Supported models are point-pillar intermediate, V2VNet, F-Cooper/V2VAM, Where2comm, and VoxelNet."
        )

    @classmethod
    def _apply_perturbations_to_attacker_features(
        cls,
        spatial_features: torch.Tensor,
        attacker_index: int,
        perturbations: Sequence[torch.Tensor] | None,
        centers: Sequence[np.ndarray] | None,
        max_perturb: float,
    ) -> torch.Tensor:
        if not perturbations or not centers:
            return spatial_features

        attacked_features = spatial_features.clone()
        attacker_features = attacked_features[attacker_index]
        for perturbation, center in zip(perturbations, centers):
            clipped_perturbation = torch.clamp(perturbation, min=-max_perturb, max=max_perturb)
            attacker_features = attacker_features + cls._build_perturbation_feature_map(attacker_features, clipped_perturbation, center)
        attacked_features[attacker_index] = attacker_features
        return attacked_features

    @staticmethod
    def _build_perturbation_feature_map(
        attacker_features: torch.Tensor,
        perturbation: torch.Tensor,
        center: np.ndarray,
    ) -> torch.Tensor:
        channels, height, width = attacker_features.shape
        center_x = float(center[0])
        center_y = float(center[1])
        aligned_center_x = int(np.floor(center_x))
        aligned_center_y = int(np.floor(center_y))

        perturbation_map = torch.zeros_like(attacker_features)
        half_patch_height = perturbation.shape[1] // 2
        half_patch_width = perturbation.shape[2] // 2
        x_start = max(aligned_center_x - half_patch_width, 0)
        x_end = min(aligned_center_x + half_patch_width, width)
        y_start = max(aligned_center_y - half_patch_height, 0)
        y_end = min(aligned_center_y + half_patch_height, height)

        patch_x_start = x_start - (aligned_center_x - half_patch_width)
        patch_x_end = patch_x_start + (x_end - x_start)
        patch_y_start = y_start - (aligned_center_y - half_patch_height)
        patch_y_end = patch_y_start + (y_end - y_start)
        perturbation_map[:, y_start:y_end, x_start:x_end] = perturbation[:, patch_y_start:patch_y_end, patch_x_start:patch_x_end]

        theta = torch.tensor(
            [
                [
                    [1.0, 0.0, (center_y - aligned_center_y) * 2.0 / max(width, 1)],
                    [0.0, 1.0, (center_x - aligned_center_x) * 2.0 / max(height, 1)],
                ]
            ],
            dtype=attacker_features.dtype,
            device=attacker_features.device,
        )
        grid = F.affine_grid(theta, [1, channels, height, width], align_corners=False)
        return F.grid_sample(perturbation_map.unsqueeze(0), grid, align_corners=False)[0]

    @staticmethod
    def _point_to_feature_index(attack_box: np.ndarray, dataset: Any) -> np.ndarray:
        lidar_range = np.asarray(dataset.pre_processor.params["cav_lidar_range"][:3], dtype=np.float32)
        voxel_size = np.asarray(dataset.pre_processor.params["args"]["voxel_size"], dtype=np.float32)
        return np.floor((attack_box[:3] - lidar_range) / voxel_size).astype(np.int32)

    @classmethod
    def _compute_spoof_loss(
        cls,
        output_dict: Mapping[str, Any],
        batch_data: Mapping[str, Any],
        dataset: Any,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(output_dict["ego"]["psm"].permute(0, 2, 3, 1)).reshape(-1)
        proposals = dataset.post_processor.delta_to_boxes3d(output_dict["ego"]["rm"], batch_data["ego"]["anchor_box"])[0]
        proposals_lwh = cls._model_boxes_to_lwh(proposals, dataset)
        target_boxes_lwh = cls._model_boxes_to_lwh(target_boxes, dataset)

        loss_terms: list[torch.Tensor] = []
        for target_box in target_boxes_lwh:
            iou_weights = cls._compute_iou_weights(proposals_lwh, target_box)
            box_mask = iou_weights >= 0.01
            if torch.any(box_mask):
                log_prob = torch.log(torch.clamp(1.0 - probabilities[box_mask], min=1e-6))
                loss_terms.append((iou_weights[box_mask] * log_prob).sum())

        if not loss_terms:
            return probabilities.sum() * 0.0
        return torch.stack(loss_terms).sum()

    @staticmethod
    def _model_boxes_to_lwh(boxes: torch.Tensor, dataset: Any) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        order = dataset.post_processor.params.get("order", "hwl")
        if order == "hwl":
            return boxes[:, [0, 1, 2, 5, 4, 3, 6]]
        if order == "lwh":
            return boxes
        raise NotImplementedError(f"Unsupported box order for AdvCP intermediate spoofing: {order}")

    @staticmethod
    def _compute_iou_weights(proposals_lwh: torch.Tensor, target_box_lwh: torch.Tensor) -> torch.Tensor:
        if proposals_lwh.shape[0] == 0:
            return torch.zeros((0,), dtype=target_box_lwh.dtype, device=target_box_lwh.device)

        repeated_target_boxes = target_box_lwh.unsqueeze(0).expand(proposals_lwh.shape[0], -1)
        proposal_corners = box_utils.boxes_to_corners2d(proposals_lwh, order="lwh")[:, :, :2]
        target_corners = box_utils.boxes_to_corners2d(repeated_target_boxes, order="lwh")[:, :, :2]
        intersection_area = AdvCoperceptionIntermediateFusionAttack._oriented_box_intersection_2d(
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
        return cls._box1_in_box2(corners1, corners2), cls._box1_in_box2(corners2, corners1)

    @staticmethod
    def _box1_in_box2(corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
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
