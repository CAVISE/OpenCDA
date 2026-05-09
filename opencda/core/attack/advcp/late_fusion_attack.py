from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy.typing as npt
import torch
from opencda.core.attack.advcp.attack_helper import AdvCPAttackHelper
from opencda.core.attack.advcp.intermediate_fusion_attack import AdvCoperceptionIntermediateFusionAttack
from opencda.core.attack.advcp.types import AdvCPAttackResult, AdvCPConfig, AdvCPMemoryData, AdvCPVisualizationContext


class AdvCoperceptionLateFusionAttack:
    REMOVAL_IOU_THRESHOLD = 0.1

    @staticmethod
    def run(
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: AdvCPConfig,
        memory_data: AdvCPMemoryData | None = None,
    ) -> AdvCPAttackResult:
        # TODO: Move this up when https://github.com/CAVISE/OpenCDA/pull/65 is merged
        from opencood.utils import box_utils

        output_dict: OrderedDict[str, Any] = OrderedDict()

        for cav_id, cav_content in batch_data.items():
            output_dict[cav_id] = model(cav_content)

        mode = AdvCPAttackHelper.require_config_value(advcp_config, "mode")
        advcp_context = AdvCPVisualizationContext(mode=mode)
        match mode:
            case "removal" | "spoofing":
                pass
            case _:
                raise NotImplementedError(f"AdvCP mode '{mode}' is not available for late fusion.")

        configured_attacker_ids = AdvCPAttackHelper.resolve_configured_attacker_ids(advcp_config)
        if len(configured_attacker_ids) == 0:
            AdvCPAttackHelper.raise_no_configured_attackers("late")

        attacker_ids, attack_boxes_by_attacker = AdvCoperceptionLateFusionAttack.resolve_spoof_boxes_by_attacker(
            advcp_config,
            memory_data,
        )
        advcp_context.attacker_ids = attacker_ids

        if not attack_boxes_by_attacker:
            scenario_data = next(iter(memory_data.values())) if memory_data else {}
            AdvCPAttackHelper.report_missing_attackers_from_current_batch(
                configured_attacker_ids,
                scenario_data.keys(),
                fusion_name="late",
            )
            return AdvCoperceptionLateFusionAttack._run_default_prediction(batch_data, output_dict, dataset, advcp_context)

        present_batch_attacker_ids, _ = AdvCPAttackHelper.resolve_present_and_missing_attackers(
            list(attack_boxes_by_attacker.keys()),
            batch_data.keys(),
        )
        attack_boxes_by_attacker_in_batch: dict[str, list[npt.NDArray]] = {
            attacker_id: attack_boxes_by_attacker[attacker_id] for attacker_id in present_batch_attacker_ids
        }
        if not attack_boxes_by_attacker_in_batch:
            AdvCPAttackHelper.report_missing_attackers_from_current_batch(
                attacker_ids,
                batch_data.keys(),
                fusion_name="late",
            )
            advcp_context.attacker_ids = []
            return AdvCoperceptionLateFusionAttack._run_default_prediction(batch_data, output_dict, dataset, advcp_context)

        pred_box3d_list = []
        pred_box2d_list = []
        pred_fake_list = []
        removed_target_corners_list: list[torch.Tensor] = []

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

            cav_attack_boxes = attack_boxes_by_attacker_in_batch.get(cav_id)
            if cav_attack_boxes:
                target_box_tensors = torch.stack(
                    [AdvCPAttackHelper.convert_box_for_model(attack_box, dataset).to(device) for attack_box in cav_attack_boxes],
                    dim=0,
                )

                match mode:
                    case "spoofing":
                        injected_scores = torch.ones((len(cav_attack_boxes),), dtype=scores.dtype, device=device)
                        injected_is_fake = torch.ones((len(cav_attack_boxes),), dtype=torch.bool, device=device)
                        boxes3d = torch.vstack([boxes3d, target_box_tensors])
                        scores = torch.hstack([scores, injected_scores])
                        is_fake = torch.hstack([is_fake, injected_is_fake])
                    case "removal":
                        keep_mask = AdvCoperceptionLateFusionAttack._compute_removal_keep_mask(
                            boxes3d,
                            target_box_tensors,
                            dataset,
                        )
                        boxes3d = boxes3d[keep_mask]
                        scores = scores[keep_mask]
                        is_fake = is_fake[keep_mask]

                        target_corners = box_utils.boxes_to_corners_3d(
                            target_box_tensors,
                            order=dataset.post_processor.params["order"],
                        )
                        removed_target_corners_list.append(box_utils.project_box3d(target_corners, transformation_matrix))

            if boxes3d.shape[0] == 0:
                continue

            boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=dataset.post_processor.params["order"])
            projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)
            projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
            boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

            pred_box2d_list.append(boxes2d_score)
            pred_box3d_list.append(projected_boxes3d)
            pred_fake_list.append(is_fake)

        if mode == "removal" and removed_target_corners_list:
            advcp_context.removed_box_tensor = torch.vstack(removed_target_corners_list)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            if mode == "removal":
                gt_box_tensor = dataset.post_processor.generate_gt_bbx(batch_data)
                empty_corners = torch.zeros((0, 8, 3), dtype=torch.float32, device=device)
                empty_score = torch.zeros((0,), dtype=torch.float32, device=device)
                return empty_corners, empty_score, gt_box_tensor, advcp_context
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

        if mode == "spoofing" and torch.any(pred_is_fake_tensor):
            advcp_context.fake_box_tensor = pred_box3d_tensor[pred_is_fake_tensor]  # noqa: DC05

        return pred_box3d_tensor, pred_score, gt_box_tensor, advcp_context

    @staticmethod
    def _compute_removal_keep_mask(
        boxes3d: torch.Tensor,
        target_boxes: torch.Tensor,
        dataset: Any,
        iou_threshold: float = REMOVAL_IOU_THRESHOLD,
    ) -> torch.Tensor:
        if boxes3d.shape[0] == 0:
            return torch.ones((0,), dtype=torch.bool, device=boxes3d.device)

        boxes_lwh = AdvCoperceptionIntermediateFusionAttack._model_boxes_to_lwh(boxes3d, dataset)
        targets_lwh = AdvCoperceptionIntermediateFusionAttack._model_boxes_to_lwh(target_boxes, dataset)

        keep_mask = torch.ones((boxes3d.shape[0],), dtype=torch.bool, device=boxes3d.device)
        for target_lwh in targets_lwh:
            ious = AdvCoperceptionIntermediateFusionAttack._compute_iou_weights(boxes_lwh, target_lwh)
            keep_mask = keep_mask & (ious < iou_threshold)
        return keep_mask

    @staticmethod
    def _run_default_prediction(
        batch_data: Any,
        output_dict: OrderedDict[str, Any],
        dataset: Any,
        advcp_context: AdvCPVisualizationContext,
    ) -> AdvCPAttackResult:
        pred_box_tensor, pred_score = dataset.post_processor.post_process(batch_data, output_dict)
        gt_box_tensor = dataset.post_processor.generate_gt_bbx(batch_data)
        return pred_box_tensor, pred_score, gt_box_tensor, advcp_context

    @staticmethod
    def resolve_spoof_boxes(
        advcp_config: AdvCPConfig,
        memory_data: AdvCPMemoryData | None,
    ) -> tuple[str | None, list[npt.NDArray]]:
        return AdvCPAttackHelper.resolve_spoof_boxes(advcp_config, memory_data)

    @staticmethod
    def resolve_spoof_boxes_by_attacker(
        advcp_config: AdvCPConfig,
        memory_data: AdvCPMemoryData | None,
    ) -> tuple[list[str], dict[str, list[npt.NDArray]]]:
        return AdvCPAttackHelper.resolve_spoof_boxes_by_attacker(advcp_config, memory_data)
