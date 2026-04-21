from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Any, Optional

import numpy as np
import torch
from opencda.core.attack.advcp.attack_helper import AdvCPAttackHelper
from opencda.core.attack.advcp.types import AdvCPAttackResult, AdvCPVisualizationContext

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.advcp_manager")


class AdvCoperceptionLateFusionAttack:
    @staticmethod
    def run(
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: dict[str, Any],
        memory_data: Optional[dict[Any, Any]] = None,
    ) -> AdvCPAttackResult:
        # TODO: Move this up when https://github.com/CAVISE/OpenCDA/pull/65 is merged
        from opencood.utils import box_utils

        output_dict: OrderedDict[str, Any] = OrderedDict()
        advcp_context: AdvCPVisualizationContext = {"attacker_ids": [], "fake_box_tensor": None, "mode": None}

        for cav_id, cav_content in batch_data.items():
            output_dict[cav_id] = model(cav_content)

        mode = AdvCPAttackHelper.require_config_value(advcp_config, "mode")
        advcp_context["mode"] = mode
        if mode == "remove":
            AdvCoperceptionLateFusionAttack._raise_removal_not_available()

        attacker_id, attack_boxes = AdvCoperceptionLateFusionAttack.resolve_spoof_boxes(advcp_config, memory_data)
        if attacker_id is not None:
            advcp_context["attacker_ids"] = [attacker_id]

        if not attack_boxes:
            return AdvCoperceptionLateFusionAttack._run_default_prediction(batch_data, output_dict, dataset, advcp_context)

        if attacker_id not in batch_data:
            logger.warning(
                "AdvCP attack will not be applied on this tick because attacker '%s' is not present in the current batch. "
                "Continuing with normal cooperative perception inference.",
                attacker_id,
            )
            advcp_context["attacker_ids"] = []
            return AdvCoperceptionLateFusionAttack._run_default_prediction(batch_data, output_dict, dataset, advcp_context)

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
                injected_box_tensors = [AdvCPAttackHelper.convert_box_for_model(attack_box, dataset).to(device) for attack_box in attack_boxes]
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
    def resolve_spoof_boxes(advcp_config: dict[str, Any], memory_data: dict[str, Any] | None) -> tuple[str | None, list[np.ndarray]]:
        return AdvCPAttackHelper.resolve_spoof_boxes(advcp_config, memory_data)

    @staticmethod
    def _raise_removal_not_available() -> None:
        raise NotImplementedError("AdvCP late-fusion removal is not available yet.")
