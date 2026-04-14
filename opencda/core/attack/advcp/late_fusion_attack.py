from collections import OrderedDict
import logging

import numpy as np
import torch

from opencood.utils import box_utils

from .utils import resolve_late_spoof_boxes

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.late_fusion_attack")


def inference_late_fusion_attack(batch_data, model, dataset, device, advcp_config=None, memory_data=None):
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    mode = (advcp_config or {}).get("mode", "spoof")
    if mode == "remove":
        _raise_late_removal_not_available()

    attacker_id, attack_boxes = resolve_late_spoof_boxes(advcp_config, memory_data)
    if not attack_boxes:
        pred_box_tensor, pred_score = dataset.post_processor.post_process(batch_data, output_dict)
        gt_box_tensor = dataset.post_processor.generate_gt_bbx(batch_data)
        return pred_box_tensor, pred_score, gt_box_tensor

    if attacker_id not in batch_data:
        logger.warning("AdvCP attacker '%s' is not present in the current batch. Falling back to normal late fusion.", attacker_id)
        pred_box_tensor, pred_score = dataset.post_processor.post_process(batch_data, output_dict)
        gt_box_tensor = dataset.post_processor.generate_gt_bbx(batch_data)
        return pred_box_tensor, pred_score, gt_box_tensor

    pred_box3d_list = []
    pred_box2d_list = []

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

        if cav_id == attacker_id:
            injected_boxes = [_convert_box_for_model(attack_box, dataset).to(device) for attack_box in attack_boxes]
            injected_boxes = torch.stack(injected_boxes, dim=0)
            injected_scores = torch.ones((len(attack_boxes),), dtype=scores.dtype, device=device)
            boxes3d = torch.vstack([boxes3d, injected_boxes])
            scores = torch.hstack([scores, injected_scores])

        if boxes3d.shape[0] == 0:
            continue

        boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=dataset.post_processor.params["order"])
        projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)
        projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
        boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

        pred_box2d_list.append(boxes2d_score)
        pred_box3d_list.append(projected_boxes3d)

    if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
        raise RuntimeError("AdvCP late spoofing produced no detection result.")

    pred_box2d_tensor = torch.vstack(pred_box2d_list)
    scores = pred_box2d_tensor[:, -1]
    pred_box3d_tensor = torch.vstack(pred_box3d_list)

    keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
    keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
    keep_index = torch.logical_and(keep_index_1, keep_index_2)
    pred_box3d_tensor = pred_box3d_tensor[keep_index]
    scores = scores[keep_index]

    keep_index = box_utils.nms_rotated(pred_box3d_tensor, scores, dataset.post_processor.params["nms_thresh"])
    pred_box3d_tensor = pred_box3d_tensor[keep_index]
    scores = scores[keep_index]

    mask = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
    pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
    pred_score = scores[mask]
    gt_box_tensor = dataset.post_processor.generate_gt_bbx(batch_data)

    return pred_box3d_tensor, pred_score, gt_box_tensor


def _convert_box_for_model(box_lwh_bottom_center, dataset):
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


def _raise_late_removal_not_available():
    raise NotImplementedError("AdvCP late-fusion removal is not available yet.")
