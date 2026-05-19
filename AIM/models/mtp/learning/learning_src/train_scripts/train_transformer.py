import torch

from collections import defaultdict
from typing import List, Tuple
from torch.utils.data import DataLoader
import torch.nn.functional as F

from AIM.models.mtp.learning.learning_src.data_scripts.data_config import config
from AIM.models.mtp.learning.learning_src.data_scripts.preprocess_utils import (
    rotation_matrix_back_with_allign_to_X,
    rotation_matrix_with_allign_to_X,
    rotation_matrix_back_with_allign_to_Y,
    rotation_matrix_with_allign_to_Y,
    normalize_yaw,
    adjust_future_yaw_delta,
    denormalize_yaw,
    z_score_denormalize,
    z_score_normalize,
)
from .train_utils import my_get_speed, my_get_yaw_new  # my_get_yaw


def transformer_train_one_epoch(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    step_weights_factor: float,
    dist_threshold: float,
    collision_penalty: bool,
    collision_penalty_factor: float,
    epoch: int,
    epochs: int,
    yaw_keys: torch.Tensor,
    yaw_values: torch.Tensor,
    y_x_mean: float,
    y_x_std: float,
    y_y_mean: float,
    y_y_std: float,
    start_prediction_time: float = 0.2,
    coord_loss_weight_x: float = 1.0,
    coord_loss_weight_y: float = 1.0,
    gt_vel_movement_threshold: float = 0.04,
    vel_loss_weight: float = 0.3,
    cls_loss_weight: float = 0.2,
    predict_movement_threshold: float = 0.2,
) -> float:
    """
    perform one epoch of transformer model training with rollout
    """
    step_weights = torch.ones(config.model.pred_len, device=device)
    step_weights[:5] *= step_weights_factor
    step_weights[0] *= step_weights_factor
    step_weights = step_weights.unsqueeze(0).unsqueeze(0)

    model.train()
    total_loss = 0

    for batch in data_loader:
        (
            x_batch,
            x_global_batch,
            y_batch,
            weights_batch,
            attn_mask_batch,
            map_infos_batch,
            map_infos_graph_batch,
            map_attn_mask_batch,
            map_boundaries,
            map_lane_reprs,
            map_lane_repr_masks,
        ) = batch
        x_batch = x_batch.to(device, non_blocking=True)
        x_global_batch = x_global_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        weights_batch = weights_batch.to(device, non_blocking=True)
        attn_mask_batch = attn_mask_batch.to(device, non_blocking=True)
        map_infos_batch = map_infos_batch.to(device, non_blocking=True)
        map_infos_graph_batch = map_infos_graph_batch.to(device, non_blocking=True)

        map_attn_mask_batch = map_attn_mask_batch.to(device, non_blocking=True)
        map_boundaries = map_boundaries.to(device, non_blocking=True).unsqueeze(-1)
        map_lane_reprs = map_lane_reprs.to(device, non_blocking=True)
        map_lane_repr_masks = map_lane_repr_masks.to(device, non_blocking=True)

        attn_mask_batch = attn_mask_batch.unsqueeze(1)
        attn_mask_batch = attn_mask_batch.expand(
            attn_mask_batch.shape[0], attn_mask_batch.shape[2], attn_mask_batch.shape[2], attn_mask_batch.shape[3]
        ).contiguous()

        map_attn_mask_batch = map_attn_mask_batch.unsqueeze(1)
        map_attn_mask_batch = map_attn_mask_batch.expand(
            map_attn_mask_batch.shape[0], attn_mask_batch.shape[1], map_attn_mask_batch.shape[2]
        ).contiguous()

        out_coords = None
        dout_coords = None
        on_predictions = False
        idx = 0
        batch_loss = 0

        optimizer.zero_grad()

        while idx < (config.model.num_predict_on_predict + 1):
            if epoch > (idx + 1) * (epochs / (config.model.num_predict_on_predict + 1)):
                on_predictions = True
            else:
                on_predictions = False

            if idx == 0:
                x_global = x_global_batch[:, :, [0, 1, 2, 3, 4, 5]]
                yaws = x_global[:, :, 3]
                x = x_batch[:, :, :, [0, 1, 2, 4, 5, -2, -1]]
            else:
                # yaws = my_get_yaw(
                #     x_global_batch[:, :, 6],
                #     x_global_batch[:, :, 7],
                #     x_global_batch[:, :, 8],
                #     x_global_batch[:, :, 9],
                #     out_coords[:, :, 0, :],
                #     yaw_keys,
                #     yaw_values,
                #     map_boundaries,
                # )
                yaws = my_get_yaw_new(
                    x_global_batch[:, :, 6],
                    x_global_batch[:, :, 7],
                    x_global_batch[:, :, 8],
                    x_global_batch[:, :, 9],
                    out_coords[:, :, 0, :],
                    map_lane_reprs,
                    map_lane_repr_masks,
                )
                yaws_denormalized = yaws.detach().clone()
                if config.data_processing.normalize_data:
                    denormalize_yaw(yaws_denormalized)

                speed = my_get_speed(dout_coords[:, :, 0, 0].detach(), dout_coords[:, :, 0, 1].detach())

                dstart_yaw = yaws_denormalized - torch.arctan2(x_global_batch[:, :, 7], x_global_batch[:, :, 6])
                dlast_yaw = torch.arctan2(x_global_batch[:, :, 9], x_global_batch[:, :, 8]) - yaws_denormalized
                adjust_future_yaw_delta(dstart_yaw)
                adjust_future_yaw_delta(dlast_yaw)

                if config.data_processing.normalize_data:
                    normalize_yaw(dstart_yaw)
                    normalize_yaw(dlast_yaw)

                x_global = torch.cat(
                    (
                        out_coords[:, :, 0, :].detach(),
                        speed.unsqueeze(-1),
                        yaws.unsqueeze(-1),
                        dstart_yaw.unsqueeze(-1),
                        dlast_yaw.unsqueeze(-1),
                    ),
                    dim=-1,
                )
                x_data_yaw = x_global.unsqueeze(1)[:, :, :, 3:4] - x_global.unsqueeze(2)[:, :, :, 3:4]
                x_data_coords = x_global.unsqueeze(1)[:, :, :, :2] - x_global.unsqueeze(2)[:, :, :, :2]
                yaw_for_rotation = x_global[:, :, 3:4].clone()
                if config.data_processing.normalize_data:
                    denormalize_yaw(yaw_for_rotation)

                if config.data_processing.align_initial_direction_to_x:
                    rotation_matrixes = rotation_matrix_with_allign_to_X(yaw_for_rotation)
                else:
                    rotation_matrixes = rotation_matrix_with_allign_to_Y(yaw_for_rotation)
                x_data_coords = torch.matmul(rotation_matrixes, x_data_coords.unsqueeze(-1)).squeeze(-1)

                x_data_yaw_denormed = x_data_yaw.clone()
                if config.data_processing.normalize_data:
                    denormalize_yaw(x_data_yaw_denormed)

                x_data_speed = x_global.unsqueeze(1)[:, :, :, 2:3].repeat(1, x_global.shape[1], 1, 1)
                x_data_skip = x_global.unsqueeze(1)[:, :, :, 4:].repeat(1, x_global.shape[1], 1, 1)
                x = torch.cat([x_data_coords, x_data_speed, x_data_skip, torch.cos(x_data_yaw_denormed), torch.sin(x_data_yaw_denormed)], dim=-1)

            yaw_cur = yaws.detach().clone()
            yaw_base = x_global_batch[:, :, 3].detach().clone()
            if config.data_processing.normalize_data:
                denormalize_yaw(yaw_cur)
                denormalize_yaw(yaw_base)

            map_infos_graph_x = map_infos_graph_batch["dot"].x
            old_map_infos_shape = (map_infos_graph_x.shape[0], config.object_map.object_vector_size)
            new_map_infos_shape = (
                x_batch.shape[0],
                x_batch.shape[1],
                int(map_infos_graph_x.shape[0] / (x_batch.shape[0] * x_batch.shape[1])),
                config.object_map.object_vector_size,
            )

            map_infos_coords = map_infos_batch.unsqueeze(1)[..., [0, 1]] - x_global.unsqueeze(2)[..., [0, 1]]
            if config.data_processing.align_initial_direction_to_x:
                map_rotation = rotation_matrix_with_allign_to_X(yaw_cur.unsqueeze(2))
            else:
                map_rotation = rotation_matrix_with_allign_to_Y(yaw_cur.unsqueeze(2))

            map_infos_coords = torch.matmul(map_rotation, map_infos_coords.unsqueeze(-1)).squeeze(-1)
            map_infos_directions = torch.matmul(map_rotation, map_infos_batch.unsqueeze(1)[..., [2, 3]].unsqueeze(-1)).squeeze(-1)
            map_infos_graph_x = torch.cat(
                [map_infos_coords, map_infos_directions, map_infos_batch.unsqueeze(1).repeat(1, map_infos_coords.shape[1], 1, 1)[..., 4:5]], dim=-1
            )
            map_infos_graph_batch["dot"].x = map_infos_graph_x.view(old_map_infos_shape)

            movement_logits, dout_coords = model(x, map_infos_graph_batch, attn_mask_batch, map_attn_mask_batch, new_map_infos_shape)
            dout_coords = dout_coords.reshape(dout_coords.shape[0], dout_coords.shape[1], config.model.pred_len, config.model.predict_vector_size)

            _dgt_coords = y_batch.reshape(y_batch.shape[0], y_batch.shape[1], config.model.num_predict, 4)[:, :, :, [0, 1]]
            dgt_coords = _dgt_coords.clone()

            if config.data_processing.align_initial_direction_to_x:
                rotations_back_current = rotation_matrix_back_with_allign_to_X(yaw_cur).to(device)
            else:
                rotations_back_current = rotation_matrix_back_with_allign_to_Y(yaw_cur).to(device)

            if idx > 0:
                if config.data_processing.normalize_data and config.data_processing.zscore_normalize:
                    z_score_denormalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                    z_score_denormalize(dgt_coords[..., 1], y_y_mean, y_y_std)

                if config.data_processing.align_initial_direction_to_x:
                    rotations_back_base = rotation_matrix_back_with_allign_to_X(yaw_base).to(device)
                    rotation_current = rotation_matrix_with_allign_to_X(yaw_cur).to(device)
                else:
                    rotations_back_base = rotation_matrix_back_with_allign_to_Y(yaw_base).to(device)
                    rotation_current = rotation_matrix_with_allign_to_Y(yaw_cur).to(device)

                dgt_coords = dgt_coords.permute(0, 1, 3, 2)
                dgt_coords = rotations_back_base @ dgt_coords
                dgt_coords = dgt_coords + (x_global_batch[:, :, [0, 1]].unsqueeze(-2) - x_global[:, :, [0, 1]].unsqueeze(-2)).permute(0, 1, 3, 2)
                dgt_coords = (rotation_current @ dgt_coords).permute(0, 1, 3, 2)

                if config.data_processing.normalize_data and config.data_processing.zscore_normalize:
                    z_score_normalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                    z_score_normalize(dgt_coords[..., 1], y_y_mean, y_y_std)

            dgt_coords = dgt_coords[:, :, idx : config.model.pred_len + idx, :]

            w = torch.tensor([coord_loss_weight_x, coord_loss_weight_y], device=dgt_coords.device, dtype=dgt_coords.dtype).view(1, 1, 1, 2)
            masked = dgt_coords - dout_coords
            per_t = (masked.square() * w).sum(dim=-1)

            if config.data_processing.normalize_data and config.data_processing.zscore_normalize:
                z_score_denormalize(dout_coords[..., 0], y_x_mean, y_x_std)
                z_score_denormalize(dout_coords[..., 1], y_y_mean, y_y_std)
                z_score_denormalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                z_score_denormalize(dgt_coords[..., 1], y_y_mean, y_y_std)

            pred_vel = my_get_speed(dout_coords[..., 0], dout_coords[..., 1])
            gt_vel = my_get_speed(dgt_coords[..., 0], dgt_coords[..., 1])
            vel_loss = F.mse_loss(pred_vel, gt_vel, reduction="none")

            gt_movement_cls = (gt_vel > gt_vel_movement_threshold).float()
            cls_loss = F.binary_cross_entropy_with_logits(movement_logits, gt_movement_cls, reduction="none")

            per_t = per_t + vel_loss_weight * vel_loss + cls_loss_weight * cls_loss

            error = (per_t * step_weights).sum(-1)
            loss = (weights_batch * error)[attn_mask_batch[:, 0, 0]].nanmean()

            dout_coords = dout_coords.permute(0, 1, 3, 2)
            dout_coords = (rotations_back_current @ dout_coords).permute(0, 1, 3, 2)
            out_coords = dout_coords + x_global[:, :, [0, 1]].unsqueeze(2)

            if collision_penalty:
                distances = torch.linalg.norm(out_coords.unsqueeze(2) - out_coords.unsqueeze(1), dim=-1)

                tri_mask = torch.tril(
                    torch.ones(attn_mask_batch.shape[1], attn_mask_batch.shape[1], dtype=torch.bool, device=attn_mask_batch.device), diagonal=-1
                ).unsqueeze(0)
                mask = (attn_mask_batch[:, 0] & tri_mask).unsqueeze(-1)
                collition_mask = (distances < dist_threshold) & mask

                if collition_mask.any():
                    _collision_penalty = (dist_threshold - distances)[collition_mask].square().mean()
                    loss += _collision_penalty * collision_penalty_factor

            loss.backward()
            batch_loss += loss.item()
            idx += 1

            if not on_predictions:
                break

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss / idx

    return total_loss / len(data_loader)


def transformer_evaluate(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    step_weights_factor: float,
    dist_threshold: float,
    mr_threshold: float,
    collision_penalty_factor: float,
    epoch: int,
    epochs: int,
    yaw_keys: torch.Tensor,
    yaw_values: torch.Tensor,
    y_x_mean: float,
    y_x_std: float,
    y_y_mean: float,
    y_y_std: float,
    start_prediction_time: float = 0.2,
    coord_loss_weight_x: float = 1.0,
    coord_loss_weight_y: float = 1.0,
    gt_vel_movement_threshold: float = 0.04,
    vel_loss_weight: float = 0.3,
    cls_loss_weight: float = 0.2,
    predict_movement_threshold: float = 0.2,
) -> Tuple[float, float, float, float, float, float, List[Tuple[int, int, float, float]]]:
    """
    evaluate transformer model on validation data
    """
    step_weights = torch.ones(config.model.pred_len, device=device)
    step_weights[:5] *= step_weights_factor
    step_weights[0] *= step_weights_factor
    step_weights = step_weights.unsqueeze(0).unsqueeze(0)

    model.eval()
    ade, fde, mr = [], [], []
    fde_by_step: defaultdict[int, list] = defaultdict(list)
    collition_penalty_by_step: defaultdict[int, list] = defaultdict(list)
    n_edge, n_collision = [], []
    val_losses, collision_penalties = [], []

    with torch.no_grad():
        for batch in data_loader:
            (
                x_batch,
                x_global_batch,
                y_batch,
                weights_batch,
                attn_mask_batch,
                map_infos_batch,
                map_infos_graph_batch,
                map_attn_mask_batch,
                map_boundaries,
                map_lane_reprs,
                map_lane_repr_masks,
            ) = batch
            x_batch = x_batch.to(device, non_blocking=True)
            x_global_batch = x_global_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            weights_batch = weights_batch.to(device, non_blocking=True)
            attn_mask_batch = attn_mask_batch.to(device, non_blocking=True)
            map_infos_batch = map_infos_batch.to(device, non_blocking=True)
            map_infos_graph_batch = map_infos_graph_batch.to(device, non_blocking=True)

            map_attn_mask_batch = map_attn_mask_batch.to(device, non_blocking=True)
            map_boundaries = map_boundaries.to(device, non_blocking=True).unsqueeze(-1)
            map_lane_reprs = map_lane_reprs.to(device, non_blocking=True)
            map_lane_repr_masks = map_lane_repr_masks.to(device, non_blocking=True)

            attn_mask_batch = attn_mask_batch.unsqueeze(1)
            attn_mask_batch = attn_mask_batch.expand(
                attn_mask_batch.shape[0], attn_mask_batch.shape[2], attn_mask_batch.shape[2], attn_mask_batch.shape[3]
            ).contiguous()

            map_attn_mask_batch = map_attn_mask_batch.unsqueeze(1)
            map_attn_mask_batch = map_attn_mask_batch.expand(
                map_attn_mask_batch.shape[0], attn_mask_batch.shape[1], map_attn_mask_batch.shape[2]
            ).contiguous()

            out_coords = None
            dout_coords = None
            on_predictions = False
            idx = 0

            while idx < (config.model.num_predict_on_predict + 1):
                if epoch > (idx + 1) * (epochs / (config.model.num_predict_on_predict + 1)):
                    on_predictions = True
                else:
                    on_predictions = False

                if idx == 0:
                    x_global = x_global_batch[:, :, [0, 1, 2, 3, 4, 5]]
                    yaws = x_global[:, :, 3]
                    x = x_batch[:, :, :, [0, 1, 2, 4, 5, -2, -1]]
                else:
                    # yaws = my_get_yaw(
                    #     x_global_batch[:, :, 6],
                    #     x_global_batch[:, :, 7],
                    #     x_global_batch[:, :, 8],
                    #     x_global_batch[:, :, 9],
                    #     out_coords[:, :, 0, :].detach(),
                    #     yaw_keys,
                    #     yaw_values,
                    #     map_boundaries,
                    # )
                    yaws = my_get_yaw_new(
                        x_global_batch[:, :, 6],
                        x_global_batch[:, :, 7],
                        x_global_batch[:, :, 8],
                        x_global_batch[:, :, 9],
                        out_coords[:, :, 0, :],
                        map_lane_reprs,
                        map_lane_repr_masks,
                    )
                    yaws_denormalized = yaws.detach().clone()
                    if config.data_processing.normalize_data:
                        denormalize_yaw(yaws_denormalized)

                    speed = my_get_speed(dout_coords[:, :, 0, 0].detach(), dout_coords[:, :, 0, 1].detach())

                    dstart_yaw = yaws_denormalized - torch.arctan2(x_global_batch[:, :, 7], x_global_batch[:, :, 6])
                    dlast_yaw = torch.arctan2(x_global_batch[:, :, 9], x_global_batch[:, :, 8]) - yaws_denormalized
                    adjust_future_yaw_delta(dstart_yaw)
                    adjust_future_yaw_delta(dlast_yaw)

                    if config.data_processing.normalize_data:
                        normalize_yaw(dstart_yaw)
                        normalize_yaw(dlast_yaw)

                    x_global = torch.cat(
                        (
                            out_coords[:, :, 0, :].detach(),
                            speed.unsqueeze(-1),
                            yaws.unsqueeze(-1),
                            dstart_yaw.unsqueeze(-1),
                            dlast_yaw.unsqueeze(-1),
                        ),
                        dim=-1,
                    )
                    x_data_yaw = x_global.unsqueeze(1)[:, :, :, 3:4] - x_global.unsqueeze(2)[:, :, :, 3:4]
                    x_data_coords = x_global.unsqueeze(1)[:, :, :, :2] - x_global.unsqueeze(2)[:, :, :, :2]
                    yaw_for_rotation = x_global[:, :, 3:4].clone()
                    if config.data_processing.normalize_data:
                        denormalize_yaw(yaw_for_rotation)

                    if config.data_processing.align_initial_direction_to_x:
                        rotation_matrixes = rotation_matrix_with_allign_to_X(yaw_for_rotation)
                    else:
                        rotation_matrixes = rotation_matrix_with_allign_to_Y(yaw_for_rotation)
                    x_data_coords = torch.matmul(rotation_matrixes, x_data_coords.unsqueeze(-1)).squeeze(-1)

                    x_data_yaw_denormed = x_data_yaw.clone()
                    if config.data_processing.normalize_data:
                        denormalize_yaw(x_data_yaw_denormed)

                    x_data_speed = x_global.unsqueeze(1)[:, :, :, 2:3].repeat(1, x_global.shape[1], 1, 1)
                    x_data_skip = x_global.unsqueeze(1)[:, :, :, 4:].repeat(1, x_global.shape[1], 1, 1)
                    x = torch.cat([x_data_coords, x_data_speed, x_data_skip, torch.cos(x_data_yaw_denormed), torch.sin(x_data_yaw_denormed)], dim=-1)

                yaw_cur = yaws.clone()
                yaw_base = x_global_batch[:, :, 3].clone()
                if config.data_processing.normalize_data:
                    denormalize_yaw(yaw_cur)
                    denormalize_yaw(yaw_base)

                map_infos_graph_x = map_infos_graph_batch["dot"].x
                old_map_infos_shape = (map_infos_graph_x.shape[0], config.object_map.object_vector_size)
                new_map_infos_shape = (
                    x_batch.shape[0],
                    x_batch.shape[1],
                    int(map_infos_graph_x.shape[0] / (x_batch.shape[0] * x_batch.shape[1])),
                    config.object_map.object_vector_size,
                )

                map_infos_coords = map_infos_batch.unsqueeze(1)[..., [0, 1]] - x_global.unsqueeze(2)[..., [0, 1]]
                if config.data_processing.align_initial_direction_to_x:
                    map_rotation = rotation_matrix_with_allign_to_X(yaw_cur.unsqueeze(2))
                else:
                    map_rotation = rotation_matrix_with_allign_to_Y(yaw_cur.unsqueeze(2))

                map_infos_coords = torch.matmul(map_rotation, map_infos_coords.unsqueeze(-1)).squeeze(-1)
                map_infos_directions = torch.matmul(map_rotation, map_infos_batch.unsqueeze(1)[..., [2, 3]].unsqueeze(-1)).squeeze(-1)
                map_infos_graph_x = torch.cat(
                    [map_infos_coords, map_infos_directions, map_infos_batch.unsqueeze(1).repeat(1, map_infos_coords.shape[1], 1, 1)[..., 4:5]],
                    dim=-1,
                )
                map_infos_graph_batch["dot"].x = map_infos_graph_x.view(old_map_infos_shape)

                movement_logits, dout_coords = model(x, map_infos_graph_batch, attn_mask_batch, map_attn_mask_batch, new_map_infos_shape)
                dout_coords = dout_coords.reshape(dout_coords.shape[0], dout_coords.shape[1], config.model.pred_len, config.model.predict_vector_size)

                _dgt_coords = y_batch.reshape(y_batch.shape[0], y_batch.shape[1], config.model.num_predict, 4)[:, :, :, [0, 1]]
                dgt_coords = _dgt_coords.clone()

                if config.data_processing.align_initial_direction_to_x:
                    rotations_back_current = rotation_matrix_back_with_allign_to_X(yaw_cur).to(device)
                else:
                    rotations_back_current = rotation_matrix_back_with_allign_to_Y(yaw_cur).to(device)

                if idx > 0:
                    if config.data_processing.normalize_data and config.data_processing.zscore_normalize:
                        z_score_denormalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                        z_score_denormalize(dgt_coords[..., 1], y_y_mean, y_y_std)

                    if config.data_processing.align_initial_direction_to_x:
                        rotations_back_base = rotation_matrix_back_with_allign_to_X(yaw_base).to(device)
                        rotation_current = rotation_matrix_with_allign_to_X(yaw_cur).to(device)
                    else:
                        rotations_back_base = rotation_matrix_back_with_allign_to_Y(yaw_base).to(device)
                        rotation_current = rotation_matrix_with_allign_to_Y(yaw_cur).to(device)

                    dgt_coords = dgt_coords.permute(0, 1, 3, 2)
                    dgt_coords = rotations_back_base @ dgt_coords
                    dgt_coords = dgt_coords + (x_global_batch[:, :, [0, 1]].unsqueeze(-2) - x_global[:, :, [0, 1]].unsqueeze(-2)).permute(0, 1, 3, 2)
                    dgt_coords = (rotation_current @ dgt_coords).permute(0, 1, 3, 2)

                    if config.data_processing.normalize_data and config.data_processing.zscore_normalize:
                        z_score_normalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                        z_score_normalize(dgt_coords[..., 1], y_y_mean, y_y_std)

                dgt_coords = dgt_coords[:, :, idx : config.model.pred_len + idx, :]

                w = torch.tensor([coord_loss_weight_x, coord_loss_weight_y], device=dgt_coords.device, dtype=dgt_coords.dtype).view(1, 1, 1, 2)
                masked = dgt_coords - dout_coords
                _per_t = (masked.square() * w).sum(dim=-1)

                if config.data_processing.normalize_data and config.data_processing.zscore_normalize:
                    z_score_denormalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                    z_score_denormalize(dgt_coords[..., 1], y_y_mean, y_y_std)
                    z_score_denormalize(dout_coords[..., 0], y_x_mean, y_x_std)
                    z_score_denormalize(dout_coords[..., 1], y_y_mean, y_y_std)

                pred_vel = my_get_speed(dout_coords[..., 0], dout_coords[..., 1])
                gt_vel = my_get_speed(dgt_coords[..., 0], dgt_coords[..., 1])
                vel_loss = F.mse_loss(pred_vel, gt_vel, reduction="none")

                gt_movement_cls = (gt_vel > gt_vel_movement_threshold).float()
                cls_loss = F.binary_cross_entropy_with_logits(movement_logits, gt_movement_cls, reduction="none")

                _per_t = _per_t + vel_loss_weight * vel_loss + cls_loss_weight * cls_loss

                _error = (_per_t * step_weights).sum(-1)
                val_loss = (weights_batch * _error)[attn_mask_batch[:, 0, 0]].nanmean()
                val_losses.append(val_loss)

                error = ((dgt_coords - dout_coords) * attn_mask_batch[:, 0, 0].unsqueeze(-1).unsqueeze(-1)).square().sum(-1) ** 0.5
                fde_vec = error[attn_mask_batch[:, 0, 0]][:, -1]
                fde.append(fde_vec)
                fde_by_step[idx].append(fde_vec)
                ade.append(error[attn_mask_batch[:, 0, 0]].mean(dim=-1))
                mr.append(((error[:, :, -1] > mr_threshold) & attn_mask_batch[:, 0, 0]).sum())

                dout_coords = dout_coords.permute(0, 1, 3, 2)
                dout_coords = (rotations_back_current @ dout_coords).permute(0, 1, 3, 2)
                out_coords = dout_coords + x_global[:, :, [0, 1]].unsqueeze(2)

                distances = torch.linalg.norm(out_coords.unsqueeze(2) - out_coords.unsqueeze(1), dim=-1)

                tri_mask = torch.tril(
                    torch.ones(attn_mask_batch.shape[1], attn_mask_batch.shape[1], dtype=torch.bool, device=attn_mask_batch.device), diagonal=-1
                ).unsqueeze(0)
                mask = (attn_mask_batch[:, 0] & tri_mask).unsqueeze(-1)
                collition_mask = (distances < dist_threshold) & mask

                if collition_mask.any():
                    _collision_penalty = (dist_threshold - distances)[collition_mask].square().mean()
                    collision_penalty = (_collision_penalty * collision_penalty_factor).unsqueeze(0)
                    collision_penalties.append(collision_penalty)
                    collition_penalty_by_step[idx].append(collision_penalty)

                dist = torch.min(distances[mask[:, :, :, 0]], dim=-1)[0]
                n_edge.append(len(dist))
                n_collision.append(torch.sum(collition_mask))

                idx += 1

                if not on_predictions:
                    break

    ade = torch.cat(ade).mean()
    fde = torch.cat(fde)
    mr = (sum(mr) / fde.numel()).item()
    fde = fde.mean()

    rollout_per_step: List[Tuple[int, int, float, float, float]] = []
    for step_idx in sorted(fde_by_step.keys()):
        t = torch.cat(fde_by_step[step_idx])
        if len(collition_penalty_by_step[step_idx]) != 0:
            collision_penalty = torch.cat(collition_penalty_by_step[step_idx])
        else:
            collision_penalty = torch.zeros((1))

        rollout_per_step.append(
            (step_idx, int(t.numel()), t.mean().item(), (t > mr_threshold).float().mean().item(), collision_penalty.mean().item())
        )

    short = rollout_per_step[:3]
    tail = rollout_per_step[-1:] if len(rollout_per_step) > 5 else []
    if rollout_per_step:
        parts = [f"s{i}:fde={fd:.4f},mr={m:.3f},col={coll:.4f}" for i, n, fd, m, coll in short]
        if tail and rollout_per_step[-1][0] != short[-1][0]:
            parts.append("…")
            parts += [f"s{i}:fde={fd:.4f},mr={m:.3f},col={coll:.4f}" for i, n, fd, m, coll in tail]
        print(f"[val epoch {epoch}] rollout " + " | ".join(parts))

    collision_rate = sum(n_collision) / sum(n_edge)
    collision_penalties = torch.tensor(collision_penalties)
    collision_penalty = collision_penalties.mean().item() if collision_penalties.numel() > 0 else 0.0

    val_losses = torch.tensor(val_losses).mean()

    return (
        ade.cpu().item(),
        fde.cpu().item(),
        mr,
        float(collision_rate),
        val_losses.cpu().item(),
        collision_penalty,
        rollout_per_step,
    )
