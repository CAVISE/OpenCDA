import torch
import random

from AIM.models.mtp.learning.learning_src.data_scripts.data_config import (
    ALLIGN_INITIAL_DIRECTION_TO_X,
    NORMALIZE_DATA,
    NUM_PREDICT,
    ZSCORE_NORMALIZE,
    PRED_LEN,
    NUM_PREDICT_ON_PREDICT,
    PREDICT_VECTOR_SIZE,
)
from AIM.models.mtp.learning.learning_src.data_scripts.preprocess_utils import (
    rotation_matrix_back_with_allign_to_X,
    rotation_matrix_with_allign_to_X,
    rotation_matrix_back_with_allign_to_Y,
    rotation_matrix_with_allign_to_Y,
    normalize_yaw,
    adjust_future_yaw_delta,
    denormalize_yaw,
    z_scrore_denormalize,
    z_scrore_normalize,
)
from .train_utils import my_get_speed, my_get_yaw


def transformer_train_one_epoch(
    model,
    device,
    data_loader,
    optimizer,
    step_weights_factor,
    dist_threshold,
    collision_penalty,
    collision_penalty_factor,
    epoch,
    epochs,
    yaw_keys,
    yaw_values,
    y_x_mean,
    y_x_std,
    y_y_mean,
    y_y_std,
    start_prediction_time=0.2,
):
    """Performs an epoch of GNN model training with rollout"""
    step_weights = torch.ones(PRED_LEN, device=device)
    step_weights[:5] *= step_weights_factor
    step_weights[0] *= step_weights_factor
    step_weights = step_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, PRED_LEN)

    model.train()
    total_loss = 0

    for batch in data_loader:
        # x_batch: [vehicle, 15]: [x_0, y_0, speed_0, yaw_0, cos yaw_0, sin yaw_0, cos yaw_start, sin yaw_start,  intent, intent, intent, start_pos, start_pos, start_pos, start_pos]
        x_batch, y_batch, weights_batch, attn_mask_batch, map_infos_batch, map_attn_mask_batch, map_boundaries = batch
        x_batch, y_batch, weights_batch, attn_mask_batch, map_infos_batch, map_attn_mask_batch, map_boundaries = (
            x_batch.to(device),
            y_batch.to(device),
            weights_batch.to(device),
            attn_mask_batch.to(device),
            map_infos_batch.to(device),
            map_attn_mask_batch.to(device),
            map_boundaries.to(device),
        )
        map_boundaries = map_boundaries.unsqueeze(-1)

        out_coords = None
        dout_coords = None

        on_predictions = False
        idx = 0
        batch_loss = 0

        while idx < (NUM_PREDICT_ON_PREDICT + 1):
            if (
                random.random() < epoch / epochs and epoch / epochs > start_prediction_time + (idx / NUM_PREDICT_ON_PREDICT) * 0.4
            ):  # max of (idx / NUM_PREDICT_ON_PREDICT / 2) = 0.5
                on_predictions = True
            else:
                on_predictions = False

            optimizer.zero_grad()

            if idx == 0:
                yaws = x_batch[:, :, 3]
                x = x_batch[:, :, [0, 1, 2, 3, 4, 5]]
            else:
                yaws = my_get_yaw(
                    x_batch[:, :, 6],
                    x_batch[:, :, 7],
                    x_batch[:, :, 8],
                    x_batch[:, :, 9],
                    out_coords[:, :, 0, :].detach(),
                    yaw_keys,
                    yaw_values,
                    map_boundaries,
                )
                speed = my_get_speed(dout_coords[:, :, 0, 0].detach(), dout_coords[:, :, 0, 1].detach())

                dstart_yaw = yaws - torch.arctan2(x_batch[:, :, 7], x_batch[:, :, 6])
                dlast_yaw = torch.arctan2(x_batch[:, :, 9], x_batch[:, :, 8]) - yaws
                adjust_future_yaw_delta(dstart_yaw)
                adjust_future_yaw_delta(dlast_yaw)

                if NORMALIZE_DATA:
                    normalize_yaw(dstart_yaw)
                    normalize_yaw(dlast_yaw)

                x = torch.cat(
                    (
                        out_coords[:, :, 0, :].detach(),
                        speed.unsqueeze(-1),
                        yaws.unsqueeze(-1),
                        dstart_yaw.unsqueeze(-1),
                        dlast_yaw.unsqueeze(-1),
                    ),
                    dim=-1,
                )

            dout_coords = model(x, map_infos_batch, attn_mask_batch, map_attn_mask_batch)
            dout_coords = dout_coords.reshape(
                dout_coords.shape[0], dout_coords.shape[1], PRED_LEN, PREDICT_VECTOR_SIZE
            )  # [b, v, PRED_LEN, PREDICT_VECTOR_SIZE]

            yaw_cur = yaws.detach().clone()
            yaw_base = x_batch[:, :, 3].detach().clone()
            if NORMALIZE_DATA:
                denormalize_yaw(yaw_cur)
                denormalize_yaw(yaw_base)

            # [x, y, v, yaw, acc, steering]
            _dgt_coords = y_batch.reshape(y_batch.shape[0], y_batch.shape[1], NUM_PREDICT, 6)[:, :, :, [0, 1]]
            dgt_coords = _dgt_coords.clone()

            if ALLIGN_INITIAL_DIRECTION_TO_X:
                rotations_back_current = rotation_matrix_back_with_allign_to_X(yaw_cur).to(device)
            else:
                rotations_back_current = rotation_matrix_back_with_allign_to_Y(yaw_cur).to(device)

            if idx > 0:
                if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                    z_scrore_denormalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                    z_scrore_denormalize(dgt_coords[..., 1], y_y_mean, y_y_std)

                if ALLIGN_INITIAL_DIRECTION_TO_X:
                    rotations_back_base = rotation_matrix_back_with_allign_to_X(yaw_base).to(device)
                    rotation_current = rotation_matrix_with_allign_to_X(yaw_cur).to(device)
                else:
                    rotations_back_base = rotation_matrix_back_with_allign_to_Y(yaw_base).to(device)
                    rotation_current = rotation_matrix_with_allign_to_Y(yaw_cur).to(device)

                dgt_coords = dgt_coords.permute(0, 1, 3, 2)  # [b, v, 2, NUM_PREDICT]
                dgt_coords = rotations_back_base @ dgt_coords
                dgt_coords = dgt_coords + (x_batch[:, :, [0, 1]].unsqueeze(-2) - x[:, :, [0, 1]].unsqueeze(-2)).permute(0, 1, 3, 2)
                dgt_coords = (rotation_current @ dgt_coords).permute(0, 1, 3, 2)

                # Renormalize back to z-score space for loss computation
                if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                    z_scrore_normalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                    z_scrore_normalize(dgt_coords[..., 1], y_y_mean, y_y_std)
            dgt_coords = dgt_coords[:, :, idx : PRED_LEN + idx, :]  # [b, v, PRED_LEN, 2]

            # Both dgt_coords and dout_coords are in z-score normalized space for loss computation
            error = (((dgt_coords - dout_coords) * attn_mask_batch[:, 0].unsqueeze(-1).unsqueeze(-1)).square().sum(-1) * step_weights).sum(-1)
            loss = (weights_batch * error)[attn_mask_batch[:, 0]].nanmean()

            # Denormalize dout_coords from z-score for use in next iteration (now in min-max normalized space)
            if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                z_scrore_denormalize(dout_coords[..., 0], y_x_mean, y_x_std)
                z_scrore_denormalize(dout_coords[..., 1], y_y_mean, y_y_std)

            # get back from deltas to plain coordinates in predictions
            dout_coords = dout_coords.permute(0, 1, 3, 2)  # [b, v, 2, PRED_LEN]
            dout_coords = (rotations_back_current @ dout_coords).permute(0, 1, 3, 2)  # [b, v, PRED_LEN, 2]
            out_coords = dout_coords + x[:, :, [0, 1]].unsqueeze(2)

            if collision_penalty:
                distances = torch.linalg.norm(out_coords.unsqueeze(2) - out_coords.unsqueeze(1), dim=-1)  # (b, v, v, n)

                tri_mask = torch.tril(
                    torch.ones(attn_mask_batch.shape[1], attn_mask_batch.shape[1], dtype=torch.bool, device=attn_mask_batch.device), diagonal=-1
                ).unsqueeze(0)  # (1, v, v)
                mask = (attn_mask_batch & tri_mask).unsqueeze(-1)  # (b, v, v, 1)
                collition_mask = (distances < dist_threshold) & mask  # (b, v, v, n)

                if torch.sum(collition_mask):  # there are can be no cars with small distanses so it will be empty
                    _collision_penalty = (dist_threshold - distances)[collition_mask].square().mean()
                    loss += _collision_penalty * collision_penalty_factor

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss += loss.item()
            idx += 1

            if not on_predictions:
                break

        total_loss += batch_loss / idx
    return total_loss / len(data_loader)


def transformer_evaluate(
    model,
    device,
    data_loader,
    step_weights_factor,
    dist_threshold,
    mr_threshold,
    collision_penalty_factor,
    epoch,
    epochs,
    yaw_keys,
    yaw_values,
    y_x_mean,
    y_x_std,
    y_y_mean,
    y_y_std,
    start_prediction_time=0.2,
):
    """Performs an epoch of model training."""
    step_weights = torch.ones(PRED_LEN, device=device)
    step_weights[:5] *= step_weights_factor
    step_weights[0] *= step_weights_factor
    step_weights = step_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, PRED_LEN)

    model.eval()
    ade, fde, mr = [], [], []
    n_edge, n_collision = [], []
    val_losses, collision_penalties = [], []

    with torch.no_grad():
        for batch in data_loader:
            # x_batch: [vehicle, 15]: [x_0, y_0, speed_0, yaw_0, cos yaw_0, sin yaw_0, cos yaw_start, sin yaw_start,  intent, intent, intent, start_pos, start_pos, start_pos, start_pos]
            x_batch, y_batch, weights_batch, attn_mask_batch, map_infos_batch, map_attn_mask_batch, map_boundaries = batch
            x_batch, y_batch, weights_batch, attn_mask_batch, map_infos_batch, map_attn_mask_batch, map_boundaries = (
                x_batch.to(device),
                y_batch.to(device),
                weights_batch.to(device),
                attn_mask_batch.to(device),
                map_infos_batch.to(device),
                map_attn_mask_batch.to(device),
                map_boundaries.to(device),
            )
            map_boundaries = map_boundaries.unsqueeze(-1)

            out_coords = None
            dout_coords = None
            on_predictions = False
            idx = 0

            while idx < (NUM_PREDICT_ON_PREDICT + 1):
                if (
                    random.random() < epoch / epochs and epoch / epochs > start_prediction_time + (idx / NUM_PREDICT_ON_PREDICT) * 0.4
                ):  # max of (idx / NUM_PREDICT_ON_PREDICT / 2) = 0.5
                    on_predictions = True
                else:
                    on_predictions = False

                if idx == 0:
                    yaws = x_batch[:, :, 3]
                    x = x_batch[:, :, [0, 1, 2, 3, 4, 5]]
                else:
                    yaws = my_get_yaw(
                        x_batch[:, :, 6],
                        x_batch[:, :, 7],
                        x_batch[:, :, 8],
                        x_batch[:, :, 9],
                        out_coords[:, :, 0, :].detach(),
                        yaw_keys,
                        yaw_values,
                        map_boundaries,
                    )
                    speed = my_get_speed(dout_coords[:, :, 0, 0].detach(), dout_coords[:, :, 0, 1].detach())

                    dstart_yaw = yaws - torch.arctan2(x_batch[:, :, 7], x_batch[:, :, 6])
                    dlast_yaw = torch.arctan2(x_batch[:, :, 9], x_batch[:, :, 8]) - yaws
                    adjust_future_yaw_delta(dstart_yaw)
                    adjust_future_yaw_delta(dlast_yaw)

                    if NORMALIZE_DATA:
                        normalize_yaw(dstart_yaw)
                        normalize_yaw(dlast_yaw)

                    x = torch.cat(
                        (
                            out_coords[:, :, 0, :].detach(),
                            speed.unsqueeze(-1),
                            yaws.unsqueeze(-1),
                            dstart_yaw.unsqueeze(-1),
                            dlast_yaw.unsqueeze(-1),
                        ),
                        dim=-1,
                    )

                dout_coords = model(x, map_infos_batch, attn_mask_batch, map_attn_mask_batch)
                dout_coords = dout_coords.reshape(
                    dout_coords.shape[0], dout_coords.shape[1], PRED_LEN, PREDICT_VECTOR_SIZE
                )  # [b, v, PRED_LEN, PREDICT_VECTOR_SIZE]

                yaw_cur = yaws.clone()
                yaw_base = x_batch[:, :, 3].clone()
                if NORMALIZE_DATA:
                    denormalize_yaw(yaw_cur)
                    denormalize_yaw(yaw_base)

                # [x, y, v, yaw, acc, steering]
                _dgt_coords = y_batch.reshape(y_batch.shape[0], y_batch.shape[1], NUM_PREDICT, 6)[:, :, :, [0, 1]]
                dgt_coords = _dgt_coords.clone()

                if ALLIGN_INITIAL_DIRECTION_TO_X:
                    rotations_back_current = rotation_matrix_back_with_allign_to_X(yaw_cur).to(device)
                else:
                    rotations_back_current = rotation_matrix_back_with_allign_to_Y(yaw_cur).to(device)

                if idx > 0:
                    if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                        z_scrore_denormalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                        z_scrore_denormalize(dgt_coords[..., 1], y_y_mean, y_y_std)

                    if ALLIGN_INITIAL_DIRECTION_TO_X:
                        rotations_back_base = rotation_matrix_back_with_allign_to_X(yaw_base).to(device)
                        rotation_current = rotation_matrix_with_allign_to_X(yaw_cur).to(device)
                    else:
                        rotations_back_base = rotation_matrix_back_with_allign_to_Y(yaw_base).to(device)
                        rotation_current = rotation_matrix_with_allign_to_Y(yaw_cur).to(device)

                    dgt_coords = dgt_coords.permute(0, 1, 3, 2)  # [b, v, 2, NUM_PREDICT]
                    dgt_coords = rotations_back_base @ dgt_coords
                    dgt_coords = dgt_coords + (x_batch[:, :, [0, 1]].unsqueeze(-2) - x[:, :, [0, 1]].unsqueeze(-2)).permute(0, 1, 3, 2)
                    dgt_coords = (rotation_current @ dgt_coords).permute(0, 1, 3, 2)

                    # Renormalize back to z-score space for loss computation
                    if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                        z_scrore_normalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                        z_scrore_normalize(dgt_coords[..., 1], y_y_mean, y_y_std)
                dgt_coords = dgt_coords[:, :, idx : PRED_LEN + idx, :]  # [b, v, PRED_LEN, 2]

                # Both dgt_coords and dout_coords are in z-score normalized space for loss computation
                _error = ((dgt_coords - dout_coords) * attn_mask_batch[:, 0].unsqueeze(-1).unsqueeze(-1)).square().sum(-1)
                _error = (_error * step_weights).sum(-1)
                val_loss = (weights_batch * _error)[attn_mask_batch[:, 0]].nanmean()
                val_losses.append(val_loss)

                # Denormalize dout_coords from z-score for use in next iteration (now in min-max normalized space)
                if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                    z_scrore_denormalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                    z_scrore_denormalize(dgt_coords[..., 1], y_y_mean, y_y_std)
                    z_scrore_denormalize(dout_coords[..., 0], y_x_mean, y_x_std)
                    z_scrore_denormalize(dout_coords[..., 1], y_y_mean, y_y_std)

                error = ((dgt_coords - dout_coords) * attn_mask_batch[:, 0].unsqueeze(-1).unsqueeze(-1)).square().sum(-1) ** 0.5
                fde.append(error[attn_mask_batch[:, 0]][:, -1])
                ade.append(error[attn_mask_batch[:, 0]].mean(dim=-1))
                mr.append(((error[:, :, -1] > mr_threshold) & attn_mask_batch[:, 0]).sum())

                # get back from deltas to plain coordinates in predictions
                dout_coords = dout_coords.permute(0, 1, 3, 2)  # [b, v, 2, PRED_LEN]
                dout_coords = (rotations_back_current @ dout_coords).permute(0, 1, 3, 2)  # [b, v, PRED_LEN, 2]
                out_coords = dout_coords + x[:, :, [0, 1]].unsqueeze(2)

                distances = torch.linalg.norm(out_coords.unsqueeze(2) - out_coords.unsqueeze(1), dim=-1)  # (b, v, v, n)

                tri_mask = torch.tril(
                    torch.ones(attn_mask_batch.shape[1], attn_mask_batch.shape[1], dtype=torch.bool, device=attn_mask_batch.device), diagonal=-1
                ).unsqueeze(0)  # (1, v, v)
                mask = (attn_mask_batch & tri_mask).unsqueeze(-1)  # (b, v, v, 1)
                collition_mask = (distances < dist_threshold) & mask  # (b, v, v, n)

                if torch.sum(collition_mask):  # there are can be no cars with small distanses so it will be empty
                    _collision_penalty = (dist_threshold - distances)[collition_mask].square().mean()
                    collision_penalties.append(_collision_penalty)

                dist = torch.min(distances[mask[:, :, :, 0]], dim=-1)[0]  # (b * v * (v-1), 1)
                n_edge.append(len(dist))
                n_collision.append(torch.sum(collition_mask))

                idx += 1

                if not on_predictions:
                    break

    ade = torch.cat(ade).mean()
    fde = torch.cat(fde)
    mr = (sum(mr) / fde.numel()).item()
    fde = fde.mean()
    collision_rate = sum(n_collision) / sum(n_edge)
    collision_penalties = torch.tensor(collision_penalties)
    collision_penalty = collision_penalties.mean().item() if collision_penalties.numel() > 0 else 0.0

    val_losses = torch.tensor(val_losses).mean()

    return (
        ade.cpu().item(),
        fde.cpu().item(),
        mr,
        collision_rate.cpu(),
        val_losses.cpu().item(),
        collision_penalty,
    )
