import torch
import random

from AIM.models.mtp.learning.learning_src.data_scripts.data_config import (
    ALLIGN_INITIAL_DIRECTION_TO_X,
    NORMALIZE_DATA,
    ZSCORE_NORMALIZE,
    NUM_PREDICT,
    PRED_LEN,
    NUM_PREDICT_ON_PREDICT,
    PREDICT_VECTOR_SIZE,
)
from AIM.models.mtp.learning.learning_src.data_scripts.preprocess_utils import (
    rotation_matrix_back_with_allign_to_X,
    rotation_matrix_with_allign_to_X,
    rotation_matrix_back_with_allign_to_Y,
    rotation_matrix_with_allign_to_Y,
    denormalize_yaw,
    normalize_yaw,
    adjust_future_yaw_delta,
    z_scrore_denormalize,
    z_scrore_normalize,
)
from .train_utils import my_get_speed, my_get_yaw


def gnn_train_one_epoch(
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
    map_boundary=100,
):
    """Performs an epoch of GNN model training with rollout"""
    step_weights = torch.ones(PRED_LEN, device=device)
    step_weights[:5] *= step_weights_factor
    step_weights[0] *= step_weights_factor

    model.train()
    total_loss = 0

    for batch in data_loader:
        batch = batch.to(device)
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
                yaws = batch.x[:, 3]
                x = batch.x[:, [0, 1, 2, 3, 4, 5]]
            else:
                yaws = my_get_yaw(
                    batch.x[:, 6],
                    batch.x[:, 7],
                    batch.x[:, 8],
                    batch.x[:, 9],
                    out_coords[:, 0, :].detach(),
                    yaw_keys,
                    yaw_values,
                    map_boundary,
                )
                speed = my_get_speed(dout_coords[:, 0, 0].detach(), dout_coords[:, 0, 1].detach())

                dstart_yaw = yaws - torch.arctan2(batch.x[:, 7], batch.x[:, 6])
                dlast_yaw = torch.arctan2(batch.x[:, 9], batch.x[:, 8]) - yaws
                adjust_future_yaw_delta(dstart_yaw)
                adjust_future_yaw_delta(dlast_yaw)

                if NORMALIZE_DATA:
                    normalize_yaw(dstart_yaw)
                    normalize_yaw(dlast_yaw)

                x = torch.cat(
                    (
                        out_coords[:, 0, :].detach(),
                        speed.unsqueeze(-1),
                        yaws.unsqueeze(-1),
                        dstart_yaw.unsqueeze(-1),
                        dlast_yaw.unsqueeze(-1),
                    ),
                    dim=-1,
                )

            dout_coords = model(x, batch.edge_index)
            dout_coords = dout_coords.reshape(dout_coords.shape[0], PRED_LEN, PREDICT_VECTOR_SIZE)  # [v, PRED_LEN, PREDICT_VECTOR_SIZE]

            yaw_cur = yaws.detach().clone()
            yaw_base = batch.x[:, 3].detach().clone()
            if NORMALIZE_DATA:
                denormalize_yaw(yaw_cur)
                denormalize_yaw(yaw_base)

            # [x, y, v, yaw, acc, steering]
            _dgt_coords = batch.y.reshape(batch.y.shape[0], NUM_PREDICT, 6)[:, :, [0, 1]]
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

                dgt_coords = dgt_coords.permute(0, 2, 1)  # [v, 2, NUM_PREDICT]
                dgt_coords = torch.bmm(rotations_back_base, dgt_coords)
                dgt_coords = dgt_coords + (batch.x[:, [0, 1]].unsqueeze(1) - x[:, [0, 1]].unsqueeze(1)).permute(0, 2, 1)
                dgt_coords = torch.bmm(rotation_current, dgt_coords).permute(0, 2, 1)

                if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                    z_scrore_normalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                    z_scrore_normalize(dgt_coords[..., 1], y_y_mean, y_y_std)

            dgt_coords = dgt_coords[:, idx : PRED_LEN + idx, :]  # [v, PRED_LEN, 2]

            error = ((dgt_coords - dout_coords).square().sum(-1) * step_weights).sum(-1)
            loss = (batch.weights * error).nanmean()

            if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                z_scrore_denormalize(dout_coords[..., 0], y_x_mean, y_x_std)
                z_scrore_denormalize(dout_coords[..., 1], y_y_mean, y_y_std)

            # get back from deltas to plain coordinates in predictions
            dout_coords = dout_coords.permute(0, 2, 1)  # [v, 2, PRED_LEN]
            dout_coords = torch.bmm(rotations_back_current, dout_coords).permute(0, 2, 1)  # [v, PRED_LEN, 2]
            out_coords = dout_coords + x[:, [0, 1]].unsqueeze(1)

            if collision_penalty:
                mask = batch.edge_index[0, :] < batch.edge_index[1, :]
                _edge = batch.edge_index[:, mask].T  # [edge',2]
                dist = torch.linalg.norm(out_coords[_edge[:, 0]] - out_coords[_edge[:, 1]], dim=-1)
                valid = dist < dist_threshold
                dist = dist_threshold - dist[valid]

                if dist.numel():  # there are can be no cars with small distanses so it will be empty
                    _collision_penalty = dist.square().mean()
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


def gnn_evaluate(
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
    map_boundary=100,
):
    """Performs an epoch of model training."""
    step_weights = torch.ones(PRED_LEN, device=device)
    step_weights[:5] *= step_weights_factor
    step_weights[0] *= step_weights_factor

    model.eval()
    ade, fde = [], []
    n_edge, n_collision = [], []
    val_losses, collision_penalties = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
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
                    yaws = batch.x[:, 3]
                    x = batch.x[:, [0, 1, 2, 3, 4, 5]]
                else:
                    yaws = my_get_yaw(
                        batch.x[:, 6],
                        batch.x[:, 7],
                        batch.x[:, 8],
                        batch.x[:, 9],
                        out_coords[:, 0, :],
                        yaw_keys,
                        yaw_values,
                        map_boundary,
                    )
                    speed = my_get_speed(dout_coords[:, 0, 0], dout_coords[:, 0, 1])

                    dstart_yaw = yaws - torch.arctan2(batch.x[:, 7], batch.x[:, 6])
                    dlast_yaw = torch.arctan2(batch.x[:, 9], batch.x[:, 8]) - yaws
                    adjust_future_yaw_delta(dstart_yaw)
                    adjust_future_yaw_delta(dlast_yaw)

                    if NORMALIZE_DATA:
                        normalize_yaw(dstart_yaw)
                        normalize_yaw(dlast_yaw)
                    x = torch.cat(
                        (
                            out_coords[:, 0, :],
                            speed.unsqueeze(-1),
                            yaws.unsqueeze(-1),
                            dstart_yaw.unsqueeze(-1),
                            dlast_yaw.unsqueeze(-1),
                        ),
                        dim=-1,
                    )

                dout_coords = model(x, batch.edge_index)
                dout_coords = dout_coords.reshape(dout_coords.shape[0], PRED_LEN, PREDICT_VECTOR_SIZE)  # [v, PRED_LEN, PREDICT_VECTOR_SIZE]

                yaw_cur = yaws.clone()
                yaw_base = batch.x[:, 3].clone()
                if NORMALIZE_DATA:
                    denormalize_yaw(yaw_cur)
                    denormalize_yaw(yaw_base)

                # [x, y, v, yaw, acc, steering]
                _dgt_coords = batch.y.reshape(batch.y.shape[0], NUM_PREDICT, 6)[:, :, [0, 1]]
                dgt_coords = _dgt_coords.clone()

                if ALLIGN_INITIAL_DIRECTION_TO_X:
                    rotations_back_current = rotation_matrix_back_with_allign_to_X(yaw_cur).to(device)
                else:
                    rotations_back_current = rotation_matrix_back_with_allign_to_Y(yaw_cur).to(device)

                if idx > 0:
                    # For idx > 0, we need to transform dgt_coords from base frame to current frame
                    # dgt_coords is in z-score normalized space, but batch.x and x are in min-max normalized space
                    # We need to denormalize dgt_coords from z-score first, do transformation, then renormalize
                    if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                        z_scrore_denormalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                        z_scrore_denormalize(dgt_coords[..., 1], y_y_mean, y_y_std)

                    if ALLIGN_INITIAL_DIRECTION_TO_X:
                        rotations_back_base = rotation_matrix_back_with_allign_to_X(yaw_base).to(device)
                        rotation_current = rotation_matrix_with_allign_to_X(yaw_cur).to(device)
                    else:
                        rotations_back_base = rotation_matrix_back_with_allign_to_Y(yaw_base).to(device)
                        rotation_current = rotation_matrix_with_allign_to_Y(yaw_cur).to(device)

                    dgt_coords = dgt_coords.permute(0, 2, 1)  # [v, 2, NUM_PREDICT]
                    dgt_coords = torch.bmm(rotations_back_base, dgt_coords)
                    dgt_coords = dgt_coords + (batch.x[:, [0, 1]].unsqueeze(1) - x[:, [0, 1]].unsqueeze(1)).permute(0, 2, 1)
                    dgt_coords = torch.bmm(rotation_current, dgt_coords).permute(0, 2, 1)

                    # Renormalize back to z-score space for loss computation
                    if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                        z_scrore_normalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                        z_scrore_normalize(dgt_coords[..., 1], y_y_mean, y_y_std)
                dgt_coords = dgt_coords[:, idx : PRED_LEN + idx, :]  # [v, PRED_LEN, 2]

                _error = (dgt_coords - dout_coords).square().sum(-1)
                error = _error**0.5
                _error = (_error * step_weights).sum(-1)

                val_loss = (batch.weights * _error).nanmean()

                val_losses.append(val_loss)

                # Denormalize dgt_coords and dout_coords from z-score for use in next iteration (now in min-max normalized space)
                if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                    z_scrore_denormalize(dgt_coords[..., 0], y_x_mean, y_x_std)
                    z_scrore_denormalize(dgt_coords[..., 1], y_y_mean, y_y_std)
                    z_scrore_denormalize(dout_coords[..., 0], y_x_mean, y_x_std)
                    z_scrore_denormalize(dout_coords[..., 1], y_y_mean, y_y_std)

                error = ((dgt_coords - dout_coords).square().sum(-1)) ** 0.5
                fde.append(error[:, -1])
                ade.append(error.mean(dim=-1))

                # get back from deltas to plain coordinates in predictions
                dout_coords = dout_coords.permute(0, 2, 1)  # [v, 2, PRED_LEN]
                dout_coords = torch.bmm(rotations_back_current, dout_coords).permute(0, 2, 1)  # [v, PRED_LEN, 2]
                out_coords = dout_coords + x[:, [0, 1]].unsqueeze(1)

                mask = batch.edge_index[0, :] < batch.edge_index[1, :]
                _edge = batch.edge_index[:, mask].T  # [edge',2]
                dist = torch.linalg.norm(out_coords[_edge[:, 0]] - out_coords[_edge[:, 1]], dim=-1)  # [edge, PRED_LEN]

                collision_penalty = dist_threshold - dist[dist < dist_threshold]
                if collision_penalty.numel():  # there are can be no cars with small distanses so it will be empty
                    collision_penalty = collision_penalty.square().mean() * collision_penalty_factor
                    collision_penalties.append(collision_penalty)

                dist = torch.min(dist, dim=-1)[0]
                n_edge.append(len(dist))
                n_collision.append((dist < dist_threshold).sum().item())

                idx += 1

                if not on_predictions:
                    break

    ade = torch.cat(ade).mean()
    fde = torch.cat(fde)
    mr = ((fde > mr_threshold).sum() / len(fde)).item()
    fde = fde.mean()
    collision_rate = sum(n_collision) / sum(n_edge)
    collision_penalties = torch.tensor(collision_penalties)
    collision_penalty = collision_penalties.mean().item() if collision_penalties.numel() > 0 else 0.0

    val_losses = torch.tensor(val_losses).mean()

    return (
        ade.item(),
        fde.item(),
        mr,
        collision_rate,
        val_losses.item(),
        collision_penalty,
    )
