import os
from pathlib import Path
import yaml
import shutil
from tqdm import tqdm
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import pickle as pkl
import random
from torch.nn.utils.rnn import pad_sequence

from CoDriving.data_scripts.data_config.data_config import (
    ALLIGN_INITIAL_DIRECTION_TO_X,
    NUM_AUGMENTATION,
    NORMALIZE_DATA,
    MAP_BOUNDARY,
    NUM_PREDICT,
    PRED_LEN,
    NUM_PREDICT_ON_PREDICT,
    SAMPLE_RATE,
    PREDICT_VECTOR_SIZE,
    # VEHICLE_MAX_SPEED,
    COLLECT_DATA_RADIUS,
)
from CoDriving.models.model_factory import ModelFactory
from CoDriving.data_scripts.dataset import CarDataset
from CoDriving.data_scripts.preprocess_utils import (
    rotation_matrix_back_with_allign_to_X,
    rotation_matrix_with_allign_to_X,
    rotation_matrix_back_with_allign_to_Y,
    rotation_matrix_with_allign_to_Y,
    transform_sumo2carla_yaw,
    transform_coords,
    normalize_yaw,
    # normalize_speed,
    normalize_coords,
    denormalize_coords,
    denormalize_yaw,
)
from CoDriving.data_scripts.metrics_logger import MetricLogger


class Dict2Class(object):
    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])


class PathConfig:
    def __init__(
        self,
        base_exp_path: str,
        base_data_path: str,
        exp_id: str,
        train_config_path: str,
        model_config_path: str,
        logs_dir_path: str,
        train_data_dir: str,
        val_data_dir: str,
    ):
        self.experiments_path = base_exp_path
        self.base_data_path = base_data_path
        self.exp_path = os.path.join(base_exp_path, exp_id)
        train_config_name = os.path.basename(train_config_path)
        model_config_name = os.path.basename(model_config_path)

        self.logs_dir = os.path.join(self.exp_path, logs_dir_path)
        self.loss_logs_path = os.path.join(self.logs_dir, "train_loss_logs.csv")
        self.logs_plot_path = os.path.join(self.logs_dir, "train_loss_logs.png")

        self.copy_train_config_path = os.path.join(self.exp_path, train_config_name)
        self.copy_model_config_path = os.path.join(self.exp_path, model_config_name)
        self.model_checkpoints_dir = os.path.join(self.exp_path, "checkpoints")
        self.train_data_dir = os.path.join(self.base_data_path, train_data_dir)
        self.val_data_dir = os.path.join(self.base_data_path, val_data_dir)

    def create_directories(self):
        os.makedirs(self.experiments_path, exist_ok=True)
        os.makedirs(self.model_checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)


def read_config_file(config_file_path: str):
    try:
        with open(config_file_path) as config_file:
            dict_config = yaml.safe_load(config_file)
            class_config = Dict2Class(dict_config)

        return class_config
    except Exception as error:
        print(error)


def copy_file(src_file_path_str: str, dest_file_path_str: str):
    src = Path(src_file_path_str)
    dst = Path(dest_file_path_str)

    if dst.exists() and src.exists():
        try:
            if src.samefile(dst):
                return

        except FileNotFoundError:
            pass

    shutil.copy2(src, dst)


def get_optimizer(optimizer_name: str):
    optimizer_name = optimizer_name.lower()
    for name in dir(optim):
        opt = getattr(optim, name)

        if isinstance(opt, type) and name.lower() == optimizer_name:
            return opt

    raise ValueError(f"Optimizer '{optimizer_name}' not found")


def init_dataloaders(train_data_dir: str, val_data_dir: str, batch_size):
    try:
        train_dataset = CarDataset(preprocess_folder=train_data_dir, mlp=False, mpc_aug=(NUM_AUGMENTATION > 0))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        val_dataset = CarDataset(preprocess_folder=val_data_dir, mlp=False, mpc_aug=(NUM_AUGMENTATION > 0))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        return train_loader, val_loader

    except Exception as error:
        print(f"Error initializing datasets: {error}")
        return None, None


def init_model(model_config_path: str, optimizer_str: str, device: str, lr, weight_decay=None):
    model = ModelFactory.create_model(model_config_path)
    model = model.to(device)
    print(model)

    optimizer_cls = get_optimizer(optimizer_str)
    optimizer = optimizer_cls(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer


# def vec_angular_loss(gt_dspeed_dyaw: torch.Tensor, out_dspeed_dyaw: torch.Tensor):
#     vout = torch.cat((torch.cos(out_dspeed_dyaw).unsqueeze(-1), torch.sin(out_dspeed_dyaw).unsqueeze(-1)), dim=-1)
#     vgt = torch.cat((torch.cos(gt_dspeed_dyaw).unsqueeze(-1), torch.sin(gt_dspeed_dyaw).unsqueeze(-1)), dim=-1)
#     return torch.pow(torch.pow(vout - vgt, 2).sum(dim=-1), 0.5)


# def atan_angular_loss(gt_dspeed_dyaw: torch.Tensor, out_dspeed_dyaw: torch.Tensor):
#     angular_loss = torch.atan2(torch.sin(gt_dspeed_dyaw - out_dspeed_dyaw), torch.cos(gt_dspeed_dyaw - out_dspeed_dyaw))
#     return torch.pow(angular_loss, 2)


# def cosine_angular_loss(gt_dspeed_dyaw: torch.Tensor, out_dspeed_dyaw: torch.Tensor):
#     angular_loss = 1 - torch.cos(gt_dspeed_dyaw - out_dspeed_dyaw)
#     return angular_loss


# def angular_loss_func(gt_dspeed_dyaw: torch.Tensor, out_dspeed_dyaw: torch.Tensor):
#     return atan_angular_loss(gt_dspeed_dyaw, out_dspeed_dyaw)


# def capture_pi(out_yaw: torch.Tensor):
#     pi = np.pi
#     if NORMALIZE_DATA:
#         pi = normalize_yaw(np.pi)

#     out_yaw[:] = (out_yaw + pi) % (2 * pi) - pi


# def calc_dx_taylor(speed: torch.Tensor, dspeed: torch.Tensor, dyaw: torch.Tensor):
#     sec_res = (speed + dspeed * SAMPLE_RATE / 2) - (speed / 6 + dspeed * SAMPLE_RATE / 8) * torch.pow(
#         dyaw * SAMPLE_RATE, 2
#     )  # calculus for 1 second period
#     return sec_res / SAMPLE_RATE


# def calc_dy_taylor(speed: torch.Tensor, dspeed: torch.Tensor, dyaw: torch.Tensor):
#     sec_res = (speed / 2 + dspeed * SAMPLE_RATE / 3) * dyaw * SAMPLE_RATE - (speed / 24 + dspeed * SAMPLE_RATE / 30) * torch.pow(
#         dyaw * SAMPLE_RATE, 3
#     )
#     return sec_res / SAMPLE_RATE


# def calc_dx_exact(speed: torch.Tensor, dspeed: torch.Tensor, dyaw: torch.Tensor, eps=1e-8):
#     sec_res = (
#         dyaw * SAMPLE_RATE * (speed + dspeed * SAMPLE_RATE) * torch.sin(dyaw * SAMPLE_RATE)
#         + dspeed * SAMPLE_RATE * torch.cos(dyaw * SAMPLE_RATE)
#         - dspeed * SAMPLE_RATE
#     ) / (torch.pow(dyaw * SAMPLE_RATE, 2) + eps)
#     return sec_res / SAMPLE_RATE


# def calc_dy_exact(speed: torch.Tensor, dspeed: torch.Tensor, dyaw: torch.Tensor, eps=1e-8):
#     sec_res = (
#         -dyaw * SAMPLE_RATE * (speed + dspeed * SAMPLE_RATE) * torch.cos(dyaw * SAMPLE_RATE)
#         + dspeed * SAMPLE_RATE * torch.sin(dyaw * SAMPLE_RATE)
#         + speed * dyaw * SAMPLE_RATE
#     ) / (torch.pow(dyaw * SAMPLE_RATE, 2) + eps)
#     return sec_res / SAMPLE_RATE


# def calc_dx(speed: torch.Tensor, dspeed: torch.Tensor, dyaw: torch.Tensor, eps=1e-4):
#     small = torch.abs(dyaw) < eps

#     dx_exact = calc_dx_exact(speed, dspeed, dyaw)
#     dx_taylor = calc_dx_taylor(speed, dspeed, dyaw)

#     dx = torch.empty_like(dyaw)
#     dx[small] = dx_taylor[small]
#     dx[~small] = dx_exact[~small]
#     return dx


# def calc_dy(speed: torch.Tensor, dspeed: torch.Tensor, dyaw: torch.Tensor, eps=1e-4):
#     small = torch.abs(dyaw) < eps

#     dy_exact = calc_dy_exact(speed, dspeed, dyaw)
#     dy_taylor = calc_dy_taylor(speed, dspeed, dyaw)

#     dy = torch.empty_like(dyaw)
#     dy[small] = dy_taylor[small]
#     dy[~small] = dy_exact[~small]
#     return dy


# def train_one_epoch(
#     model,
#     device,
#     data_loader,
#     optimizer,
#     step_weights_factor,
#     dist_threshold,
#     collision_penalty,
#     collision_penalty_factor,
#     cos_sim_penalty,
#     speed_penalty,
# ):
#     """Performs an epoch of model training.

#     Parameters:
#     model (nn.Module): Model to be trained.
#     device (torch.Device): Device used for training.
#     data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
#     optimizer (torch.optim.Optimizer): Optimizer used to update model.
#     train_config: object of config of train process.

#     Returns:
#     float: Avg loss for epoch.
#     """
#     step_weights = torch.ones(PRED_LEN, device=device)
#     step_weights[:5] *= step_weights_factor
#     step_weights[0] *= step_weights_factor

#     model.train()
#     total_loss = 0

#     for batch in data_loader:
#         batch = batch.to(device)
#         out = None
#         out_dspeed_dyaw = None

#         for idx in range(NUM_PREDICT_ON_PREDICT + 1):
#             optimizer.zero_grad()
#             # [x, y, v, yaw, intention(3-bit)] -> [x, y, intention]
#             # out = model(batch.x[:, [0, 1, 4, 5, 6]], batch.edge_index)
#             # out = out.reshape(-1, PRED_LEN, 2)  # [v, pred, 2]

#             if idx == 0:
#                 x = batch.x
#             else:
#                 # out - already new coords from last prediction
#                 # out_dspeed_dyaw - deltas in speed and yaw
#                 x = torch.cat((out[:, 0, :].detach(), x[:, [2, 3]] + out_dspeed_dyaw[:, 0, :].detach(), x[:, [4, 5, 6]]), dim=1)
#                 # capture_pi(x[:, 3])

#             out_dspeed_dyaw = model(x, batch.edge_index)
#             out_dspeed_dyaw = out_dspeed_dyaw.reshape(-1, PRED_LEN, 2)  # [v, pred, 2]
#             # capture_pi(out_dspeed_dyaw[:, :, 1]) <<<<<<<<<<<<<< не надо так делаеть если что))))

#             # по что лосс взрывается так что clamp
#             # out_dspeed_dyaw = torch.stack([torch.clamp(out_dspeed_dyaw[:, :, 0], -15, 15), torch.clamp(out_dspeed_dyaw[:, :, 1], -2, 2)], dim=2)

#             # dx = (x[:, 2].unsqueeze(-1) + out_dspeed_dyaw[:, :, 0]) * torch.cos(out_dspeed_dyaw[:, :, 1] / 2)
#             # dy = (x[:, 2].unsqueeze(-1) + out_dspeed_dyaw[:, :, 0]) * torch.sin(out_dspeed_dyaw[:, :, 1] / 2)
#             dx = calc_dx(x[:, 2].unsqueeze(-1), out_dspeed_dyaw[:, :, 0], out_dspeed_dyaw[:, :, 1])
#             dy = calc_dy(x[:, 2].unsqueeze(-1), out_dspeed_dyaw[:, :, 0], out_dspeed_dyaw[:, :, 1])
#             dout = torch.cat((dx.unsqueeze(-1), dy.unsqueeze(-1)), dim=-1)

#             yaw = x[:, 3].detach().cpu().numpy()

#             if NORMALIZE_DATA:
#                 de_normalize_yaw(yaw)

#             if ALLIGN_INITIAL_DIRECTION_TO_X:
#                 rotations = torch.stack([rotation_matrix_back_with_allign_to_X(yaw[i]) for i in range(batch.x.shape[0])]).to(dout.device)
#             else:
#                 rotations = torch.stack([rotation_matrix_back_with_allign_to_Y(yaw[i]) for i in range(batch.x.shape[0])]).to(dout.device)

#             # [x, y, v, yaw, acc, steering]
#             # gt = batch.y.reshape(-1, PRED_LEN, 6)[:, :, [0, 1]]

#             gt_dspeed_dyaw_correction = (batch.x[:, [2, 3]] - x[:, [2, 3]]).unsqueeze(1)
#             gt_dspeed_dyaw = batch.y.reshape(-1, NUM_PREDICT, 6)[:, :, [2, 3]] + gt_dspeed_dyaw_correction
#             gt_dspeed_dyaw = gt_dspeed_dyaw[:, idx : PRED_LEN + idx, :]
#             # capture_pi(gt_dspeed_dyaw[:, :, 1])

#             # gt_dx = (x[:, 2].unsqueeze(-1) + gt_dspeed_dyaw[:, :, 0]) * torch.cos(gt_dspeed_dyaw[:, :, 1] / 2)
#             # gt_dy = (x[:, 2].unsqueeze(-1) + gt_dspeed_dyaw[:, :, 0]) * torch.sin(gt_dspeed_dyaw[:, :, 1] / 2)
#             gt_dx = calc_dx(x[:, 2].unsqueeze(-1), gt_dspeed_dyaw[:, :, 0], gt_dspeed_dyaw[:, :, 1])
#             gt_dy = calc_dy(x[:, 2].unsqueeze(-1), gt_dspeed_dyaw[:, :, 0], gt_dspeed_dyaw[:, :, 1])
#             dgt = torch.cat((gt_dx.unsqueeze(-1), gt_dy.unsqueeze(-1)), dim=-1)

#             if NORMALIZE_DATA:
#                 denom = dgt.detach().abs().clamp(min=1e-5)
#                 error = (((dgt - dout) / denom).square().sum(-1) * step_weights).sum(-1)
#             else:
#                 error = ((dgt - dout).square().sum(-1) * step_weights).sum(-1)

#             # yaw_loss = torch.atan2(
#             #     torch.sin(out_dspeed_dyaw[:, :, 1] - gt_dspeed_dyaw[:, :, 1]),
#             #     torch.cos(out_dspeed_dyaw[:, :, 1] - gt_dspeed_dyaw[:, :, 1]),
#             # )
#             yaw_loss = angular_loss_func(gt_dspeed_dyaw[:, :, 1], out_dspeed_dyaw[:, :, 1])
#             yaw_loss = (yaw_loss * step_weights).sum(-1)
#             error = error + yaw_loss * cos_sim_penalty

#             speed_loss = ((gt_dspeed_dyaw[:, :, 0] - out_dspeed_dyaw[:, :, 0]).square() * step_weights).sum(-1)
#             error = speed_loss * speed_penalty + error

#             # косинусная близость теоретически думал на шум в предикте координат
#             # cos_sim = torch.nn.functional.cosine_similarity(dout, dgt, dim=-1)
#             # direction_loss = 1 - cos_sim.mean(dim=-1)
#             # error = error + cos_sim_penalty * direction_loss

#             # лосс на шумы в координатах
#             # error = error + cos_sim_penalty * torch.sum(torch.relu(-dgt * dout), dim=(-1))[:, 0]

#             loss = (batch.weights * error).nanmean()

#             # get back from deltas to plain coordinates in predictions
#             dout = dout.permute(0, 2, 1)  # [v, 2, pred]
#             dout = torch.bmm(rotations, dout).permute(0, 2, 1)  # [v, pred, 2]
#             out = dout + x[:, [0, 1]].unsqueeze(1)

#             if collision_penalty:
#                 mask = batch.edge_index[0, :] < batch.edge_index[1, :]
#                 _edge = batch.edge_index[:, mask].T  # [edge',2]
#                 dist = torch.linalg.norm(out[_edge[:, 0]] - out[_edge[:, 1]], dim=-1)
#                 dist = dist_threshold - dist[dist < dist_threshold]

#                 if dist.numel():  # there are can be no cars with small distanses so it will be empty
#                     _collision_penalty = dist.square().mean()
#                     loss += _collision_penalty * collision_penalty_factor

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item() / (NUM_PREDICT_ON_PREDICT + 1)

#     return total_loss / len(data_loader)


# def evaluate(model, device, data_loader, step_weights_factor, dist_threshold, mr_threshold, collision_penalty_factor, cos_sim_penalty, speed_penalty):
#     """Performs an epoch of model training.

#     Parameters:
#     model (nn.Module): Model to be trained.
#     device (torch.Device): Device used for training.
#     data_loader (torch.utils.data.DataLoader): Data loader containing all batches.

#     Returns:
#     list of evaluation metrics (including ADE, FDE, etc.).
#     """
#     step_weights = torch.ones(PRED_LEN, device=device)
#     step_weights[:5] *= step_weights_factor
#     step_weights[0] *= step_weights_factor

#     model.eval()
#     ade, fde = [], []
#     n_edge, n_collision = [], []
#     val_losses, collision_penalties = [], []

#     with torch.no_grad():
#         for batch in data_loader:
#             batch = batch.to(device)
#             out = None
#             out_dspeed_dyaw = None

#             for idx in range(NUM_PREDICT_ON_PREDICT + 1):
#                 # out = model(batch.x[:, [0, 1, 4, 5, 6]], batch.edge_index)
#                 # out = out.reshape(-1, PRED_LEN, 2)  # [v, PRED_LEN, 2]

#                 if idx == 0:
#                     x = batch.x
#                 else:
#                     x = torch.cat((out[:, 0, :].detach(), x[:, [2, 3]] + out_dspeed_dyaw[:, 0, :].detach(), x[:, [4, 5, 6]]), dim=1)
#                     # capture_pi(x[:, 3])

#                 out_dspeed_dyaw = model(x, batch.edge_index)
#                 out_dspeed_dyaw = out_dspeed_dyaw.reshape(-1, PRED_LEN, 2)  # [v, pred, 2]
#                 # capture_pi(out_dspeed_dyaw[:, :, 1]) <<<<<<<<<<<<<< не надо так делаеть если что))))

#                 # по что лосс взрывается так что clamp
#                 # out_dspeed_dyaw = torch.stack([torch.clamp(out_dspeed_dyaw[:, :, 0], -15, 15), torch.clamp(out_dspeed_dyaw[:, :, 1], -2, 2)], dim=2)

#                 # dx = (x[:, 2].unsqueeze(-1) + out_dspeed_dyaw[:, :, 0]) * torch.cos(out_dspeed_dyaw[:, :, 1] / 2)
#                 # dy = (x[:, 2].unsqueeze(-1) + out_dspeed_dyaw[:, :, 0]) * torch.sin(out_dspeed_dyaw[:, :, 1] / 2)
#                 dx = calc_dx(x[:, 2].unsqueeze(-1), out_dspeed_dyaw[:, :, 0], out_dspeed_dyaw[:, :, 1])
#                 dy = calc_dy(x[:, 2].unsqueeze(-1), out_dspeed_dyaw[:, :, 0], out_dspeed_dyaw[:, :, 1])
#                 dout = torch.cat((dx.unsqueeze(-1), dy.unsqueeze(-1)), dim=-1)

#                 yaw = x[:, 3].detach().cpu().numpy()

#                 if NORMALIZE_DATA:
#                     de_normalize_yaw(yaw)

#                 if ALLIGN_INITIAL_DIRECTION_TO_X:
#                     rotations = torch.stack([rotation_matrix_back_with_allign_to_X(yaw[i]) for i in range(batch.x.shape[0])]).to(dout.device)
#                 else:
#                     rotations = torch.stack([rotation_matrix_back_with_allign_to_Y(yaw[i]) for i in range(batch.x.shape[0])]).to(dout.device)

#                 # gt = batch.y.reshape(-1, PRED_LEN, 6)[:, :, [0, 1]]

#                 gt_dspeed_dyaw_correction = (batch.x[:, [2, 3]] - x[:, [2, 3]]).unsqueeze(1)
#                 gt_dspeed_dyaw = batch.y.reshape(-1, NUM_PREDICT, 6)[:, :, [2, 3]] + gt_dspeed_dyaw_correction
#                 gt_dspeed_dyaw = gt_dspeed_dyaw[:, idx : PRED_LEN + idx, :]
#                 # capture_pi(gt_dspeed_dyaw[:, :, 1])

#                 # gt_dx = (x[:, 2].unsqueeze(-1) + gt_dspeed_dyaw[:, :, 0]) * torch.cos(gt_dspeed_dyaw[:, :, 1] / 2)
#                 # gt_dy = (x[:, 2].unsqueeze(-1) + gt_dspeed_dyaw[:, :, 0]) * torch.sin(gt_dspeed_dyaw[:, :, 1] / 2)
#                 gt_dx = calc_dx(x[:, 2].unsqueeze(-1), gt_dspeed_dyaw[:, :, 0], gt_dspeed_dyaw[:, :, 1])
#                 gt_dy = calc_dy(x[:, 2].unsqueeze(-1), gt_dspeed_dyaw[:, :, 0], gt_dspeed_dyaw[:, :, 1])
#                 dgt = torch.cat((gt_dx.unsqueeze(-1), gt_dy.unsqueeze(-1)), dim=-1)

#                 _error = (dgt - dout).square().sum(-1)
#                 error = _error.clone() ** 0.5
#                 _error = (_error * step_weights).sum(-1)

#                 # yaw_loss = torch.atan2(
#                 #     torch.sin(out_dspeed_dyaw[:, :, 1] - gt_dspeed_dyaw[:, :, 1]), torch.cos(out_dspeed_dyaw[:, :, 1] - gt_dspeed_dyaw[:, :, 1])
#                 # )
#                 yaw_loss = angular_loss_func(gt_dspeed_dyaw[:, :, 1], out_dspeed_dyaw[:, :, 1])
#                 yaw_loss = (yaw_loss * step_weights).sum(-1)
#                 _error = _error + yaw_loss * cos_sim_penalty

#                 speed_loss = ((gt_dspeed_dyaw[:, :, 0] - out_dspeed_dyaw[:, :, 0]).square() * step_weights).sum(-1)
#                 _error = speed_loss * speed_penalty + _error

#                 # cos_sim = torch.nn.functional.cosine_similarity(dout, dgt, dim=-1)
#                 # direction_loss = 1 - cos_sim.mean(dim=-1)
#                 # _error = _error + cos_sim_penalty * direction_loss

#                 # _error = _error + cos_sim_penalty * torch.sum(torch.relu(-dgt * dout), dim=(-1))[:, 0]

#                 val_loss = (batch.weights * _error).nanmean()
#                 val_losses.append(val_loss)
#                 fde.append(error[:, -1])
#                 ade.append(error.mean(dim=-1))

#                 dout = dout.permute(0, 2, 1)  # [v, 2, pred]
#                 dout = torch.bmm(rotations, dout).permute(0, 2, 1)  # [v, pred, 2]
#                 out = dout + x[:, [0, 1]].unsqueeze(1)

#                 mask = batch.edge_index[0, :] < batch.edge_index[1, :]
#                 _edge = batch.edge_index[:, mask].T  # [edge',2]
#                 dist = torch.linalg.norm(out[_edge[:, 0]] - out[_edge[:, 1]], dim=-1)  # [edge, 30]

#                 collision_penalty = dist_threshold - dist[dist < dist_threshold]
#                 if collision_penalty.numel():  # there are can be no cars with small distanses so it will be empty
#                     collision_penalty = collision_penalty.square().mean() * collision_penalty_factor
#                     collision_penalties.append(collision_penalty)

#                 dist = torch.min(dist, dim=-1)[0]
#                 n_edge.append(len(dist))
#                 n_collision.append((dist < dist_threshold).sum().item())

#                 # out = out.permute(0,2,1)    # [v, 2, pred]
#                 # yaw = batch.x[:,3].detach().cpu().numpy()
#                 # rotations = torch.stack([rotation_matrix_back(yaw[i])  for i in range(batch.x.shape[0])]).to(out.device)
#                 # out = torch.bmm(rotations, out).permute(0,2,1)       # [v, pred, 2]
#                 # out += batch.x[:,[7,8]].unsqueeze(1)
#                 # # gt = batch.y.reshape(-1,50,6)[:,:,[0,1]]
#                 # # error = ((gt-out).square().sum(-1) * step_weights).sum(-1)
#                 # # loss = (batch.weights * error).nanmean()

#                 # mask = (batch.edge_index[0,:] < batch.edge_index[1,:])
#                 # _edge = batch.edge_index[:, mask].T   # [edge',2]
#                 # # pos1 = torch.stack([out[_edge[i, 0]] for i in range(_edge.shape[0])])
#                 # # pos2 = torch.stack([out[_edge[i, 1]] for i in range(_edge.shape[0])])
#                 # # dist = torch.linalg.norm(pos1 - pos2, dim=-1)
#                 # dist = torch.linalg.norm(out[_edge[:,0]] - out[_edge[:,1]], dim=-1) # [edge, 50]

#     ade = torch.cat(ade).mean()
#     fde = torch.cat(fde)
#     mr = ((fde > mr_threshold).sum() / len(fde)).item()
#     fde = fde.mean()
#     collision_rate = sum(n_collision) / sum(n_edge)
#     collision_penalties = torch.tensor(collision_penalties)
#     collision_penalty = collision_penalties.mean().item() if collision_penalties.numel() > 0 else 0.0

#     val_losses = torch.tensor(val_losses).mean()

#     return (
#         ade.item(),
#         fde.item(),
#         mr,
#         collision_rate,
#         val_losses.item(),
#         collision_penalty,
#     )


def load_yaw_dict():
    with open("../opencda/assets/yaw_dict_10m.pkl", "rb") as f:
        yaw_dict = pkl.load(f)
        rename = {
            "left_up": 10 * 0 + 0,
            "left_right": 10 * 0 + 1,
            "left_down": 10 * 0 + 2,
            "up_right": 10 * 1 + 0,
            "up_down": 10 * 1 + 1,
            "up_left": 10 * 1 + 2,
            "right_down": 10 * 2 + 0,
            "right_left": 10 * 2 + 1,
            "right_up": 10 * 2 + 2,
            "down_left": 10 * 3 + 0,
            "down_up": 10 * 3 + 1,
            "down_right": 10 * 3 + 2,
        }

        yaw_dict = {rename.get(k, k): v for k, v in yaw_dict.items()}
        return yaw_dict


def my_get_yaw(start_positions: torch.Tensor, intentions: torch.Tensor, pos: torch.Tensor, yaw_keys: torch.Tensor, yaw_vals: torch.Tensor):
    start_positions_nums = torch.argmax(start_positions, dim=-1)
    intentions_nums = torch.argmax(intentions, dim=-1)

    idxs = 10 * start_positions_nums + intentions_nums
    mask = idxs.unsqueeze(1) == yaw_keys.unsqueeze(0)
    positions = mask.nonzero()[:, 1]
    route = yaw_vals[positions]

    carla_pos = pos.clone()  # carla coords
    if NORMALIZE_DATA:
        carla_pos = denormalize_coords(carla_pos, COLLECT_DATA_RADIUS)

    sumo_pos = transform_coords(carla_pos)
    deltas = sumo_pos.unsqueeze(1) - route.float()[:, :, :-1]
    dists = torch.norm(deltas, dim=-1)

    min_idx = torch.argmin(dists, dim=-1)
    batch_idx = torch.arange(route.size(0), device=route.device)
    yaws_deg = route[batch_idx, min_idx, -1].to(torch.float32)

    yaws_rad = torch.deg2rad(yaws_deg)
    yaws_rad_carla = transform_sumo2carla_yaw(yaws_rad)

    if NORMALIZE_DATA:
        yaws_rad_carla = normalize_yaw(yaws_rad_carla)
    return yaws_rad_carla


def my_get_speed(dx: torch.Tensor, dy: torch.Tensor):
    speed = (dx**2 + dy**2) ** 0.5 * SAMPLE_RATE
    # vehicle_max_speed = VEHICLE_MAX_SPEED if not NORMALIZE_DATA else normalize_speed(VEHICLE_MAX_SPEED, VEHICLE_MAX_SPEED)
    # mask = (speed > vehicle_max_speed)
    # speed[mask] = vehicle_max_speed
    return speed


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
    start_prediction_time=0.2,
):
    """Performs an epoch of GNN model training with rollout

    :param model (nn.Module): Model to be trained.
    :param device (torch.Device): Device used for training.
    :param data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
    :param optimizer (torch.optim.Optimizer): Optimizer used to update model.
    :param step_weights_factor: multiplication factor for each step of trajectory prediction
    :param dist_threshold: distence of collition
    :param collision_penalty: flag to use collition in loss
    :param collision_penalty_factor: multiplcation factor of collition to apply to loss
    :param epoch: current epoch
    :param epochs: total number of epochs
    :param yaw_keys: yaw dictionary keys of yaw dict calculated before
    :param yaw_values: yaw dictionary values of yaw dict calculated before
    :param start_prediction_time: number of epoch / epochs to start making predictions on prediction (set maximum of 0.5 for correct calculus)

    :return float: Avg loss for epoch.
    """
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
                random.random() < epoch / epochs and epoch / epochs > start_prediction_time + (idx / NUM_PREDICT_ON_PREDICT) / 2
            ):  # max of (idx / NUM_PREDICT_ON_PREDICT / 2) = 0.5
                on_predictions = True
            else:
                on_predictions = False

            optimizer.zero_grad()

            if idx == 0:
                x = batch.x[:, [0, 1, 2, 3, 4, 5, 6]]
            else:
                yaws = my_get_yaw(batch.x[:, [7, 8, 9, 10]], batch.x[:, [4, 5, 6]], out_coords[:, 0, :].detach(), yaw_keys, yaw_values)
                speed = my_get_speed(dout_coords[:, 0, 0].detach(), dout_coords[:, 0, 1].detach())
                x = torch.cat(
                    (
                        out_coords[:, 0, :].detach(),
                        speed.unsqueeze(-1),
                        yaws.unsqueeze(-1),
                        batch.x[:, [4, 5, 6]],
                    ),
                    dim=-1,
                )

            dout_coords = model(x, batch.edge_index)
            dout_coords = dout_coords.reshape(-1, PRED_LEN, PREDICT_VECTOR_SIZE)  # [v, PRED_LEN, PREDICT_VECTOR_SIZE]

            yaw_cur = x[:, 3].detach().cpu().numpy()
            yaw_base = batch.x[:, 3].detach().cpu().numpy()
            if NORMALIZE_DATA:
                denormalize_yaw(yaw_cur)
                denormalize_yaw(yaw_base)

            # [x, y, v, yaw, acc, steering]
            dgt_coords = batch.y.reshape(-1, NUM_PREDICT, 6)[:, :, [0, 1]]

            if ALLIGN_INITIAL_DIRECTION_TO_X:
                rotations_back_current = torch.stack([rotation_matrix_back_with_allign_to_X(yaw_cur[i]) for i in range(batch.x.shape[0])]).to(device)
            else:
                rotations_back_current = torch.stack([rotation_matrix_back_with_allign_to_Y(yaw_cur[i]) for i in range(batch.x.shape[0])]).to(device)

            if idx > 0:
                if ALLIGN_INITIAL_DIRECTION_TO_X:
                    rotations_back_base = torch.stack([rotation_matrix_back_with_allign_to_X(yaw_base[i]) for i in range(batch.x.shape[0])]).to(
                        device
                    )
                    rotation_current = torch.stack([rotation_matrix_with_allign_to_X(yaw_cur[i]) for i in range(batch.x.shape[0])]).to(device)
                else:
                    rotations_back_base = torch.stack([rotation_matrix_back_with_allign_to_Y(yaw_base[i]) for i in range(batch.x.shape[0])]).to(
                        device
                    )
                    rotation_current = torch.stack([rotation_matrix_with_allign_to_Y(yaw_cur[i]) for i in range(batch.x.shape[0])]).to(device)

                dgt_coords = dgt_coords.permute(0, 2, 1)  # [v, 2, NUM_PREDICT]
                dgt_coords = torch.bmm(rotations_back_base, dgt_coords)
                dgt_coords = dgt_coords + (batch.x[:, [0, 1]].unsqueeze(1) - x[:, [0, 1]].unsqueeze(1)).permute(0, 2, 1)
                dgt_coords = torch.bmm(rotation_current, dgt_coords).permute(0, 2, 1)
            dgt_coords = dgt_coords[:, idx : PRED_LEN + idx, :]  # [v, PRED_LEN, 2]

            # if NORMALIZE_DATA:
            #     denom = dgt_coords.detach().abs().clamp(min=1e-5)
            #     error = (((dgt_coords - dout_coords) / denom).square().sum(-1) * step_weights).sum(-1)
            # else:
            #     error = ((dgt_coords - dout_coords).square().sum(-1) * step_weights).sum(-1)
            error = ((dgt_coords - dout_coords).square().sum(-1) * step_weights).sum(-1)

            loss = (batch.weights * error).nanmean()

            # get back from deltas to plain coordinates in predictions
            dout_coords = dout_coords.permute(0, 2, 1)  # [v, 2, PRED_LEN]
            dout_coords = torch.bmm(rotations_back_current, dout_coords).permute(0, 2, 1)  # [v, PRED_LEN, 2]
            out_coords = dout_coords + x[:, [0, 1]].unsqueeze(1)

            if collision_penalty:
                mask = batch.edge_index[0, :] < batch.edge_index[1, :]
                _edge = batch.edge_index[:, mask].T  # [edge',2]
                dist = torch.linalg.norm(out_coords[_edge[:, 0]] - out_coords[_edge[:, 1]], dim=-1)
                dist = dist_threshold - dist[dist < dist_threshold]

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
    start_prediction_time=0.2,
):
    """Performs an epoch of model training.

    :param model (nn.Module): Model to be trained.
    :param device (torch.Device): Device used for training.
    :param data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
    :param step_weights_factor: multiplication factor for each step of trajectory prediction
    :param dist_threshold: distance of collition
    :param mr_threshold: bad difference in last step prediction
    :param collision_penalty_factor: multiplcation factor of collition to apply to loss
    :param epoch: current epoch
    :param epochs: total number of epochs
    :param yaw_keys: yaw dictionary keys of yaw dict calculated before
    :param yaw_values: yaw dictionary values of yaw dict calculated before
    :param start_prediction_time: number of epoch / epochs to start making predictions on prediction (set maximum of 0.5 for correct calculus)

    :return list of evaluation metrics (including ADE, FDE, etc.).
    """
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
                    random.random() < epoch / epochs and epoch / epochs > start_prediction_time + (idx / NUM_PREDICT_ON_PREDICT) / 2
                ):  # max of (idx / NUM_PREDICT_ON_PREDICT / 2) = 0.5
                    on_predictions = True
                else:
                    on_predictions = False

                if idx == 0:
                    x = batch.x[:, [0, 1, 2, 3, 4, 5, 6]]
                else:
                    yaws = my_get_yaw(batch.x[:, [7, 8, 9, 10]], batch.x[:, [4, 5, 6]], out_coords[:, 0, :], yaw_keys, yaw_values)
                    speed = my_get_speed(dout_coords[:, 0, 0], dout_coords[:, 0, 1])
                    x = torch.cat(
                        (
                            out_coords[:, 0, :],
                            speed.unsqueeze(-1),
                            yaws.unsqueeze(-1),
                            batch.x[:, [4, 5, 6]],
                        ),
                        dim=-1,
                    )

                dout_coords = model(x, batch.edge_index)
                dout_coords = dout_coords.reshape(-1, PRED_LEN, PREDICT_VECTOR_SIZE)  # [v, PRED_LEN, PREDICT_VECTOR_SIZE]

                yaw_cur = x[:, 3].cpu().numpy()
                yaw_base = batch.x[:, 3].cpu().numpy()
                if NORMALIZE_DATA:
                    denormalize_yaw(yaw_cur)
                    denormalize_yaw(yaw_base)

                # [x, y, v, yaw, acc, steering]
                dgt_coords = batch.y.reshape(-1, NUM_PREDICT, 6)[:, :, [0, 1]]

                if ALLIGN_INITIAL_DIRECTION_TO_X:
                    rotations_back_current = torch.stack([rotation_matrix_back_with_allign_to_X(yaw_cur[i]) for i in range(batch.x.shape[0])]).to(
                        device
                    )
                else:
                    rotations_back_current = torch.stack([rotation_matrix_back_with_allign_to_Y(yaw_cur[i]) for i in range(batch.x.shape[0])]).to(
                        device
                    )

                if idx > 0:
                    if ALLIGN_INITIAL_DIRECTION_TO_X:
                        rotations_back_base = torch.stack([rotation_matrix_back_with_allign_to_X(yaw_base[i]) for i in range(batch.x.shape[0])]).to(
                            device
                        )
                        rotation_current = torch.stack([rotation_matrix_with_allign_to_X(yaw_cur[i]) for i in range(batch.x.shape[0])]).to(device)
                    else:
                        rotations_back_base = torch.stack([rotation_matrix_back_with_allign_to_Y(yaw_base[i]) for i in range(batch.x.shape[0])]).to(
                            device
                        )
                        rotation_current = torch.stack([rotation_matrix_with_allign_to_Y(yaw_cur[i]) for i in range(batch.x.shape[0])]).to(device)

                    dgt_coords = dgt_coords.permute(0, 2, 1)  # [v, 2, NUM_PREDICT]
                    dgt_coords = torch.bmm(rotations_back_base, dgt_coords)
                    dgt_coords = dgt_coords + (batch.x[:, [0, 1]].unsqueeze(1) - x[:, [0, 1]].unsqueeze(1)).permute(0, 2, 1)
                    dgt_coords = torch.bmm(rotation_current, dgt_coords).permute(0, 2, 1)
                dgt_coords = dgt_coords[:, idx : PRED_LEN + idx, :]  # [v, PRED_LEN, 2]

                # if NORMALIZE_DATA:
                #     denom = dgt_coords.detach().abs().clamp(min=1e-5)
                #     _error = (((dgt_coords - dout_coords) / denom).square().sum(-1) * step_weights).sum(-1)
                # else:
                #     _error = ((dgt_coords - dout_coords).square().sum(-1) * step_weights).sum(-1)
                _error = ((dgt_coords - dout_coords).square().sum(-1) * step_weights).sum(-1)

                _error = (_error * step_weights).sum(-1)
                error = (dgt_coords - dout_coords).square().sum(-1) ** 0.5

                val_loss = (batch.weights * _error).nanmean()
                val_losses.append(val_loss)
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


METRICS = ["ade", "fde", "mr", "collision_rate", "val_loss", "collision_penalties"]


def train_one_config(
    process_conection,
    train_config_path: str,
    model_config_path: str,
    expirements_path: str,
    data_path: str,
    logs_dir_name: str,
    device_str: str,
    save_last_checkpoints=True,
):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    train_config = read_config_file(train_config_path)
    train_config_name = os.path.splitext(os.path.basename(train_config_path))[0]
    model_config_name = os.path.splitext(os.path.basename(model_config_path))[0]
    exp_id = f"{train_config_name}_{model_config_name}"

    path_config = PathConfig(
        expirements_path,
        data_path,
        exp_id,
        train_config_path,
        model_config_path,
        logs_dir_name,
        train_config.train_data_dir,
        train_config.val_data_dir,
    )
    path_config.create_directories()
    copy_file(train_config_path, path_config.copy_train_config_path)
    copy_file(model_config_path, path_config.copy_model_config_path)

    try:
        device = torch.device(device_str)
    except Exception as e:
        print(e)
        device = torch.device("cpu")

    metrics_log_epoch_frequency = train_config.metrics_log_epoch_frequency
    epochs = train_config.epoch

    step_weights_factor = train_config.step_weights_factor
    collision_penalty = train_config.collision_penalty
    collision_penalty_factor = train_config.collision_penalty_factor
    dist_threshold = train_config.dist_threshold
    mr_threshold = train_config.mr_threshold
    start_prediction_time = train_config.start_prediction_time

    if NORMALIZE_DATA:
        dist_threshold = normalize_coords(dist_threshold, COLLECT_DATA_RADIUS)
        mr_threshold = normalize_coords(mr_threshold, COLLECT_DATA_RADIUS)

    push_to_hf = train_config.push_to_hf
    hf_project_path = train_config.hf_project_path

    train_loader, val_loader = init_dataloaders(path_config.train_data_dir, path_config.val_data_dir, train_config.batch_size)
    model, optimizer = init_model(
        path_config.copy_model_config_path,
        train_config.optimizer,
        device,
        train_config.lr,
        train_config.weight_decay,
    )

    min_ade = 1e6
    min_fde = 1e6
    best_epoch = 0
    patience = 100
    record = []

    loss_logger = MetricLogger(path_config.loss_logs_path, path_config.logs_plot_path, enable_plot=True)
    loss_logger.clear_file()

    metric_loggers = {}
    for metric in METRICS:
        metric_loggers[metric] = MetricLogger(
            os.path.join(path_config.logs_dir, f"{metric}_logs.csv"),
            os.path.join(path_config.logs_dir, f"{metric}_logs.png"),
            enable_plot=True,
        )
        metric_loggers[metric].clear_file()

    yaw_dict = load_yaw_dict()
    yaw_keys = torch.tensor(list(yaw_dict.keys()), device=device)
    values_list = list(yaw_dict.values())
    values_list = [torch.tensor(v, device=device) for v in values_list]
    yaw_values = pad_sequence(values_list, batch_first=True, padding_value=MAP_BOUNDARY)

    process_conection.send(1)  # tells parent process that everything is inited
    process_conection.close()

    for epoch in tqdm(range(0, epochs)):
        epoch_loss = gnn_train_one_epoch(
            model,
            device,
            train_loader,
            optimizer,
            step_weights_factor,
            dist_threshold,
            collision_penalty,
            collision_penalty_factor,
            epoch,
            epochs,
            yaw_keys,
            yaw_values,
            start_prediction_time,
        )
        loss_logger.add_metric_points([epoch], [epoch * len(train_loader)], [epoch_loss])

        if epoch % metrics_log_epoch_frequency == 0:
            ade, fde, mr, collision_rate, val_loss, collision_penalties = gnn_evaluate(
                model,
                device,
                val_loader,
                step_weights_factor,
                dist_threshold,
                mr_threshold,
                collision_penalty_factor,
                epoch,
                epochs,
                yaw_keys,
                yaw_values,
                start_prediction_time,
            )
            epoch_metrics = {
                "ade": ade,
                "fde": fde,
                "mr": mr,
                "collision_rate": collision_rate,
                "val_loss": val_loss,
                "collision_penalties": collision_penalties,
            }
            record.append(epoch_metrics)

            if fde < min_fde:
                min_ade, min_fde = ade, fde
                best_epoch = epoch

            elif (epoch - best_epoch) > patience:
                if patience > 1600:  # x16
                    print(f"Earlier stops, Best Epoch: {best_epoch}, Min ADE: {min_ade}, Min FDE: {min_fde}, MR: {mr}, CR:{collision_rate}")
                    break
                else:
                    optimizer.param_groups[0]["lr"] *= 0.5
                    patience *= 2

        if save_last_checkpoints and epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(
                    path_config.model_checkpoints_dir,
                    f"model_{'wp' if collision_penalty else 'np'}_{exp_id}_{str(epoch).zfill(4)}.pth",
                ),
            )

    loss_logger.plot_metric()

    record_df = pd.DataFrame(record)
    for metric in METRICS:
        metric_loggers[metric].add_metric_points(
            np.arange(0, epochs, step=metrics_log_epoch_frequency),
            np.arange(
                0,
                len(val_loader) * epochs,
                step=len(val_loader) * metrics_log_epoch_frequency,
            ),
            record_df[metric],
        )
        metric_loggers[metric].plot_metric()

    if push_to_hf:
        best_model_path = os.path.join(
            path_config.model_checkpoints_dir,
            f"model_{'wp' if collision_penalty else 'np'}_{exp_id}_{str(epochs - 1).zfill(4)}.pth",
        )
        best_state_dict = torch.load(best_model_path)
        model.load_state_dict(best_state_dict)
        model.push_to_hub(hf_project_path)

    if "cuda" in device_str:
        torch.cuda.empty_cache()
