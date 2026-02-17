import os
from pathlib import Path
import yaml
import shutil
from tqdm import tqdm
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle as pkl
import math
from torch.nn.utils.rnn import pad_sequence

from AIM.models.mtp.learning.data_path_config import DATA_PATH, Y_X_DISTR_FILE, Y_Y_DISTR_FILE, YAW_DICT_PATH
from AIM.models.mtp.learning.learning_src.data_scripts.data_config import (
    NUM_AUGMENTATION,
)
from AIM.models.mtp.mtp_models.model_factory import ModelFactory
from AIM.models.mtp.learning.learning_src.data_scripts.dataset import GnnCarDataset, TransformerCarDataset
from AIM.models.mtp.learning.learning_src.data_scripts.metrics_logger import MetricLogger
from .train_gnn import gnn_train_one_epoch, gnn_evaluate
from .train_transformer import transformer_train_one_epoch, transformer_evaluate


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


def my_collate_fn(batch):
    x = torch.stack([b[0].x for b in batch])  # (batch_size, max_v, 11)
    y = torch.stack([b[0].y for b in batch])  # (batch_size, max_v, pred_len*6)

    map_infos = torch.stack([b[1] for b in batch])  # (batch_size, max_c, k, k)
    map_attn_masks = torch.stack([b[2] for b in batch])  # (batch_size, max_c, max_c)
    map_boundaries = torch.stack([b[3] for b in batch])  # (batch_size, 1)

    weights = torch.stack([b[0].weights for b in batch])  # (batch_size, max_v)
    attn_mask = torch.stack([b[0].attn_mask for b in batch])  # (batch_size, max_v, max_v)
    return x, y, weights, attn_mask, map_infos, map_attn_masks, map_boundaries


def init_dataloaders(train_data_dir: str, val_data_dir: str, batch_size, is_transformer=False):
    try:
        if is_transformer:
            train_dataset = TransformerCarDataset(preprocess_folder=train_data_dir, reprocess=False, mpc_aug=(NUM_AUGMENTATION > 0))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=my_collate_fn)

            val_dataset = TransformerCarDataset(preprocess_folder=val_data_dir, reprocess=False, mpc_aug=(NUM_AUGMENTATION > 0))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=my_collate_fn)

        else:
            train_dataset = GnnCarDataset(preprocess_folder=train_data_dir, mlp=False, mpc_aug=(NUM_AUGMENTATION > 0))
            train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            val_dataset = GnnCarDataset(preprocess_folder=val_data_dir, mlp=False, mpc_aug=(NUM_AUGMENTATION > 0))
            val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

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


def load_yaw_dict():
    with open(YAW_DICT_PATH, "rb") as f:
        yaw_dict = pkl.load(f)
        rename = {
            "left_up": 10 * math.atan2(0, 1) + math.atan2(-1, 0),
            "left_right": 10 * math.atan2(0, 1) + math.atan2(0, 1),
            "left_down": 10 * math.atan2(0, 1) + math.atan2(1, 0),
            "up_right": 10 * math.atan2(1, 0) + math.atan2(0, -1),
            "up_down": 10 * math.atan2(1, 0) + math.atan2(1, 0),
            "up_left": 10 * math.atan2(1, 0) + math.atan2(0, 1),
            "right_down": 10 * math.atan2(0, -1) + math.atan2(1, 0),
            "right_left": 10 * math.atan2(0, -1) + math.atan2(0, -1),
            "right_up": 10 * math.atan2(0, -1) + math.atan2(-1, 0),
            "down_left": 10 * math.atan2(-1, 0) + math.atan2(0, -1),
            "down_up": 10 * math.atan2(-1, 0) + math.atan2(-1, 0),
            "down_right": 10 * math.atan2(-1, 0) + math.atan2(0, 1),
        }

        yaw_dict = {rename.get(k, k): v for k, v in yaw_dict.items()}
        return yaw_dict


def train_one_epoch(
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
    is_transformer=False,
):
    """Performs an epoch of model training

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

    if is_transformer:
        return transformer_train_one_epoch(
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
            start_prediction_time,
        )
    else:
        return gnn_train_one_epoch(
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
            start_prediction_time,
        )


def evaluate(
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
    is_transformer=False,
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
    if is_transformer:
        return transformer_evaluate(
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
            start_prediction_time,
        )
    else:
        return gnn_evaluate(
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
            start_prediction_time,
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
    is_transformer = train_config.is_transformer

    push_to_hf = train_config.push_to_hf
    hf_project_path = train_config.hf_project_path

    train_loader, val_loader = init_dataloaders(
        path_config.train_data_dir, path_config.val_data_dir, train_config.batch_size, is_transformer=is_transformer
    )
    model, optimizer = init_model(
        path_config.copy_model_config_path,
        train_config.optimizer,
        device,
        train_config.lr,
        train_config.weight_decay,
    )

    y_x_mean, y_x_std = pkl.load(open(os.path.join(DATA_PATH, "csv", Y_X_DISTR_FILE), "rb"))
    y_y_mean, y_y_std = pkl.load(open(os.path.join(DATA_PATH, "csv", Y_Y_DISTR_FILE), "rb"))

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
    yaw_values = pad_sequence(values_list, batch_first=True, padding_value=10000000)

    process_conection.send(1)  # tells parent process that everything is inited
    process_conection.close()

    for epoch in tqdm(range(0, epochs)):
        epoch_loss = train_one_epoch(
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
            y_x_mean,
            y_x_std,
            y_y_mean,
            y_y_std,
            start_prediction_time,
            is_transformer=is_transformer,
        )
        loss_logger.add_metric_points([epoch], [epoch * len(train_loader)], [epoch_loss])

        if epoch % metrics_log_epoch_frequency == 0:
            ade, fde, mr, collision_rate, val_loss, collision_penalties = evaluate(
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
                y_x_mean,
                y_x_std,
                y_y_mean,
                y_y_std,
                start_prediction_time,
                is_transformer=is_transformer,
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

        if save_last_checkpoints and (epoch == epochs - 1):
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
