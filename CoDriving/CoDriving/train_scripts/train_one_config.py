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

from CoDriving.models.model_factory import ModelFactory
from CoDriving.dataset_scripts.dataset import CarDataset, rotation_matrix_back
from CoDriving.dataset_scripts.metrics_logger import MetricLogger
# from CoDriving.config.config import DT, OBS_LEN, PRED_LEN


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
    self.loss_logs_path = os.path.join(self.logs_dir, 'train_loss_logs.csv')
    self.logs_plot_path = os.path.join(self.logs_dir, 'train_loss_logs.png')

    self.copy_train_config_path = os.path.join(self.exp_path, train_config_name)
    self.copy_model_config_path = os.path.join(self.exp_path, model_config_name)
    self.model_checkpoints_dir = os.path.join(self.exp_path, 'checkpoints')
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


def remove_files_in_directory(directory_path: str):
  """
  Removes all files within a specified directory,
  leaving subdirectories and the main directory intact.
  """
  if not os.path.isdir(directory_path):
    print(f"Error: '{directory_path}' is not a valid directory.")
    return

  for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
      try:
        os.remove(file_path)
        print(f"Removed file: {file_path}")

      except OSError as e:
        print(f"Error removing file {file_path}: {e}")


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

    if (isinstance(opt, type) and name.lower() == optimizer_name):
      return opt

  raise ValueError(f"Optimizer '{optimizer_name}' not found")


def init_dataloaders(train_data_dir: str, val_data_dir: str, batch_size):
  try:
    train_dataset = CarDataset(preprocess_folder=train_data_dir, mlp=False, mpc_aug=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    val_dataset = CarDataset(preprocess_folder=val_data_dir, mlp=False, mpc_aug=True)
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


def train_one_epoch(
  model,
  device,
  data_loader,
  optimizer,
  step_weights_factor,
  dist_threshold,
  collision_penalty,
  collision_penalty_factor,
):
  """Performs an epoch of model training.

  Parameters:
  model (nn.Module): Model to be trained.
  device (torch.Device): Device used for training.
  data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
  optimizer (torch.optim.Optimizer): Optimizer used to update model.
  train_config: object of config of train process.

  Returns:
  float: Avg loss for epoch.
  """
  step_weights = torch.ones(30, device=device)
  step_weights[:5] *= step_weights_factor
  step_weights[0] *= step_weights_factor

  model.train()
  total_loss = 0

  for batch in data_loader:
    batch = batch.to(device)
    optimizer.zero_grad()
    # [x, y, v, yaw, intention(3-bit)] -> [x, y, intention]
    out = model(batch.x[:, [0, 1, 4, 5, 6]], batch.edge_index)
    out = out.reshape(-1, 30, 2)  # [v, pred, 2]
    out = out.permute(0, 2, 1)  # [v, 2, pred]
    yaw = batch.x[:, 3].detach().cpu().numpy()
    rotations = torch.stack([rotation_matrix_back(yaw[i]) for i in range(batch.x.shape[0])]).to(out.device)
    out = torch.bmm(rotations, out).permute(0, 2, 1)  # [v, pred, 2]
    out += batch.x[:, [0, 1]].unsqueeze(1)
    # [x, y, v, yaw, acc, steering]
    gt = batch.y.reshape(-1, 30, 6)[:, :, [0, 1]]
    error = ((gt - out).square().sum(-1) * step_weights).sum(-1)
    loss = (batch.weights * error).nanmean()

    if collision_penalty:
      mask = batch.edge_index[0, :] < batch.edge_index[1, :]
      _edge = batch.edge_index[:, mask].T  # [edge',2]
      dist = torch.linalg.norm(out[_edge[:, 0]] - out[_edge[:, 1]], dim=-1)
      dist = dist_threshold - dist[dist < dist_threshold]

      if dist.numel(): # there are can be no cars with small distanses so it will be empty
        _collision_penalty = dist.square().mean()
        loss += _collision_penalty * collision_penalty_factor

    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  return total_loss / len(data_loader)


def evaluate(
  model,
  device,
  data_loader,
  step_weights_factor,
  dist_threshold,
  mr_threshold,
  collision_penalty_factor,
):
  """Performs an epoch of model training.

  Parameters:
  model (nn.Module): Model to be trained.
  device (torch.Device): Device used for training.
  data_loader (torch.utils.data.DataLoader): Data loader containing all batches.

  Returns:
  list of evaluation metrics (including ADE, FDE, etc.).
  """
  step_weights = torch.ones(30, device=device)
  step_weights[:5] *= step_weights_factor
  step_weights[0] *= step_weights_factor

  model.eval()
  ade, fde = [], []
  n_edge, n_collision = [], []
  val_losses, collision_penalties = [], []

  with torch.no_grad():
    for batch in data_loader:
      batch = batch.to(device)
      out = model(batch.x[:, [0, 1, 4, 5, 6]], batch.edge_index)
      out = out.reshape(-1, 30, 2)  # [v, 30, 2]

      out = out.permute(0, 2, 1)  # [v, 2, pred]
      yaw = batch.x[:, 3].detach().cpu().numpy()
      rotations = torch.stack([rotation_matrix_back(yaw[i]) for i in range(batch.x.shape[0])]).to(out.device)
      out = torch.bmm(rotations, out).permute(0, 2, 1)  # [v, pred, 2]
      out += batch.x[:, [0, 1]].unsqueeze(1)

      gt = batch.y.reshape(-1, 30, 6)[:, :, [0, 1]]
      _error = (gt - out).square().sum(-1)
      error = _error.clone() ** 0.5
      _error = (_error * step_weights).sum(-1)
      val_loss = (batch.weights * _error).nanmean()
      val_losses.append(val_loss)
      fde.append(error[:, -1])
      ade.append(error.mean(dim=-1))

      mask = batch.edge_index[0, :] < batch.edge_index[1, :]
      _edge = batch.edge_index[:, mask].T  # [edge',2]
      dist = torch.linalg.norm(out[_edge[:, 0]] - out[_edge[:, 1]], dim=-1)  # [edge, 30]

      collision_penalty = dist_threshold - dist[dist < dist_threshold]
      if collision_penalty.numel(): # there are can be no cars with small distanses so it will be empty
        collision_penalty = collision_penalty.square().mean() * collision_penalty_factor
        collision_penalties.append(collision_penalty)

      dist = torch.min(dist, dim=-1)[0]
      n_edge.append(len(dist))
      n_collision.append((dist < 2).sum().item())

      # out = out.permute(0,2,1)    # [v, 2, pred]
      # yaw = batch.x[:,3].detach().cpu().numpy()
      # rotations = torch.stack([rotation_matrix_back(yaw[i])  for i in range(batch.x.shape[0])]).to(out.device)
      # out = torch.bmm(rotations, out).permute(0,2,1)       # [v, pred, 2]
      # out += batch.x[:,[7,8]].unsqueeze(1)
      # # gt = batch.y.reshape(-1,50,6)[:,:,[0,1]]
      # # error = ((gt-out).square().sum(-1) * step_weights).sum(-1)
      # # loss = (batch.weights * error).nanmean()

      # mask = (batch.edge_index[0,:] < batch.edge_index[1,:])
      # _edge = batch.edge_index[:, mask].T   # [edge',2]
      # # pos1 = torch.stack([out[_edge[i, 0]] for i in range(_edge.shape[0])])
      # # pos2 = torch.stack([out[_edge[i, 1]] for i in range(_edge.shape[0])])
      # # dist = torch.linalg.norm(pos1 - pos2, dim=-1)
      # dist = torch.linalg.norm(out[_edge[:,0]] - out[_edge[:,1]], dim=-1) # [edge, 50]

  ade = torch.cat(ade).mean()
  fde = torch.cat(fde)
  mr = ((fde > mr_threshold).sum() / len(fde)).item()
  fde = fde.mean()
  collision_rate = sum(n_collision) / sum(n_edge)
  collision_penalties = torch.tensor(collision_penalties).mean()
  val_losses = torch.tensor(val_losses).mean()

  return ade.item(), fde.item(), mr, collision_rate, val_losses.item(), collision_penalties.item()


METRICS = ['ade', 'fde', 'mr', 'collision_rate', 'val_loss', 'collision_penalties']
# METRICS = ['ade', 'fde', 'mr', 'collision_rate', 'val_loss']


def train_one_config(train_config_path: str, model_config_path: str, expirements_path: str, data_path: str):
  train_config = read_config_file(train_config_path)
  path_config = PathConfig(
    expirements_path,
    data_path,
    train_config.exp_id,
    train_config_path,
    model_config_path,
    train_config.logs_dir_path,
    train_config.train_data_dir,
    train_config.val_data_dir
  )
  path_config.create_directories()
  copy_file(train_config_path, path_config.copy_train_config_path)
  copy_file(model_config_path, path_config.copy_model_config_path)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  metrics_log_epoch_frequency = train_config.metrics_log_epoch_frequency
  epochs = train_config.epoch
  collision_penalty = train_config.collision_penalty

  push_to_hf = train_config.push_to_hf
  hf_project_path = train_config.hf_project_path

  train_loader, val_loader = init_dataloaders(path_config.train_data_dir, path_config.val_data_dir, train_config.batch_size)
  model, optimizer = init_model(path_config.copy_model_config_path, train_config.optimizer, device, train_config.lr, train_config.weight_decay)
  if train_config.clear_exp_checkpoints:
    remove_files_in_directory(path_config.model_checkpoints_dir)

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
      os.path.join(path_config.logs_dir, f'{metric}_logs.csv'),
      os.path.join(path_config.logs_dir, f'{metric}_logs.png'),
      enable_plot=True)
    metric_loggers[metric].clear_file()

  for epoch in tqdm(range(0, epochs)):
    epoch_loss = train_one_epoch(
      model, 
      device, 
      train_loader, 
      optimizer, 
      train_config.step_weights_factor, 
      train_config.dist_threshold, 
      train_config.collision_penalty,
      train_config.collision_penalty_factor
    )
    loss_logger.add_metric_points([epoch], [epoch * len(train_loader)], [epoch_loss])

    if epoch % metrics_log_epoch_frequency == 0:
      ade, fde, mr, collision_rate, val_loss, collision_penalties = evaluate(
        model, 
        device, 
        val_loader, 
        train_config.step_weights_factor, 
        train_config.dist_threshold, 
        train_config.collision_penalty,
        train_config.collision_penalty_factor
      )
      epoch_metrics = {'ade': ade, 'fde': fde, 'mr': mr, 'collision_rate': collision_rate, 'val_loss': val_loss, 'collision_penalties': collision_penalties}
      record.append(epoch_metrics)

      torch.save(
        model.state_dict(),
        os.path.join(path_config.model_checkpoints_dir, f"model_{'wp' if collision_penalty else 'np'}_{train_config.exp_id}_{str(epoch).zfill(4)}.pth")
      )

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

  loss_logger.plot_metric()

  record_df = pd.DataFrame(record)
  for metric in METRICS:
    metric_loggers[metric].add_metric_points(
      np.arange(0, epochs, step=metrics_log_epoch_frequency),
      np.arange(0, len(val_loader) * epochs, step=len(val_loader) * metrics_log_epoch_frequency),
      record_df[metric])
    metric_loggers[metric].plot_metric()

  if push_to_hf:
    best_model_path = os.path.join(path_config.model_checkpoints_dir, f"model_{'wp' if collision_penalty else 'np'}_{train_config.exp_id}_{str(best_epoch).zfill(4)}.pth")
    best_state_dict = torch.load(best_model_path)
    model.load_state_dict(best_state_dict)
    model.push_to_hub(hf_project_path)

  # pkl_file = f"model_{'mlp' if mlp else 'gnn'}_{'wp' if collision_penalty else 'np'}_{exp_id}_e3.pkl"
  # # pkl_file = f"model_{'mlp' if mlp else 'gnn'}_mtl_sumo_0911_e3.pkl"
  # with open(f"{model_path}/{pkl_file}", "wb") as handle:
  #   pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
