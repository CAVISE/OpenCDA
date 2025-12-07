import argparse
import os
import pickle

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pandas as pd

from CoDriving.models.model_factory import ModelFactory
from CoDriving.dataset_scripts.dataset import CarDataset
from CoDriving.dataset_scripts.metrics_logger import MetricLogger
from CoDriving.config.config import DT, OBS_LEN, PRED_LEN
from data_config import *


os.makedirs(EXPIREMENTS_PATH, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_len, pred_len, dt = OBS_LEN, PRED_LEN, DT

parser = argparse.ArgumentParser(description="")
parser.add_argument("--train_folder", type=str, help="path to the training set", default="csv/train_pre")
parser.add_argument("--val_folder", type=str, help="path to the validation set", default="csv/val_pre")
parser.add_argument("--epoch", type=int, help="number of total training epochs", default=20)
parser.add_argument("--exp_id", type=str, help="experiment ID", default="test")
parser.add_argument("--batch_size", type=int, help="batch size", default=32)
args = parser.parse_args()

batch_size = args.batch_size  # 8000
train_folder = args.train_folder
train_folder = os.path.join(DATA_PATH, train_folder)
val_folder = args.val_folder
val_folder = os.path.join(DATA_PATH, val_folder)

exp_id = args.exp_id
exp_path = os.path.join(EXPIREMENTS_PATH, exp_id)
model_path = os.path.join(exp_path, 'checkpoints')
os.makedirs(model_path, exist_ok=True)

mlp = False
collision_penalty = False

train_dataset = CarDataset(preprocess_folder=train_folder, mlp=False, mpc_aug=True)
val_dataset = CarDataset(preprocess_folder=val_folder, mlp=False, mpc_aug=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

model_config_path = os.path.join(exp_path, 'model_config.yaml')
model = ModelFactory.create_model(model_config_path)
print(model)


def rotation_matrix_back(yaw):
  """
  Rotate back.
  https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
  """
  rotation = np.array([
      [np.cos(-np.pi / 2 + yaw), -np.sin(-np.pi / 2 + yaw)],
      [np.sin(-np.pi / 2 + yaw), np.cos(-np.pi / 2 + yaw)]
  ])
  rotation = torch.tensor(rotation).float()
  return rotation


def train(model, device, data_loader, optimizer, loss_logger, epoch_num, collision_penalty=False):
  """Performs an epoch of model training.

  Parameters:
  model (nn.Module): Model to be trained.
  device (torch.Device): Device used for training.
  data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
  optimizer (torch.optim.Optimizer): Optimizer used to update model.
  collision_penalty: set it to True if you want to use collision penalty.

  Returns:
  float: Total loss for epoch.
  """
  model.train()
  total_loss = 0

  step_weights = torch.ones(30, device=device)
  step_weights[:5] *= 5
  step_weights[0] *= 5
  dist_threshold = 4

  log_loss_frequency = 100
  avg_log_loss = 0

  epoch_timestamp = 1
  data_loader_len = len(data_loader)

  for batch in data_loader:
    batch = batch.to(device)
    optimizer.zero_grad()
    out = model(batch.x[:, [0, 1, 4, 5, 6]], batch.edge_index)  # [x, y, v, yaw, intention(3-bit)] -> [x, y, intention]
    out = out.reshape(-1, 30, 2)  # [v, pred, 2]
    out = out.permute(0, 2, 1)  # [v, 2, pred]
    yaw = batch.x[:, 3].detach().cpu().numpy()
    rotations = torch.stack([rotation_matrix_back(yaw[i]) for i in range(batch.x.shape[0])]).to(out.device)
    out = torch.bmm(rotations, out).permute(0, 2, 1)  # [v, pred, 2]
    out += batch.x[:, [0, 1]].unsqueeze(1)
    gt = batch.y.reshape(-1, 30, 6)[:, :, [0, 1]]  # [x, y, v, yaw, acc, steering]
    error = ((gt - out).square().sum(-1) * step_weights).sum(-1)
    loss = (batch.weights * error).nanmean()

    if collision_penalty:
      mask = batch.edge_index[0, :] < batch.edge_index[1, :]
      _edge = batch.edge_index[:, mask].T  # [edge',2]
      dist = torch.linalg.norm(out[_edge[:, 0]] - out[_edge[:, 1]], dim=-1)
      dist = dist_threshold - dist[dist < dist_threshold]
      _collision_penalty = dist.square().mean()
      # print(f"loss: {loss.item()}, collision penalty: {collision_penalty.item()}")
      loss += _collision_penalty * 20

    global_timestamp = data_loader_len * epoch_num + epoch_timestamp
    avg_log_loss += loss.item()
    if global_timestamp % log_loss_frequency == 0 or epoch_timestamp == data_loader_len:
      loss_logger.add_metric_points([epoch], [global_timestamp], [avg_log_loss / log_loss_frequency])
      avg_log_loss = 0

    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    epoch_timestamp += 1

  loss_logger.plot_metric()
  return total_loss / len(data_loader)


def evaluate(model, device, data_loader):
  """Performs an epoch of model training.

  Parameters:
  model (nn.Module): Model to be trained.
  device (torch.Device): Device used for training.
  data_loader (torch.utils.data.DataLoader): Data loader containing all batches.

  Returns:
  list of evaluation metrics (including ADE, FDE, etc.).
  """
  step_weights = torch.ones(30, device=device)
  step_weights[:5] *= 5
  step_weights[0] *= 5
  dist_threshold = 4
  mr_threshold = 4
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
      collision_penalty = collision_penalty.square().mean() * 20
      collision_penalties.append(collision_penalty)

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
      dist = torch.min(dist, dim=-1)[0]
      n_edge.append(len(dist))
      n_collision.append((dist < 2).sum().item())

  ade = torch.cat(ade).mean()
  fde = torch.cat(fde)
  mr = ((fde > mr_threshold).sum() / len(fde)).item()
  fde = fde.mean()
  collision_rate = sum(n_collision) / sum(n_edge)
  collision_penalties = torch.tensor(collision_penalties).mean()
  val_losses = torch.tensor(val_losses).mean()

  return ade.item(), fde.item(), mr, collision_rate, val_losses.item(), collision_penalties.item()


model = model.to(device)

min_ade = 1e6
min_fde = 1e6
best_epoch = 0
patience = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
record = []

logs_dir = os.path.join(exp_path, 'logs')
loss_logfile_path = os.path.join(logs_dir, 'train_loss_logs.csv')
loss_plotfile_path = os.path.join(logs_dir, 'train_loss_logs.png')

loss_logger = MetricLogger(loss_logfile_path, loss_plotfile_path, enable_plot=True)
loss_logger.clear_file()

metrics_log_frequency = 2
metrics = ['ade', 'fde', 'mr', 'collision_rate', 'val_loss', 'collision_penalties']
metric_loggers = {
    metric: MetricLogger(
        os.path.join(logs_dir, f'{metric}_logs.csv'),
        os.path.join(logs_dir, f'{metric}_logs.png'),
        enable_plot=True
    ) for metric in metrics
}
for metric in metrics:
  metric_loggers[metric].clear_file()

epochs = args.epoch
for epoch in tqdm(range(0, epochs)):
  loss = train(model, device, train_loader, optimizer, loss_logger, epoch)
  if epoch % metrics_log_frequency == 0:
    ade, fde, mr, collision_rate, val_loss, collision_penalties = evaluate(model, device, val_loader)
    epoch_metrics = {'ade': ade, 'fde': fde, 'mr': mr, 'collision_rate': collision_rate, 'val_loss': val_loss, 'collision_penalties': collision_penalties}
    record.append(epoch_metrics)

    print(
        f"Epoch {epoch}: Train Loss: {loss}, ADE: {ade}, FDE: {fde}, MR: {mr}, CR:{collision_rate}, \
            Val_loss: {val_loss}, CP: {collision_penalties}, lr: {optimizer.param_groups[0]['lr']}."
    )
    torch.save(
        model.state_dict(),
        model_path + f"/model_{'mlp' if mlp else 'gnn'}_{'wp' if collision_penalty else 'np'}_{exp_id}_e3_{str(epoch).zfill(4)}.pth",
    )
    model.push_to_hub("kijjjj/test-model1")

    if fde < min_fde:
      min_ade, min_fde = ade, fde
      best_epoch = epoch
      print(" !!! New smallest FDE !!! ")
    elif (epoch - best_epoch) > patience:
      if patience > 1600:  # x16
        print(
            f"{'MLP' if mlp else 'GNN'} earlier stops, Best Epoch: {best_epoch}, Min ADE: {min_ade}, \
                    Min FDE: {min_fde}, MR: {mr}, CR:{collision_rate}."
        )
        break
      else:
        optimizer.param_groups[0]["lr"] *= 0.5
        patience *= 2

record_df = pd.DataFrame(record)
for metric in metrics:
  metric_loggers[metric].add_metric_points(
      np.arange(0, epochs, step=metrics_log_frequency),
      np.arange(0, len(val_loader) * epochs, step=len(val_loader) * metrics_log_frequency),
      record_df[metric])
  metric_loggers[metric].plot_metric()

# pkl_file = f"model_{'mlp' if mlp else 'gnn'}_{'wp' if collision_penalty else 'np'}_{exp_id}_e3.pkl"
# # pkl_file = f"model_{'mlp' if mlp else 'gnn'}_mtl_sumo_0911_e3.pkl"
# with open(f"{model_path}/{pkl_file}", "wb") as handle:
#   pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
