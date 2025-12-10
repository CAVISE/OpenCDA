import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch

from add_path import *

from CoDriving.dataset_scripts.dataset import MPC_Block, adjust_future_deltas, rotation_matrix, transform_sumo2carla
from CoDriving.config.config import NUM_PREDICT, OBS_LEN, PRED_LEN
from CoDriving.dataset_scripts.utils.feature_utils import get_intention_from_vehicle_id
from data_config import *

import concurrent.futures
from multiprocessing import Value, Lock


def process_file(csv_file, intentuion_config):
  df = pd.read_csv(os.path.join(csv_folder, csv_file))
  all_features = list()
  for track_id, remain_df in df.groupby("TRACK_ID"):
    if len(remain_df) >= (OBS_LEN + PRED_LEN):
      coords = remain_df[["X", "Y", "speed", "yaw"]].values
      coords[:, 3] = np.deg2rad(coords[:, 3])
      transform_sumo2carla(coords)
      intention = get_intention_from_vehicle_id(
          track_id, intentuion_config)[:3]
      features = np.hstack((coords, intention * np.ones((coords.shape[0], 3))))
      all_features.append(features)

  num_rows = features.shape[0]
  # [vehicle, steps(obs+pred), 7]: [x, y, speed, yaw, intent, intent, intent]
  all_features = np.array(all_features)
  acc_delta_padding = np.empty(
      (all_features.shape[0], all_features.shape[1], 2))
  acc_delta_padding[:] = np.nan
  all_features = np.concatenate(
      (all_features, acc_delta_padding), axis=-1
  )  # [vehicle, steps, 9]: [x, y, speed, yaw, intent, intent, intent, acc, delta]
  num_cars = len(all_features)
  edges = [[x, y] for x in range(num_cars) for y in range(num_cars)]
  edge_index = torch.tensor(edges, dtype=torch.long).T  # [2, edge]
  noise_range = 3.0

  # for each timestep, create an interaction graph
  for row in range(0, num_rows - NUM_PREDICT):
    x = all_features[:, row, :7]  # [vehicle, 7]

    # translate and then rotate Gt
    y = (all_features[:, row + 1: row + 1 + NUM_PREDICT, :2] - all_features[:, row: row + 1, :2]).transpose(
        0, 2, 1
    )  # [vehicle, PRED_LEN, 2] -> [vehicle, 2, PRED_LEN]
    rotations = np.array([rotation_matrix(x[i][3])
                         for i in range(x.shape[0])])  # [vehicle, 2, 2]
    # [vehicle, 2, PRED_LEN], transform y into local coordinate system
    y = rotations @ y
    y = y.transpose(0, 2, 1)  # [vehicle, PRED_LEN, 2]

    # use MPC to compute acc and delta
    curr_states = all_features[:, row, :4]  # [vehicle, 4]
    # [vehicle, PRED_LEN, 4], [x, y, speed, yaw]
    future_states = all_features[:, row + 1: row + 1 + NUM_PREDICT, :4]
    adjust_future_deltas(curr_states, future_states)
    # [vehicle, PRED_LEN, 2], [acc, delta]
    acc_delta_old = all_features[:, row + 1: row + 1 + NUM_PREDICT, -2:]
    shifted_curr, mpc_output = MPC_Block(
        curr_states, future_states, acc_delta_old, noise_range=0
    )  # [vehicle, 4], [vehicle, PRED_LEN, 6]: [x, y, v, yaw, acc, delta]
    # store the control vector to accelerate future MPC opt
    all_features[:, row + 1: row + 1 +
                 NUM_PREDICT, -2:] = mpc_output[:, :, -2:]
    speed = all_features[:, row + 1: row + 1 +
                         NUM_PREDICT, 2:3]  # [vehicle, PRED_LEN, 1]
    yaw = (
        all_features[:, row + 1: row + 1 + NUM_PREDICT, 3:4] -
        all_features[:, row: row + 1, 3:4] + np.pi / 2
    )  # [vehicle, PRED_LEN, 1], align the initial direction to +y
    # [vehicle, PRED_LEN*6]
    y = np.concatenate(
        (y, speed, yaw, mpc_output[:, :, -2:]), axis=2).reshape(num_cars, -1)
    data = (torch.tensor(x, dtype=torch.float), torch.tensor(
        y, dtype=torch.float), edge_index, torch.tensor([row]))
    # x: [vehicle, 7]: [x_0, y_0, speed_0, yaw_0, intent, intent, intent]
    # y: [vehicle, PRED_LEN * 6]: [[x_1, y_1, v_1, yaw_1, acc_1, delta_1, x_2, y_2...], ...]
    with open(f"{preprocess_folder}/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}-0.pkl", "wb") as handle:
      pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for a in range(n_mpc_aug):
      shifted_curr, mpc_output = MPC_Block(
          curr_states, future_states, acc_delta_old, noise_range=noise_range
      )  # [vehicle, 4], [vehicle, PRED_LEN, 6]: [x, y, v, yaw, acc, delta]
      x_argumented = x.copy()
      x_argumented[:, :2] = shifted_curr[:, :2]
      y = (mpc_output[:, :, :2] - np.expand_dims(shifted_curr[:,
           :2], axis=1)).transpose(0, 2, 1)  # [vehicle, 2, PRED_LEN]
      y = rotations @ y
      y = y.transpose(0, 2, 1)  # [vehicle, PRED_LEN, 2]
      mpc_output[:, :, 3:4] = mpc_output[:, :, 3:4] - \
          all_features[:, row: row + 1, 3:4] + np.pi / 2
      # [vehicle, PRED_LEN, 6]
      y = np.concatenate((y, mpc_output[:, :, 2:]), axis=-1)
      y = y.reshape(num_cars, -1)
      data = (torch.tensor(x_argumented, dtype=torch.float), torch.tensor(
          y, dtype=torch.float), edge_index, torch.tensor([row]))
      with open(f"{preprocess_folder}/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}-{a + 1}.pkl", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with lock:
    completed_tasks_amount.value += 1
    print(f"{csv_file} done {completed_tasks_amount.value}/{len(csv_files)}")


if __name__ == "__main__":
  cpu_amount = os.cpu_count()

  parser = argparse.ArgumentParser(description="")
  parser.add_argument("--csv_folder", type=str, help="path to the data set (*.csv)", default="csv/train")
  parser.add_argument("--pkl_folder", type=str, help="path to the preprocessed data (*.pkl)", default="csv/train_pre")
  parser.add_argument("--num_mpc_aug", type=int, help="number of MPC augmentation", default=2)
  parser.add_argument("--processes", type=int, help=f"amount of processes(max: {cpu_amount})", default=cpu_amount)
  parser.add_argument(
    "--intention_config",
    type=str,
    help="Name of file with routes and intentins. It must be in sumo/intentions/ directory",
    default="simple_separate_10m_intentions.json",
  )
  args = parser.parse_args()

  csv_folder = args.csv_folder
  csv_folder = os.path.join(DATA_PATH, csv_folder)

  preprocess_folder = args.pkl_folder
  preprocess_folder = os.path.join(DATA_PATH, preprocess_folder)

  os.makedirs(preprocess_folder, exist_ok=True)
  n_mpc_aug = args.num_mpc_aug
  processes = args.processes
  intention_config = args.intention_config

  completed_tasks_amount = Value("i", 0)
  lock = Lock()

  csv_files = [i for i in os.listdir(
      csv_folder) if os.path.splitext(i)[1] == ".csv"]
  intention_config_path = os.path.join(
      DATA_PATH, "sumo", "intentions", intention_config)

  print("Processing started")
  executor = concurrent.futures.ProcessPoolExecutor(max_workers=processes)
  try:
    futures = [executor.submit(
        process_file, file, intention_config_path) for file in csv_files]
    for future in concurrent.futures.as_completed(futures):
      try:
        future.result()
      except Exception as e:
        print(f"Error: {e}")
    print("Done")
  except KeyboardInterrupt:
    print("\nInterrupted by the user. Termination of tasks has begun. It may take some time.")
    print("!!!DON'T INTERRUPT IT, JUST WAIT!!!")
    for f in futures:
      f.cancel()
    sys.exit(1)
  finally:
    executor.shutdown()
