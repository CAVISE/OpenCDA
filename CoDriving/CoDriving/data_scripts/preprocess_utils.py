import pickle
import numpy as np
import pandas as pd
import torch
import os


from CoDriving.data_scripts.dataset import (
    MPC_Block,
    adjust_future_deltas,
    rotation_matrix_with_allign_to_Y,
    rotation_matrix_with_allign_to_X,
    transform_sumo2carla,
)
from CoDriving.data_scripts.data_config.data_config import NUM_PREDICT, OBS_LEN, PRED_LEN, ALLIGN_INITIAL_DIRECTION_TO_X
from CoDriving.data_scripts.utils.feature_utils import get_intention_from_vehicle_id


def min_max_normalize(x, mmax, mmin=0):
    return (x - mmin) / (mmax - mmin)


def process_file(
    csv_folder: str,
    csv_file: str,
    preprocess_folder: str,
    intentuion_config,
    n_mpc_aug,
    normalize,
):
    """
    Docstring for process_file

    :param csv_folder: csv folder with csv data files
    :type csv_folder: str
    :param csv_file: name of csv file in folder
    :type csv_file: str
    :param preprocess_folder: folder for storing preprocessed data in pkls
    :type preprocess_folder: str
    :param intentuion_config: path to intention config file
    :param n_mpc_aug: number of mpc augmentations
    :param normalize: True if normalization needed
    :param allign_initial_direction_to_x: if True: in carla coordinate system rotate coordanate system so +X to be direction of motion of car \
          else rotate coordanate system so +Y to be direction of motion of car
    """
    df = pd.read_csv(os.path.join(csv_folder, csv_file))
    all_features = list()
    for track_id, remain_df in df.groupby("TRACK_ID"):
        if len(remain_df) >= (OBS_LEN + PRED_LEN):
            coords = remain_df[["X", "Y", "speed", "yaw"]].values
            coords[:, 3] = np.deg2rad(coords[:, 3])
            transform_sumo2carla(coords)
            intention = get_intention_from_vehicle_id(track_id, intentuion_config)[:3]
            features = np.hstack((coords, intention * np.ones((coords.shape[0], 3))))
            all_features.append(features)

    num_rows = features.shape[0]
    # [vehicle, steps(obs+pred), 7]: [x, y, speed, yaw, intent, intent, intent]
    all_features = np.array(all_features)
    acc_delta_padding = np.empty((all_features.shape[0], all_features.shape[1], 2))
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
        y = (all_features[:, row + 1 : row + 1 + NUM_PREDICT, :2] - all_features[:, row : row + 1, :2]).transpose(
            0, 2, 1
        )  # [vehicle, PRED_LEN, 2] -> [vehicle, 2, PRED_LEN]

        if ALLIGN_INITIAL_DIRECTION_TO_X:
            rotations = np.array([rotation_matrix_with_allign_to_X(x[i][3]) for i in range(x.shape[0])])  # [vehicle, 2, 2]
        else:
            rotations = np.array([rotation_matrix_with_allign_to_Y(x[i][3]) for i in range(x.shape[0])])  # [vehicle, 2, 2]

        # [vehicle, 2, PRED_LEN], transform y into local coordinate system
        y = rotations @ y
        y = y.transpose(0, 2, 1)  # [vehicle, PRED_LEN, 2]

        # use MPC to compute acc and delta
        curr_states = all_features[:, row, :4]  # [vehicle, 4]
        # [vehicle, PRED_LEN, 4], [x, y, speed, yaw]
        future_states = all_features[:, row + 1 : row + 1 + NUM_PREDICT, :4]
        adjust_future_deltas(curr_states, future_states)
        # [vehicle, PRED_LEN, 2], [acc, delta]
        acc_delta_old = all_features[:, row + 1 : row + 1 + NUM_PREDICT, -2:]
        shifted_curr, mpc_output = MPC_Block(
            curr_states, future_states, acc_delta_old, noise_range=0
        )  # [vehicle, 4], [vehicle, PRED_LEN, 6]: [x, y, v, yaw, acc, delta]
        # store the control vector to accelerate future MPC opt
        all_features[:, row + 1 : row + 1 + NUM_PREDICT, -2:] = mpc_output[:, :, -2:]
        speed = all_features[:, row + 1 : row + 1 + NUM_PREDICT, 2:3]  # [vehicle, PRED_LEN, 1]

        if ALLIGN_INITIAL_DIRECTION_TO_X:
            yaw = (
                all_features[:, row + 1 : row + 1 + NUM_PREDICT, 3:4] - all_features[:, row : row + 1, 3:4]
            )  # [vehicle, PRED_LEN, 1], align the initial direction to +X
        else:
            yaw = (
                all_features[:, row + 1 : row + 1 + NUM_PREDICT, 3:4] - all_features[:, row : row + 1, 3:4] + np.pi / 2
            )  # [vehicle, PRED_LEN, 1], align the initial direction to +Y

        # [vehicle, PRED_LEN*6]
        y = np.concatenate((y, speed, yaw, mpc_output[:, :, -2:]), axis=2).reshape(num_cars, -1)
        data = (
            torch.tensor(x, dtype=torch.float),
            torch.tensor(y, dtype=torch.float),
            edge_index,
            torch.tensor([row]),
        )
        # x: [vehicle, 7]: [x_0, y_0, speed_0, yaw_0, intent, intent, intent]
        # y: [vehicle, PRED_LEN * 6]: [[x_1, y_1, v_1, yaw_1, acc_1, delta_1, x_2, y_2...], ...]
        with open(
            f"{preprocess_folder}/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}-0.pkl",
            "wb",
        ) as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # пока думаю, не нужна аугментация, во-первый результаты странные в данных, во-вторых чтоб искючить число параметров для подбора
        return
        for a in range(n_mpc_aug):
            shifted_curr, mpc_output = MPC_Block(
                curr_states, future_states, acc_delta_old, noise_range=noise_range
            )  # [vehicle, 4], [vehicle, PRED_LEN, 6]: [x, y, v, yaw, acc, delta]
            x_argumented = x.copy()
            x_argumented[:, :2] = shifted_curr[:, :2]
            y = (mpc_output[:, :, :2] - np.expand_dims(shifted_curr[:, :2], axis=1)).transpose(0, 2, 1)  # [vehicle, 2, PRED_LEN]
            y = rotations @ y
            y = y.transpose(0, 2, 1)  # [vehicle, PRED_LEN, 2]

            if ALLIGN_INITIAL_DIRECTION_TO_X:
                mpc_output[:, :, 3:4] = mpc_output[:, :, 3:4] - all_features[:, row : row + 1, 3:4]
            else:
                mpc_output[:, :, 3:4] = mpc_output[:, :, 3:4] - all_features[:, row : row + 1, 3:4] + np.pi / 2

            # [vehicle, PRED_LEN, 6]
            y = np.concatenate((y, mpc_output[:, :, 2:]), axis=-1)
            y = y.reshape(num_cars, -1)
            data = (
                torch.tensor(x_argumented, dtype=torch.float),
                torch.tensor(y, dtype=torch.float),
                edge_index,
                torch.tensor([row]),
            )
            with open(
                f"{preprocess_folder}/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}-{a + 1}.pkl",
                "wb",
            ) as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
