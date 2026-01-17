import pickle
import numpy as np
import pandas as pd
import torch
import os


from CoDriving.data_scripts.data_config.data_config import (
    OBS_LEN,
    NUM_PREDICT,
    ALLIGN_INITIAL_DIRECTION_TO_X,
    NUM_AUGMENTATION,
    CONTROLL_RADIUS,
    VEHICLE_MAX_SPEED,
    NORMALIZE_DATA,
)
from CoDriving.MPC_XY_Frame.MPC_XY_Frame import linear_mpc_control_data_aug
from CoDriving.data_scripts.utils.feature_utils import get_intention_from_vehicle_id


def rotation_matrix_with_allign_to_Y(yaw):
    """
    Make the current direction aligns to +Y axis (In CARLA coordinate system).
    https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
    """
    rotation = np.array(
        [
            [np.cos(np.pi / 2 - yaw), -np.sin(np.pi / 2 - yaw)],
            [np.sin(np.pi / 2 - yaw), np.cos(np.pi / 2 - yaw)],
        ]
    )
    rotation = torch.tensor(rotation).float()
    return rotation


def rotation_matrix_back_with_allign_to_Y(yaw):
    """
    Rotate back (In CARLA coordinate system).
    https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
    """
    rotation = np.array(
        [
            [np.cos(-np.pi / 2 + yaw), -np.sin(-np.pi / 2 + yaw)],
            [np.sin(-np.pi / 2 + yaw), np.cos(-np.pi / 2 + yaw)],
        ]
    )
    rotation = torch.tensor(rotation).float()
    return rotation


def rotation_matrix_with_allign_to_X(yaw):
    """
    Make the current direction aligns to +X axis (In CARLA coordinate system).
    https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
    """
    rotation = np.array(
        [
            [np.cos(-yaw), -np.sin(-yaw)],
            [np.sin(-yaw), np.cos(-yaw)],
        ]
    )
    rotation = torch.tensor(rotation).float()
    return rotation


def rotation_matrix_back_with_allign_to_X(yaw):
    """
    Rotate back (In CARLA coordinate system).
    https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
    """
    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ]
    )
    rotation = torch.tensor(rotation).float()
    return rotation


def adjust_future_deltas(curr_states, future_states):
    """
    The range of delta angle is [-180, 180], in order to avoid the jump, adjust the future delta angles.

    :param curr_states: [vehicle, 4]
    :param future_states: [vehicle, pred_len, 4]
    """

    assert curr_states.shape[0] == future_states.shape[0]
    num_vehicle = curr_states.shape[0]
    num_step = future_states.shape[1]

    for i_vehicle in range(num_vehicle):
        for i_step in range(num_step):
            if (future_states[i_vehicle, i_step, 3] - curr_states[i_vehicle, 3]) < -np.pi:
                future_states[i_vehicle, i_step, 3] += 2 * np.pi
            elif (future_states[i_vehicle, i_step, 3] - curr_states[i_vehicle, 3]) > np.pi:
                future_states[i_vehicle, i_step, 3] -= 2 * np.pi

    return future_states


def transform_coords(coords):
    """
    In-place. Transform coords in both directions: sumo <-> carla \
        ([x_sumo, y_sumo] = [x_carla, -y_carla]).
    expected coords on last dim 
    """
    coords[..., 1] = -coords[..., 1]
    return coords


def transform_sumo2carla_yaw(yaw):
    """
    In-place. In RAD transform yaw from sumo to carla
    """
    yaw -= float(np.deg2rad(90))
    mask = yaw > float(np.deg2rad(180))
    yaw[mask] -= float(np.deg2rad(360))
    return yaw


def transform_carla2sumo_yaw(yaw):
    """
    In-place. In RAD transform yaw from carla to sumo
    """
    yaw += float(np.deg2rad(90))
    mask = yaw < 0
    yaw[mask] += float(np.deg2rad(360))
    return yaw


def transform_sumo2carla(states):
    """
    In-place. In RAD transform from sumo to carla: [x_carla, y_carla, velosity, yaw_carla] = [x_sumo, -y_sumo, velosity, yaw_sumo-90]. \
        yaw_carla in [-90, 270] so if > 180 yaw_carla = yaw_carla - 360. After that yaw_carla in [-180, 180]
    Note:
        - the coordinate system in Carla is more convenient since the angle increases in the direction of rotation from +x to +y, while in sumo this is from +y to +x.
        - the coordinate system in Carla is a left-handed Cartesian coordinate system.

    expected arraylike of size [..., 4]
    """
    transform_coords(states[..., :2])
    transform_sumo2carla_yaw(states[..., 3])
    return states


def transform_carla2sumo(states):
    """
    In-place. In RAD transform from carla to sumo: [x_sumo, y_sumo, velosity, yaw_sumo] = [x_carla, -y_carla, velosity, yaw_carla+90]. \
        yaw_sumo in [-90, 270] so if < 0 yaw_sumo = yaw_sumo + 360. After that yaw_sumo in [0, 360]

    expected arraylike of size [..., 4]
    """
    transform_coords(states[..., :2])
    transform_carla2sumo_yaw(states[..., 3])
    return states


# def transform_coords(coords: np.ndarray):
#     """
#     Transform coords in both directions: sumo <-> carla \
#         ([x_sumo, y_sumo] = [x_carla, -y_carla])
#     """
#     if coords.ndim == 1:
#         coords[1] = -coords[1]

#     elif coords.ndim == 2:
#         coords[:, 1] = -coords[:, 1]
#     else:
#         raise NotImplementedError

#     return coords


# def transform_carla2sumo_yaw(yaw):
#     """
#     In RAD transform yaw from carla to sumo
#     """
#     yaw += np.deg2rad(90)
#     yaw[yaw < np.deg2rad(0)] += np.deg2rad(360)
#     return yaw


# def transform_sumo2carla_yaw(yaw):
#     """
#     In RAD transform yaw from sumo to carla
#     """
#     yaw -= np.deg2rad(90)
#     yaw[yaw > np.deg2rad(180)] -= np.deg2rad(360)
#     return yaw


# def transform_sumo2carla(states: np.ndarray):
#     """
#     In RAD transform from sumo to carla: [x_carla, y_carla, yaw_carla] = [x_sumo, -y_sumo, yaw_sumo-90]. \
#         yaw_carla in [-90, 270] so if > 180 yaw_carla = yaw_carla - 360. After that yaw_carla in [-180, 180]
#     Note:
#         - the coordinate system in Carla is more convenient since the angle increases in the direction of rotation from +x to +y, while in sumo this is from +y to +x.
#         - the coordinate system in Carla is a left-handed Cartesian coordinate system.
#     """
#     if states.ndim == 1:
#         states[:3] = transform_coords(states[:3])
#         states[3] = transform_sumo2carla_yaw(states[3])

#     elif states.ndim == 2:
#         states[:, :3] = transform_coords(states[:, :3])
#         states[:, 3] = transform_sumo2carla_yaw(states[:, 3])
#     else:
#         raise NotImplementedError

#     return states


# def transform_carla2sumo(states: np.ndarray):
#     """
#     In RAD transform from carla to sumo: [x_sumo, y_sumo, yaw_sumo] = [x_carla, -y_carla, yaw_carla+90]. \
#         yaw_sumo in [-90, 270] so if < 0 yaw_sumo = yaw_sumo + 360. After that yaw_sumo in [0, 360]
#     """
#     if states.ndim == 1:
#         states[:3] = transform_coords(states[:3])
#         states[3] = transform_carla2sumo_yaw(states[3])

#     elif states.ndim == 2:
#         states[:, :3] = transform_coords(states[:, :3])
#         states[:, 3] = transform_carla2sumo_yaw(states[:, 3])
#     else:
#         raise NotImplementedError

#     return states


# def transform_sumo2carla_yaw_torch(yaw: torch.Tensor):
#     """
#     Transform yaw from SUMO to CARLA format

#     :param yaw: tensor in radians
#     """
#     yaw_carla = yaw.clone()

#     yaw_carla -= torch.deg2rad(torch.tensor(90.0, device=yaw_carla.device))
#     mask = yaw_carla > torch.deg2rad(torch.tensor(180.0, device=yaw_carla.device))
#     yaw_carla[mask] -= torch.deg2rad(torch.tensor(360.0, device=yaw_carla.device))

#     return yaw_carla


def MPC_Block(
    curr_states: np.ndarray,
    target_states: np.ndarray,
    acc_delta_old: np.ndarray,
    noise_range: float = 0.0,
):
    """
    :param curr_states: [vehicle, 4], [[x, y, speed, yaw], ...]
    :param target_states: [vehicle, pred_len, 4]
    :param acc_delta_old: [vehicle, pred_len, 2]
    :param noise_range: noise on the lateral direction

    :return shifted_curr: [vehicle, 4]
    :return mpc_output: [vehicle, pred_len, 6], [x, y, speed, yaw, acc, delta]
    """

    # acc_delta_new = np.zeros_like(acc_delta_old)
    num_vehicles = curr_states.shape[0]
    pred_len = target_states.shape[1]
    shifted_curr = np.zeros((num_vehicles, 4))
    mpc_output = np.zeros((num_vehicles, pred_len, 6))
    for v in range(num_vehicles):
        shifted_curr[v], mpc_output[v] = MPC_module(curr_states[v], target_states[v], acc_delta_old[v], noise_range)
    return shifted_curr, mpc_output


def MPC_module(
    curr_state_v: np.ndarray,
    target_states_v: np.ndarray,
    acc_delta_old_v: np.ndarray,
    noise_range: float = 0.0,
):
    """
    :param curr_state_v: [4], [x_0, y_0, speed_0, yaw_0]
    :param target_states_v: [pred_len, 4], [[x_1, y_1, speed_1, yaw_1], ...]
    :param acc_delta_old_v: [pred_len, 2], [[acc_1, delta_1], ...]
    :param noise_range: noise on the lateral direction

    :return shifted_curr: [4]
    :return mpc_output: [pred_len, 6]
    """

    acc_delta_old_v[np.isnan(acc_delta_old_v)] = 0.0  # [pred_len, 2]
    a_old = acc_delta_old_v[:, 0].tolist()
    delta_old = acc_delta_old_v[:, 1].tolist()

    if noise_range > 0:
        curr_state_v = curr_state_v.copy()  # avoid add noise in-place
        noise_direction = curr_state_v[3] - np.deg2rad(90)
        # TODO: uniform or Gaussian distribution?
        noise_length = np.random.uniform(low=-1, high=1) * noise_range
        noise = np.array([np.cos(noise_direction), np.sin(noise_direction)]) * noise_length
        curr_state_v[:2] += noise

    curr_state_v = curr_state_v.reshape(1, 4)

    target_states_v = np.concatenate((curr_state_v, target_states_v), axis=0)  # [pred_len+1, 4]
    _curr_state_v = curr_state_v.reshape(-1).tolist()

    target_states_v = target_states_v.T
    a_opt, delta_opt, x_opt, y_opt, v_opt, yaw_opt = linear_mpc_control_data_aug(target_states_v, _curr_state_v, a_old, delta_old)

    mpc_output = np.concatenate(
        (
            x_opt[1:].reshape(-1, 1),
            y_opt[1:].reshape(-1, 1),
            v_opt[1:].reshape(-1, 1),
            yaw_opt[1:].reshape(-1, 1),
            a_opt.reshape(-1, 1),
            delta_opt.reshape(-1, 1),
        ),
        axis=1,
    )

    return curr_state_v.reshape(-1), mpc_output


def normalize(x, circle_boundary: float, new_circle_boundary: float):
    """
    In-place. Normalizing x data from [-circle_boundary, circle_boundary] to [-new_circle_boundary, new_circle_boundary]

    :param x: data
    :param circle_boundary: circle_boundary of given data, circle_boundary is symmetric
    :param new_circle_boundary: new_circle_boundary of given data, new_circle_boundary is symmetric
    """
    diff_ratio = new_circle_boundary / circle_boundary
    x *= diff_ratio
    return x


def normalize_yaw(yaw):
    """
    In-place. Normalizing yaw from [-pi, pi] to [-1, 1]

    :param yaw
    """
    return normalize(yaw, float(np.pi), 1)


def denormalize_yaw(yaw):
    """
    In-place. DEnormalizing yaw from [-1, 1] to [-pi, pi]

    :param yaw
    """
    return normalize(yaw, 1, float(np.pi))


def normalize_coords(coords, circle_boundary: float):
    """
    In-place. Normalizing coords from [-circle_boundary, circle_boundary] to [-1, 1]

    :param coords
    :param circle_boundary: boundary of circle on which cars are controlled on intersection
    """
    return normalize(coords, circle_boundary, 1)


def denormalize_coords(coords, circle_boundary: float):
    """
    In-place. DEnormalizing coords from to [-1, 1] to [-circle_boundary, circle_boundary]

    :param coords
    :param circle_boundary: boundary of circle on which cars are controlled on intersection
    """
    return normalize(coords, 1, circle_boundary)


def normalize_speed(speed, vechicle_max_speed: float):
    """
    In-place. Normalizing speed from [0, vechicle_max_speed] to [0, 1]

    :param speed
    :param vechicle_max_speed: max speed of vechicle on intersection (in current intersection coordinate system)
    """
    return normalize(speed, vechicle_max_speed, 1)


def denormalize_speed(speed, vechicle_max_speed: float):
    """
    In-place. DEnormalizing speed from [0, 1] to [0, vechicle_max_speed]

    :param speed
    :param vechicle_max_speed: max speed of vechicle on intersection (in current intersection coordinate system)
    """
    return normalize(speed, 1, vechicle_max_speed)


def normalize_input_data(x, circle_boundary: float, vechicle_max_speed: float):
    """
    In-place. Normalizing x - input data to [-1, 1] in coords, yaw; [0, 1] in speed

    :param x with shape [vehicle, 7] - for each vehicle [x_0, y_0, speed_0, yaw_0, intent, intent, intent]
    """
    normalize_coords(x[:, :2], circle_boundary)
    normalize_speed(x[:, 2], vechicle_max_speed)
    normalize_yaw(x[:, 3])
    return x


def de_normalize_input_data(x, circle_boundary: float, vechicle_max_speed: float):
    """
    In-place. De normalizing x - input data from [-1, 1] in coords, yaw; [0, 1] in speed

    :param x with shape [vehicle, 7] - for each vehicle [x_0, y_0, speed_0, yaw_0, intent, intent, intent]
    """
    denormalize_coords(x[:, :2], circle_boundary)
    denormalize_speed(x[:, 2], vechicle_max_speed)
    denormalize_yaw(x[:, 3])
    return x


def normalize_target_data(y, circle_boundary: float, vechicle_max_speed: float):
    """
    In-place. Normalizing y - input data to [-1, 1] in coords, yaw; [0, 1] in speed

    :param y with shape [vehicle, NUM_PREDICT, 6] - for each vehicle [[x_1, y_1, v_1, yaw_1, acc_1, delta_1], ...]
    :param circle_boundary: boundary of circle on which cars are controlled on intersection
    :param vechicle_max_speed: max speed of vechicle on intersection (in current intersection coordinate system)
    """
    normalize_coords(y[:, :, :2], circle_boundary)
    normalize_speed(y[:, :, 2], vechicle_max_speed)
    normalize_yaw(y[:, :, 3])
    return y


def de_normalize_target_data(y, circle_boundary, vechicle_max_speed):
    """
    De normalizing y - input data from [-1, 1] in coords, yaw; [0, 1] in speed

    :param y with shape [vehicle, NUM_PREDICT, 6] - for each vehicle [[x_1, y_1, v_1, yaw_1, acc_1, delta_1], ...]
    """
    denormalize_coords(y[:, :, :2], circle_boundary)
    denormalize_speed(y[:, :, 2], vechicle_max_speed)
    denormalize_yaw(y[:, :, 3])
    return y


def preprocess_file(
    csv_folder_path: str,
    csv_file: str,
    preprocess_folder_path: str,
    intentuion_config,
    start_position_config,
):
    """
    Preprocesses csv file and save several pkl files

    :param csv_folder_path: csv folder with csv data files path
    :type csv_folder_path: str
    :param csv_file: name of csv file in folder
    :type csv_file: str
    :param preprocess_folder_path: folder for storing preprocessed data in pkls path
    :type preprocess_folder_path: str
    :param intentuion_config: path to intention config file
    :param start_position_config: path to start positions config
    """
    df = pd.read_csv(os.path.join(csv_folder_path, csv_file))
    all_features = list()

    for track_id, remain_df in df.groupby("TRACK_ID"):
        if len(remain_df) >= (OBS_LEN + NUM_PREDICT):
            coords = remain_df[["X", "Y", "speed", "yaw"]].values
            coords[:, 3] = np.deg2rad(coords[:, 3])
            coords = transform_sumo2carla(coords)
            intention, start_position = get_intention_from_vehicle_id(track_id, intentuion_config, start_position_config)
            features = np.hstack((coords, intention * np.ones((coords.shape[0], 3)), start_position * np.ones((coords.shape[0], 4))))
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
        x = all_features[:, row, :11]  # [vehicle, 11]

        # translate and then rotate Gt
        y = (all_features[:, row + 1 : row + 1 + NUM_PREDICT, :2] - all_features[:, row : row + 1, :2]).transpose(
            0, 2, 1
        )  # [vehicle, NUM_PREDICT, 2] -> [vehicle, 2, NUM_PREDICT]

        if ALLIGN_INITIAL_DIRECTION_TO_X:
            rotations = np.array([rotation_matrix_with_allign_to_X(x[i][3]) for i in range(x.shape[0])])  # [vehicle, 2, 2]
        else:
            rotations = np.array([rotation_matrix_with_allign_to_Y(x[i][3]) for i in range(x.shape[0])])  # [vehicle, 2, 2]

        # [vehicle, 2, NUM_PREDICT], transform y into local coordinate system
        y = rotations @ y
        y = y.transpose(0, 2, 1)  # [vehicle, NUM_PREDICT, 2]

        # use MPC to compute acc and delta
        curr_states = all_features[:, row, :4]  # [vehicle, 4]
        # [vehicle, NUM_PREDICT, 4], [x, y, speed, yaw]
        future_states = all_features[:, row + 1 : row + 1 + NUM_PREDICT, :4]
        future_states = adjust_future_deltas(curr_states, future_states)

        # [vehicle, NUM_PREDICT, 2], [acc, delta]
        acc_delta_old = all_features[:, row + 1 : row + 1 + NUM_PREDICT, -2:]
        shifted_curr, mpc_output = MPC_Block(
            curr_states, future_states, acc_delta_old, noise_range=0
        )  # [vehicle, 4], [vehicle, NUM_PREDICT, 6]: [x, y, v, yaw, acc, delta]
        # store the control vector to accelerate future MPC opt
        all_features[:, row + 1 : row + 1 + NUM_PREDICT, -2:] = mpc_output[:, :, -2:]

        # speed = all_features[:, row + 1 : row + 1 + NUM_PREDICT, 2:3]  # [vehicle, NUM_PREDICT, 1]
        speed = (
            all_features[:, row + 1 : row + 1 + NUM_PREDICT, 2:3] - all_features[:, row : row + 1, 2:3]
        )  # [vehicle, NUM_PREDICT, 1]

        # this is not an angle in local coordinate system this is a yaw with which data point was rotated. BUT for +X allignment theese yaws are the same
        if ALLIGN_INITIAL_DIRECTION_TO_X:
            yaw = (
                all_features[:, row + 1 : row + 1 + NUM_PREDICT, 3:4] - all_features[:, row : row + 1, 3:4]
            )  # [vehicle, NUM_PREDICT, 1], align the initial direction to +X
        else:
            yaw = (
                all_features[:, row + 1 : row + 1 + NUM_PREDICT, 3:4] - all_features[:, row : row + 1, 3:4] + np.pi / 2
            )  # [vehicle, NUM_PREDICT, 1], align the initial direction to +Y

        # [vehicle, NUM_PREDICT, 6]
        y = np.concatenate((y, speed, yaw, mpc_output[:, :, -2:]), axis=2)
        if NORMALIZE_DATA:
            normalize_target_data(y, CONTROLL_RADIUS, VEHICLE_MAX_SPEED)
            normalize_input_data(x, CONTROLL_RADIUS, VEHICLE_MAX_SPEED)

        # [vehicle, NUM_PREDICT*6]
        y = y.reshape(num_cars, -1)

        data = (
            torch.tensor(x, dtype=torch.float),
            torch.tensor(y, dtype=torch.float),
            edge_index,
            torch.tensor([row]),
        )
        # x: [vehicle, 11]: [x_0, y_0, speed_0, yaw_0, intent, intent, intent, start_pos, start_pos, start_pos, start_pos]
        # y: [vehicle, NUM_PREDICT * 6]: [[x_1, y_1, v_1, yaw_1, acc_1, delta_1, x_2, y_2...], ...]
        with open(
            f"{preprocess_folder_path}/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}-0.pkl",
            "wb",
        ) as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # just for checks without augmentation to dont make any misstake, because augmentation gives strange data distribution
        return
        for a in range(NUM_AUGMENTATION):
            shifted_curr, mpc_output = MPC_Block(
                curr_states, future_states, acc_delta_old, noise_range=noise_range
            )  # [vehicle, 4], [vehicle, NUM_PREDICT, 6]: [x, y, v, yaw, acc, delta]
            x_argumented = x.copy()
            x_argumented[:, :2] = shifted_curr[:, :2]
            y = (mpc_output[:, :, :2] - np.expand_dims(shifted_curr[:, :2], axis=1)).transpose(0, 2, 1)  # [vehicle, 2, NUM_PREDICT]
            y = rotations @ y
            y = y.transpose(0, 2, 1)  # [vehicle, NUM_PREDICT, 2]

            if ALLIGN_INITIAL_DIRECTION_TO_X:
                mpc_output[:, :, 3:4] = mpc_output[:, :, 3:4] - all_features[:, row : row + 1, 3:4]
            else:
                mpc_output[:, :, 3:4] = mpc_output[:, :, 3:4] - all_features[:, row : row + 1, 3:4] + np.pi / 2

            # [vehicle, NUM_PREDICT, 6]
            y = np.concatenate((y, mpc_output[:, :, 2:]), axis=-1)
            y = y.reshape(num_cars, -1)

            # if NORMALIZE_DATA:
            #     normalize_target_data(y)
            #     normalize_input_data(x)

            data = (
                torch.tensor(x_argumented, dtype=torch.float),
                torch.tensor(y, dtype=torch.float),
                edge_index,
                torch.tensor([row]),
            )
            with open(
                f"{preprocess_folder_path}/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}-{a + 1}.pkl",
                "wb",
            ) as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
