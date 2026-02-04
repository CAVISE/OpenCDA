# %%

import os
import pickle
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from MPC_XY_Frame import linear_mpc_control_data_aug
from utils.config import DT, PRED_LEN


pred_len, dt = PRED_LEN, DT


def rotation_matrix(yaw: float) -> npt.NDArray[np.float64]:
    """
    Build a 2D rotation matrix to align the current heading to +y axis.

    Parameters
    ----------
    yaw : float
        Yaw angle in radians.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        Rotation matrix of shape (2, 2).

    References
    -------
    Make the current direction aligns to +y axis.
    https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
    """
    rotation = np.array([[np.cos(np.pi / 2 - yaw), -np.sin(np.pi / 2 - yaw)], [np.sin(np.pi / 2 - yaw), np.cos(np.pi / 2 - yaw)]])
    return rotation


class CarDataset(InMemoryDataset):
    """
    In-memory PyG dataset built from preprocessed `.pkl` graph samples.

    Parameters
    ----------
    preprocess_folder : str
        Directory containing `.pkl` files created by the preprocessing script.
    plot : bool, optional
        Whether to create debug plots (default: False).
    mlp : bool, optional
        If True, keeps only self-loop edges (for MLP baseline) (default: False).
    mpc_aug : bool, optional
        If False, keeps only augmentation index 0 samples (default: True).

    Attributes
    ----------
    preprocess_folder : str
        Input folder with `.pkl` samples.
    plot : bool
        Plot flag.
    mlp : bool
        Whether to use self-loop edges only.
    mpc_aug : bool
        Whether to include MPC-augmented samples.
    """

    # read from preprocessed data
    def __init__(self, preprocess_folder: str, plot: bool = False, mlp: bool = False, mpc_aug: bool = True):
        self.preprocess_folder = preprocess_folder
        self.plot = plot
        self.mlp = mlp
        self.mpc_aug = mpc_aug
        super().__init__(preprocess_folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self) -> None:
        """
        Convert `.pkl` samples into a single `InMemoryDataset` object.

        The input `.pkl` file is expected to contain:
        x : torch.Tensor
            Shape (V, 7) with features [x, y, v, yaw, intent(3)].
        y : torch.Tensor
            Shape (V, PRED_LEN*6) with targets [x, y, v, yaw, acc, steer] per step.
        edge_index : torch.Tensor
            Shape (2, E) with graph connectivity.
        t : torch.Tensor
            Shape (1,) or scalar-like row index.
        """
        preprocess_files = os.listdir(self.preprocess_folder)
        preprocess_files.sort()
        graphs = list()

        if self.plot:
            fig, _ = plt.subplots()  # plot the mcp augmented data
            # fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f"images/{self.preprocess_folder}", exist_ok=True)

        for file in tqdm(preprocess_files):
            if os.path.splitext(file)[1] != ".pkl":
                continue
            if not self.mpc_aug:
                if os.path.splitext(file)[0].split("-")[1] != "0":
                    continue
            data = pickle.load(open(os.path.join(self.preprocess_folder, file), "rb"))
            # x: [v, 7], [x, y, v, yaw, intention(3-bit)],
            # y: [v, pred_len*6], [x, y, v, yaw, acc, steering],
            # edge_indices: [2, edge],
            # t: [], row index in a csv file
            n_v = data[0].shape[0]
            weights = torch.ones(n_v)
            turn_index = (data[0][:, 4] + data[0][:, 6]).bool()  # left- and right-turn cases with higher weights
            center_index1 = (data[0][:, 0].abs() < 30) * (data[0][:, 1].abs() < 30)  # vehicles in the central area with higher weights
            center_index2 = (data[0][:, 0].abs() < 40) * (data[0][:, 1].abs() < 40)
            weights[turn_index] *= 1.5
            weights[center_index1] *= 4
            weights[center_index2] *= 4

            if self.mlp:
                self_loop_index = data[2][0, :] == data[2][1, :]
                graph = Data(x=data[0], y=data[1], edge_index=data[2][:, self_loop_index], t=data[3], weights=weights)
            else:
                graph = Data(x=data[0], y=data[1], edge_index=data[2], t=data[3], weights=weights)
            # [v,7], [v, pred_len*6], [2, edge], []
            graphs.append(graph)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


def adjust_future_deltas(curr_states: npt.NDArray[np.float64], future_states: npt.NDArray[np.float64]) -> None:
    """
    Adjust future yaw angles to avoid discontinuities around +/- pi.

    Parameters
    ----------
    curr_states : numpy.typing.NDArray[numpy.float64]
        Current states of shape (V, 4): [x, y, speed, yaw].
    future_states : numpy.typing.NDArray[numpy.float64]
        Future states of shape (V, pred_len, 4): [x, y, speed, yaw].
        Modified in-place.
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

    return None


def MPC_Block(
    curr_states: npt.NDArray[np.float64], target_states: npt.NDArray[np.float64], acc_delta_old: npt.NDArray[np.float64], noise_range: float = 0.0
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Run MPC for a batch of vehicles.

    Parameters
    ----------
    curr_states : numpy.typing.NDArray[numpy.float64]
        Current states, shape (V, 4): [x, y, speed, yaw].
    target_states : numpy.typing.NDArray[numpy.float64]
        Target/future states, shape (V, pred_len, 4).
    acc_delta_old : numpy.typing.NDArray[numpy.float64]
        Previous control warm-start, shape (V, pred_len, 2): [acc, delta].
    noise_range : float, optional
        Lateral noise magnitude applied inside MPC_module (default: 0.0).

    Returns
    -------
    shifted_curr : numpy.typing.NDArray[numpy.float64]
        Possibly shifted current states, shape (V, 4).
    mpc_output : numpy.typing.NDArray[numpy.float64]
        MPC outputs, shape (V, pred_len, 6): [x, y, speed, yaw, acc, delta]
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
    curr_state_v: npt.NDArray[np.float64],
    target_states_v: npt.NDArray[np.float64],
    acc_delta_old_v: npt.NDArray[np.float64],
    noise_range: float = 0.0,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Run MPC for a single vehicle.

    Parameters
    ----------
    curr_state_v : numpy.typing.NDArray[numpy.float64]
        Current state, shape (4,): [x0, y0, speed0, yaw0].
    target_states_v : numpy.typing.NDArray[numpy.float64]
        Future/target states, shape (pred_len, 4).
    acc_delta_old_v : numpy.typing.NDArray[numpy.float64]
        Warm-start controls, shape (pred_len, 2): [acc, delta].
        NaNs are replaced by zeros in-place.
    noise_range : float, optional
        Lateral noise magnitude (default: 0.0).

    Returns
    -------
    shifted_curr : numpy.typing.NDArray[numpy.float64]
        Current state after optional lateral noise, shape (4,).
    mpc_output : numpy.typing.NDArray[numpy.float64]
        MPC output sequence, shape (pred_len, 6): [x, y, v, yaw, acc, delta].
    """

    acc_delta_old_v[np.isnan(acc_delta_old_v)] = 0.0  # [pred_len, 2]
    a_old = acc_delta_old_v[:, 0].tolist()
    delta_old = acc_delta_old_v[:, 1].tolist()

    if noise_range > 0:
        curr_state_v = curr_state_v.copy()  # avoid add noise in-place
        noise_direction = curr_state_v[3] - np.deg2rad(90)
        noise_length = np.random.uniform(low=-1, high=1) * noise_range  # TODO: uniform or Gaussian distribution?
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


def transform_sumo2carla(states: npt.NDArray[np.float64]) -> None:
    """
    In-place coordinate transform from SUMO to CARLA conventions.

    Parameters
    ----------
    states : numpy.typing.NDArray[numpy.float64]
        State array containing at least indices [1] (y) and [3] (yaw).
        Supported shapes:
        - (4,) single state [x, y, speed, yaw]
        - (N, 4) batch of states

    Notes:
    -----
        - the coordinate system in Carla is more convenient since the angle increases in the direction of rotation from +x to +y, while in sumo this is from +y to +x.
        - the coordinate system in Carla is a left-handed Cartesian coordinate system.
    """
    if states.ndim == 1:
        states[1] = -states[1]
        states[3] -= np.deg2rad(90)
    elif states.ndim == 2:
        states[:, 1] = -states[:, 1]
        states[:, 3] -= np.deg2rad(90)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    train_folder = "csv/train_pre"
    train_dataset = CarDataset(preprocess_folder=train_folder, mlp=False, mpc_aug=True)
