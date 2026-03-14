import os
import pickle
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from enum import Enum
from tqdm import tqdm
from typing import Optional, Tuple

from .data_config import MAP_LANE_NAME, MAP_PARTS_NAME, MAP_IMG_NAME, COOL_DATA_RADIUS1, COOL_DATA_RADIUS2


class GnnCarDataset(InMemoryDataset):
    """
    dataset for gnn model
    """

    def __init__(self, preprocess_folder: str, mlp: bool = False, mpc_aug: bool = True) -> None:
        """
        initialize gnn car dataset

        :param preprocess_folder: path to directory with preprocessed data
        :param mlp: flag if using mlp model (only self-loops in graph)
        :param mpc_aug: flag if using mpc augmented data
        """
        self.preprocess_folder = preprocess_folder
        self.mlp = mlp
        self.mpc_aug = mpc_aug
        super().__init__(preprocess_folder)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> list[str]:
        """
        get processed file names based on model type and augmentation

        :return: list of processed file names
        """
        pt_name = "data"
        pt_name += "_mlp" if self.mlp else "_gnn"

        if self.mpc_aug:
            pt_name += "_aug"

        return [f"{pt_name}.pt"]

    def process(self) -> None:
        """
        convert raw data into gnn-readable format by constructing graphs from connectivity matrices
        """
        preprocess_subfolders = os.listdir(self.preprocess_folder)
        graphs = list()

        for preprocess_subfolder in preprocess_subfolders:
            if not (preprocess_subfolder.split("_")[-1] == "10m"):
                continue

            preprocess_subfolder_path = os.path.join(self.preprocess_folder, preprocess_subfolder)
            preprocess_files = os.listdir(preprocess_subfolder_path)
            preprocess_files.sort()

            for file in tqdm(preprocess_files):
                if os.path.splitext(file)[1] != ".pkl":
                    continue

                if not self.mpc_aug:
                    if os.path.splitext(file)[0].split("-")[-1] != "0":
                        continue

                file_path = os.path.join(preprocess_subfolder_path, file)
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                # x: [v, 11], [x, y, v, yaw, intention(3-bit), start_position(4-bit)],
                # y: [v, pred_len*6], [x, y, v, yaw, acc, steering],
                # edge_indices: [2, edge],
                # t: [], row index in a csv file
                n_v = data[0].shape[0]
                weights = torch.ones(n_v)
                # left- and right-turn cases with higher weights
                turn_index = (data[0][:, 4] + data[0][:, 6]).bool()
                # vehicles in the central area with higher weights

                map_boundary = 100

                center_index1 = (data[0][:, 0].abs() < 0.3 * map_boundary) * (data[0][:, 1].abs() < 0.3 * map_boundary)
                center_index2 = (data[0][:, 0].abs() < 0.4 * map_boundary) * (data[0][:, 1].abs() < 0.4 * map_boundary)
                weights[turn_index] *= 1.5
                weights[center_index1] *= 4
                weights[center_index2] *= 4

                if self.mlp:
                    self_loop_index = data[2][0, :] == data[2][1, :]
                    graph = Data(
                        x=data[0],
                        y=data[1],
                        edge_index=data[2][:, self_loop_index],
                        t=data[3],
                        weights=weights,
                    )
                else:
                    graph = Data(x=data[0], y=data[1], edge_index=data[2], t=data[3], weights=weights)
                # [v,11], [v, pred_len*6], [2, edge], []
                graphs.append(graph)

            data, slices = self.collate(graphs)

        torch.save((data, slices), self.processed_paths[0])


class MAP_level_info(Enum):
    """
    Map encoding type:

    MAP_level_info.map: map level encoding \n
    MAP_level_info.lane: lane level encoding \n
    MAP_level_info.parts: parts level encoding
    """

    map = 0
    lane = 1
    parts = 2


class TransformerDataCell:
    """
    data holder class for transformer training process
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, attn_mask: torch.Tensor, map_idx: str) -> None:
        """
        initialize transformer data cell

        :param x: car input vectors
        :param y: output vectors
        :param weights: weights for each sample in dataset
        :param attn_mask: attention mask for vehicles in sample
        :param map_idx: map index corresponding to this sample
        """
        self.x = x
        self.y = y
        self.weights = weights
        self.attn_mask = attn_mask
        self.map_idx = map_idx


class TransformerCarDataset(Dataset):
    """
    Dataset for transformer
    """

    def __init__(
        self, preprocess_folder: str, reprocess: bool = False, mpc_aug: bool = True, map_info_level: MAP_level_info = MAP_level_info.lane
    ) -> None:
        """
        initialize transformer car dataset

        :param preprocess_folder: path to directory with preprocessed data samples
        :param reprocess: prepare data for dataset even if dataset file exists
        :param mpc_aug: flag if augmented data in preprocessed samples
        :param map_info_level: map info level type
        """
        self.preprocess_folder = preprocess_folder
        self.processed_dir = os.path.join(self.preprocess_folder, "transformer_processed")
        self.processed_path = os.path.join(self.processed_dir, "processed.pt")
        self.reprocess = reprocess
        self.mpc_aug = mpc_aug
        self.map_info_level = map_info_level
        super().__init__()

        self.load_maps()
        self.process()
        self.complite_data = torch.load(self.processed_path, weights_only=False)

    def get_map_info_path(self, preprocess_subfolder_path: str) -> Optional[str]:
        """
        get map info file path based on map info level

        :param preprocess_subfolder_path: preprocess subfolder path connected with specific map

        :return: path to map info file
        """
        if self.map_info_level == MAP_level_info.map:
            return os.path.join(preprocess_subfolder_path, f"{MAP_IMG_NAME}.npy")

        if self.map_info_level == MAP_level_info.lane:
            return os.path.join(preprocess_subfolder_path, f"{MAP_LANE_NAME}.npy")

        elif self.map_info_level == MAP_level_info.parts:
            return os.path.join(preprocess_subfolder_path, f"{MAP_PARTS_NAME}.npy")
        return None

    def load_maps(self) -> None:
        """
        load map information including map images, attention masks for map channels, and map boundaries
        """
        self.map_infos = {}
        self.map_boundaries = {}
        self.map_channels_masks = {}
        max_map_channels = 0

        preprocess_subfolders = os.listdir(self.preprocess_folder)
        for preprocess_subfolder in preprocess_subfolders:
            if not (preprocess_subfolder.split("_")[-1] == "10m"):
                continue

            preprocess_subfolder_path = os.path.join(self.preprocess_folder, preprocess_subfolder)
            map_info_path = self.get_map_info_path(preprocess_subfolder_path)
            map_info = np.load(map_info_path, allow_pickle=True)

            num_channels = map_info[0][0].shape[0]
            max_map_channels = num_channels if num_channels > max_map_channels else max_map_channels

        for preprocess_subfolder in preprocess_subfolders:
            if not (preprocess_subfolder.split("_")[-1] == "10m"):
                continue

            preprocess_subfolder_path = os.path.join(self.preprocess_folder, preprocess_subfolder)
            map_info_path = self.get_map_info_path(preprocess_subfolder_path)
            map_info = np.load(map_info_path, allow_pickle=True)

            cur_num_channels = map_info[0][0].shape[0]
            self.map_infos[preprocess_subfolder] = torch.tensor(map_info[0][0])
            self.map_infos[preprocess_subfolder] = nn.functional.pad(
                self.map_infos[preprocess_subfolder], (0, 0, 0, max_map_channels - cur_num_channels)
            )
            map_channel_mask = torch.ones((max_map_channels, max_map_channels), dtype=torch.bool)
            map_channel_mask[:, cur_num_channels:] = False
            map_channel_mask[cur_num_channels:, :] = False
            self.map_channels_masks[preprocess_subfolder] = map_channel_mask
            self.map_boundaries[preprocess_subfolder] = torch.tensor(map_info[0][2])

    def process(self) -> None:
        """
        process car vectors and save to dataset processed file
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        if os.path.exists(self.processed_path) and not self.reprocess:
            return

        preprocess_subfolders = os.listdir(self.preprocess_folder)

        whole_data = []
        max_vechile_count = 0

        for preprocess_subfolder in preprocess_subfolders:
            if not (preprocess_subfolder.split("_")[-1] == "10m"):
                continue

            preprocess_subfolder_path = os.path.join(self.preprocess_folder, preprocess_subfolder)
            preprocess_files = os.listdir(preprocess_subfolder_path)

            for file in preprocess_files:
                if os.path.splitext(file)[1] != ".pkl":
                    continue
                if not self.mpc_aug:
                    if os.path.splitext(file)[0].split("-")[-1] != "0":
                        continue

                file_path = os.path.join(preprocess_subfolder_path, file)
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                n_v = data[0].shape[0]
                max_vechile_count = n_v if max_vechile_count < n_v else max_vechile_count

        for preprocess_subfolder in preprocess_subfolders:
            if not (preprocess_subfolder.split("_")[-1] == "10m"):
                continue

            preprocess_subfolder_path = os.path.join(self.preprocess_folder, preprocess_subfolder)
            preprocess_files = os.listdir(preprocess_subfolder_path)
            preprocess_files.sort()
            map_boundary = self.map_boundaries[preprocess_subfolder]

            for file in tqdm(preprocess_files):
                if os.path.splitext(file)[1] != ".pkl":
                    continue
                if not self.mpc_aug:
                    if os.path.splitext(file)[0].split("-")[-1] != "0":
                        continue

                file_path = os.path.join(preprocess_subfolder_path, file)
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                # x: [vehicle, steps, 12]: [x, y, speed, yaw, dstart_yaw, dlast_yaw, start_cos, start_sin, last_cos, last_sin, acc, delta]
                # y: [v, pred_len*6], [x, y, v, yaw, acc, steering],
                # edge_indices: [2, edge],

                n_v = data[0].shape[0]
                weights = torch.ones(n_v)

                # left- and right-turn cases with higher weights
                turn_index = (data[0][:, 5] != 0).bool()
                # vehicles in the central area with higher weights
                center_index1 = (data[0][:, 0].abs() < COOL_DATA_RADIUS1 * map_boundary) * (data[0][:, 1].abs() < COOL_DATA_RADIUS1 * map_boundary)
                center_index2 = (data[0][:, 0].abs() < COOL_DATA_RADIUS2 * map_boundary) * (data[0][:, 1].abs() < COOL_DATA_RADIUS2 * map_boundary)
                weights[turn_index] *= 6
                weights[center_index1] *= 4
                weights[center_index2] *= 4

                x = nn.functional.pad(data[0], (0, 0, 0, max_vechile_count - n_v))
                y = nn.functional.pad(data[1], (0, 0, 0, max_vechile_count - n_v))
                weights = nn.functional.pad(weights, (0, max_vechile_count - n_v))
                attn_mask = torch.ones((max_vechile_count, max_vechile_count), dtype=torch.bool)
                attn_mask[:, n_v:] = False
                attn_mask[n_v:, :] = False
                data_cell = TransformerDataCell(x, y, weights=weights, attn_mask=attn_mask, map_idx=preprocess_subfolder)

                whole_data.append(data_cell)

        torch.save(whole_data, self.processed_path)

    def __len__(self):
        """
        get dataset length

        :return: number of samples in dataset
        """
        return len(self.complite_data)

    def __getitem__(self, index: int) -> Tuple[TransformerDataCell, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        get item from dataset by index

        :param index: sample index

        :return: tuple of (data_cell, map_info, map_channels_mask, map_boundaries)
        """
        data_cell = self.complite_data[index]
        return data_cell, self.map_infos[data_cell.map_idx], self.map_channels_masks[data_cell.map_idx], self.map_boundaries[data_cell.map_idx]
