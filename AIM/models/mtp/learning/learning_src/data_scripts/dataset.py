import os
import pickle
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset, HeteroData
from tqdm import tqdm
from typing import Optional, Tuple

from .data_config import config
from AIM.models.mtp.learning.learning_src.data_scripts.preprocess_utils import (
    rotation_matrix_with_allign_to_X,
    rotation_matrix_with_allign_to_Y,
    denormalize_yaw,
)


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


class TransformerDataCell:
    """
    data holder class for transformer training process
    """

    def __init__(
        self, x: torch.Tensor, x_global: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, attn_mask: torch.Tensor, map_idx: str
    ) -> None:
        """
        initialize transformer data cell

        :param x: car input vectors
        :param y: output vectors
        :param weights: weights for each sample in dataset
        :param attn_mask: attention mask for vehicles in sample
        :param map_idx: map index corresponding to this sample
        """
        self.x = x
        self.x_global = x_global
        self.y = y
        self.weights = weights
        self.attn_mask = attn_mask
        self.map_idx = map_idx


class TransformerCarDataset(Dataset):
    """
    Dataset for transformer
    """

    def __init__(self, preprocess_folder: str, reprocess: bool = False, mpc_aug: bool = True) -> None:
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
        return os.path.join(preprocess_subfolder_path, f"{config.image_map.map_lane_name}_object.npy")

    @classmethod
    def _create_graph_connections(cls, map_info: torch.Tensor) -> torch.Tensor:
        # map_info.shape() (lanes, n_dots, feat)

        n_lanes, n_dots, n_feat = map_info.shape
        map_info_flatten = map_info.reshape(n_lanes * n_dots, n_feat)

        # dots on the same lane mask
        lane_ids = torch.arange(n_lanes).repeat_interleave(n_dots)
        same_lane_mask = lane_ids[:, None] == lane_ids[None, :]

        # dots dotproducts mask (dots have the same orientation)
        tangents = map_info_flatten[..., 2:4]
        tangents_norms = tangents.norm(dim=-1, keepdim=True)
        tangents_normed = tangents / tangents_norms
        map_info_flatten_dotpoducts = tangents_normed @ tangents_normed.T
        map_info_flatten_dotpoducts_mask = map_info_flatten_dotpoducts >= config.object_map.cos_dotproduct_threshold

        # convert each dot into each dot local coordinate system
        coords = map_info[..., :2]
        delta = coords[:, 1:, :] - coords[:, :-1, :]
        angles = torch.atan2(delta[..., 1], delta[..., 0])
        angles = torch.cat([angles, angles[:, -1:].clone()], dim=1)
        angles = angles.reshape(n_lanes * n_dots)

        dmap_info_flatten_coords = map_info_flatten.unsqueeze(0)[..., :2] - map_info_flatten.unsqueeze(1)[..., :2]
        if config.data_processing.align_initial_direction_to_x:
            rotations = rotation_matrix_with_allign_to_X(angles).unsqueeze(1)
        else:
            rotations = rotation_matrix_with_allign_to_Y(angles).unsqueeze(1)
        dmap_info_flatten_coords = torch.matmul(rotations, dmap_info_flatten_coords.unsqueeze(-1)).squeeze(-1)

        # left, right, neighbour dots mask
        neighbour_low_lateral = config.object_map.neighbour_low_lateral_factor * 2 / config.object_map.n_lane_samples
        left_dot_mask = dmap_info_flatten_coords[..., 1] < -neighbour_low_lateral
        right_dot_mask = dmap_info_flatten_coords[..., 1] > neighbour_low_lateral

        neighbour_radius_high = config.object_map.neighbour_factor_high * 2 / config.object_map.n_lane_samples
        local_coord_norms = (dmap_info_flatten_coords[..., 0] ** 2 + dmap_info_flatten_coords[..., 1] ** 2) ** 0.5
        neighbour_mask = local_coord_norms <= neighbour_radius_high

        # successor, predicessor graph connections
        successor_mask = torch.diag(torch.ones(n_lanes * n_dots - 1, dtype=torch.bool), diagonal=1)
        predicessor_mask = torch.diag(torch.ones(n_lanes * n_dots - 1, dtype=torch.bool), diagonal=-1)

        successor_mask = successor_mask & same_lane_mask
        predicessor_mask = predicessor_mask & same_lane_mask

        successor_src, successor_dst = successor_mask.nonzero(as_tuple=True)
        predicessor_src, predicessor_dst = predicessor_mask.nonzero(as_tuple=True)

        successor_edges = torch.stack([successor_src, successor_dst], dim=0)
        predicessor_edges = torch.stack([predicessor_src, predicessor_dst], dim=0)

        # left, right graph connections
        INF = 1e6
        left_mask = (~same_lane_mask) & map_info_flatten_dotpoducts_mask & left_dot_mask & neighbour_mask
        right_mask = (~same_lane_mask) & map_info_flatten_dotpoducts_mask & right_dot_mask & neighbour_mask

        left_dist = local_coord_norms.masked_fill(~left_mask, INF)
        right_dist = local_coord_norms.masked_fill(~right_mask, INF)

        left_min_dist, left_idx = left_dist.min(dim=1)
        right_min_dist, right_idx = right_dist.min(dim=1)
        left_valid = left_min_dist < INF
        right_valid = right_min_dist < INF

        src = torch.arange(n_lanes * n_dots, device=map_info.device)
        left_src = src[left_valid]
        left_dst = left_idx[left_valid]

        right_src = src[right_valid]
        right_dst = right_idx[right_valid]

        left_edges = torch.stack([left_src, left_dst], dim=0)
        right_edges = torch.stack([right_src, right_dst], dim=0)

        return successor_edges, predicessor_edges, left_edges, right_edges, map_info_flatten

    @classmethod
    def _create_map_graph(cls, map_info, max_map_dots):
        cur_num_dots = map_info.shape[0] * map_info.shape[1]
        successor_edges, predicessor_edges, left_edges, right_edges, map_info_flatten = cls._create_graph_connections(torch.tensor(map_info))

        map_heteroData = HeteroData()
        map_heteroData["dot"].x = nn.functional.pad(map_info_flatten, (0, 0, 0, (max_map_dots - cur_num_dots))).float()
        num_nodes = map_heteroData["dot"].num_nodes
        successor_powers = cls._all_edge_index_powers(successor_edges, num_nodes, k=config.model.k_dot_steps)

        for i, edge_index in enumerate(successor_powers):
            map_heteroData["dot", f"successor_{2**i}", "dot"].edge_index = edge_index

        predicessor_powers = cls._all_edge_index_powers(predicessor_edges, num_nodes, k=config.model.k_dot_steps)
        for i, edge_index in enumerate(predicessor_powers):
            map_heteroData["dot", f"predicessor_{2**i}", "dot"].edge_index = edge_index

        map_heteroData["dot", "left", "dot"].edge_index = left_edges
        map_heteroData["dot", "right", "dot"].edge_index = right_edges
        return map_heteroData

    @classmethod
    def _all_edge_index_powers(cls, edge_index: torch.Tensor, num_nodes: int, k: int):
        matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device, dtype=torch.float32)
        matrix[edge_index[0], edge_index[1]] = 1.0
        edge_indices_list = [edge_index]

        matrix_pow = matrix
        for i in range(1, k + 1):
            matrix_pow = matrix_pow @ matrix_pow
            edge_indices_list.append(matrix_pow.to_sparse_coo().indices())

        return edge_indices_list

    def _get_map_dots_yaws(self, map_info: torch.Tensor):
        # map_info.shape (n_lane, n_dots, dot_vec)
        map_info_coords = torch.tensor(map_info[..., [0, 1]])
        dcoords = map_info_coords.unsqueeze(2) - map_info_coords.unsqueeze(1)

        norms = (dcoords[..., 0] ** 2 + dcoords[..., 1] ** 2) ** 0.5 - config.vehicle.length
        norms_mask = torch.tril(torch.ones((map_info_coords.shape[0], norms.shape[1], norms.shape[1]), dtype=torch.bool))
        INF = 1e6
        norms[~norms_mask] = INF
        start_args, start_idxs = torch.min(torch.abs(norms), dim=-1)

        end_idxs = start_idxs + 1
        zero_mask = torch.gather(norms, dim=2, index=start_idxs.unsqueeze(-1)).squeeze(-1) < 0
        end_idxs[zero_mask] = start_idxs[zero_mask] - 1
        start_yaws_mask = end_idxs < 0
        end_idxs[start_yaws_mask] = 1

        start_coords = torch.gather(map_info_coords, dim=1, index=start_idxs.unsqueeze(-1).expand(-1, -1, 2))
        end_coords = torch.gather(map_info_coords, dim=1, index=end_idxs.unsqueeze(-1).expand(-1, -1, 2))

        k = end_coords - start_coords
        kx, ky = k[..., 0:1], k[..., 1:2]
        b = start_coords - map_info_coords
        bx, by = b[..., 0:1], b[..., 1:2]

        a1 = kx**2 + ky**2
        b1 = 2 * (kx * bx + ky * by)
        c1 = bx**2 + by**2 - config.vehicle.length**2

        p = b1**2 / (4 * a1**2) - c1 / a1
        m = b1 / (2 * a1)

        t1 = p**0.5 - m
        t2 = -(p**0.5) - m

        t1_mask = (0 <= t1) & (t1 <= 1)
        t = t1.clone()
        t[~t1_mask] = t2[~t1_mask]

        yaw_vector = map_info_coords - ((end_coords - start_coords) * t + start_coords)
        start_yaws_mask_lane_inds = torch.where(start_yaws_mask)[0]
        yaw_vector[start_yaws_mask] = (map_info_coords[start_yaws_mask_lane_inds][:, 1:2, :] - map_info_coords[start_yaws_mask_lane_inds])[
            start_yaws_mask
        ]
        yaw_vector_norms = (yaw_vector[..., 0] ** 2 + yaw_vector[..., 1] ** 2) ** 0.5
        yaw_vector = yaw_vector / yaw_vector_norms.unsqueeze(-1)  # in carla coordinate system

        return yaw_vector

    def load_maps(self):
        """
        load map information including map images, attention masks for map channels, and map boundaries
        """
        self.map_boundaries = {}
        self.map_masks = {}
        self.map_heteroDatas = {}
        self.map_lane_repr = {}
        self.map_lane_repr_masks = {}
        max_map_dots = 0

        preprocess_subfolders = os.listdir(self.preprocess_folder)
        for preprocess_subfolder in preprocess_subfolders:
            if not (preprocess_subfolder.split("_")[-1] == "10m"):
                continue

            preprocess_subfolder_path = os.path.join(self.preprocess_folder, preprocess_subfolder)
            map_info_path = self.get_map_info_path(preprocess_subfolder_path)
            map_info = np.load(map_info_path, allow_pickle=True)

            num_dots = map_info[0][0].shape[0] * map_info[0][0].shape[1]
            max_map_dots = num_dots if num_dots > max_map_dots else max_map_dots

        for preprocess_subfolder in preprocess_subfolders:
            if not (preprocess_subfolder.split("_")[-1] == "10m"):
                continue

            preprocess_subfolder_path = os.path.join(self.preprocess_folder, preprocess_subfolder)
            map_info_path = self.get_map_info_path(preprocess_subfolder_path)
            map_info = np.load(map_info_path, allow_pickle=True)

            cur_num_dots = map_info[0][0].shape[0] * map_info[0][0].shape[1]
            self.map_boundaries[preprocess_subfolder] = torch.tensor(map_info[0][1])
            self.map_heteroDatas[preprocess_subfolder] = self._create_map_graph(map_info[0][0], max_map_dots)

            map_mask = torch.ones((max_map_dots), dtype=torch.bool)
            map_mask[cur_num_dots:] = False
            self.map_masks[preprocess_subfolder] = map_mask

            lanes_yaw_vectors = self._get_map_dots_yaws(map_info[0][0])
            self.map_lane_repr[preprocess_subfolder] = nn.functional.pad(
                torch.cat([torch.tensor(map_info[0][0]), lanes_yaw_vectors], dim=-1),
                (0, 0, 0, 0, 0, (max_map_dots - cur_num_dots) // map_info[0][0].shape[1]),
            )
            self.map_lane_repr_masks[preprocess_subfolder] = torch.ones(
                (max_map_dots // map_info[0][0].shape[1], map_info[0][0].shape[1]), dtype=torch.bool
            )
            self.map_lane_repr_masks[preprocess_subfolder][map_info[0][0].shape[0] :, :] = False

            print()

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
            # map_boundary = self.map_boundaries[preprocess_subfolder]

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
                eps = 1e-3
                turn_index = (torch.abs(data[0][:, 9] - data[0][:, 7]) > eps).bool() & (torch.abs(data[0][:, 8] - data[0][:, 6]) > eps).bool()
                # vehicles in the central area with higher weights
                center_index1 = (data[0][:, 0].abs() < config.vehicle.cool_data_radius1) * (data[0][:, 1].abs() < config.vehicle.cool_data_radius1)
                center_index2 = (data[0][:, 0].abs() < config.vehicle.cool_data_radius2) * (data[0][:, 1].abs() < config.vehicle.cool_data_radius2)
                weights[turn_index] *= 6
                weights[center_index1] *= 4
                weights[center_index2] *= 4

                x_data_yaw = data[0].unsqueeze(0)[:, :, 3:4] - data[0].unsqueeze(1)[:, :, 3:4]
                x_data_coords = data[0].unsqueeze(0)[:, :, :2] - data[0].unsqueeze(1)[:, :, :2]
                yaw_for_rotation = data[0][:, 3:4].clone()
                if config.data_processing.normalize_data:
                    denormalize_yaw(yaw_for_rotation)

                if config.data_processing.align_initial_direction_to_x:
                    rotation_matrixes = rotation_matrix_with_allign_to_X(yaw_for_rotation)
                else:
                    rotation_matrixes = rotation_matrix_with_allign_to_Y(yaw_for_rotation)

                x_data_coords = torch.matmul(rotation_matrixes, x_data_coords.unsqueeze(-1)).squeeze(-1)

                x_data_yaw_denormed = x_data_yaw.clone()
                if config.data_processing.normalize_data:
                    denormalize_yaw(x_data_yaw_denormed)

                x_data_speed = data[0].unsqueeze(0)[:, :, 2:3].repeat(data[0].shape[0], 1, 1)
                x_data_skip = data[0].unsqueeze(0)[:, :, 4:-2].repeat(data[0].shape[0], 1, 1)
                x_data = torch.cat(
                    [x_data_coords, x_data_speed, x_data_yaw, x_data_skip, torch.cos(x_data_yaw_denormed), torch.sin(x_data_yaw_denormed)], dim=-1
                )

                x = nn.functional.pad(x_data, (0, 0, 0, max_vechile_count - n_v, 0, max_vechile_count - n_v))
                x_global = nn.functional.pad(data[0], (0, 0, 0, max_vechile_count - n_v))
                y = nn.functional.pad(data[1], (0, 0, 0, max_vechile_count - n_v))
                weights = nn.functional.pad(weights, (0, max_vechile_count - n_v))
                attn_mask = torch.ones((max_vechile_count, max_vechile_count), dtype=torch.bool)
                attn_mask[:, n_v:] = False
                attn_mask[n_v:, :] = False
                data_cell = TransformerDataCell(x=x, x_global=x_global, y=y, weights=weights, attn_mask=attn_mask, map_idx=preprocess_subfolder)

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
        return (
            data_cell,
            self.map_heteroDatas[data_cell.map_idx],
            self.map_masks[data_cell.map_idx],
            self.map_boundaries[data_cell.map_idx],
            self.map_lane_repr[data_cell.map_idx],
            self.map_lane_repr_masks[data_cell.map_idx],
        )
