# %%

import os
import pickle

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from CoDriving.data_scripts.data_config.data_config import MAP_BOUNDARY


class CarDataset(InMemoryDataset):
    # read from preprocessed data
    def __init__(self, preprocess_folder, plot=False, mlp=False, mpc_aug=True):
        self.preprocess_folder = preprocess_folder
        self.plot = plot
        self.mlp = mlp
        self.mpc_aug = mpc_aug
        super().__init__(preprocess_folder)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        pt_name = "data"
        pt_name += "_mlp" if self.mlp else "_gnn"
        if self.mpc_aug:
            pt_name += "_aug"
        return [f"{pt_name}.pt"]

    def process(self):
        """
        Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.
        """
        preprocess_files = os.listdir(self.preprocess_folder)
        preprocess_files.sort()
        graphs = list()

        if self.plot:
            fig, ax = plt.subplots()  # plot the mcp augmented data
            fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f"images/{self.preprocess_folder}", exist_ok=True)

        for file in tqdm(preprocess_files):
            if os.path.splitext(file)[1] != ".pkl":
                continue
            if not self.mpc_aug:
                if os.path.splitext(file)[0].split("-")[-1] != "0":
                    continue
            data = pickle.load(open(os.path.join(self.preprocess_folder, file), "rb"))
            # x: [v, 7], [x, y, v, yaw, intention(3-bit)],
            # y: [v, pred_len*6], [x, y, v, yaw, acc, steering],
            # edge_indices: [2, edge],
            # t: [], row index in a csv file
            n_v = data[0].shape[0]
            weights = torch.ones(n_v)
            # left- and right-turn cases with higher weights
            turn_index = (data[0][:, 4] + data[0][:, 6]).bool()
            # vehicles in the central area with higher weights
            center_index1 = (data[0][:, 0].abs() < 0.3 * MAP_BOUNDARY) * (data[0][:, 1].abs() < 0.3 * MAP_BOUNDARY)
            center_index2 = (data[0][:, 0].abs() < 0.4 * MAP_BOUNDARY) * (data[0][:, 1].abs() < 0.4 * MAP_BOUNDARY)
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
            # [v,7], [v, pred_len*6], [2, edge], []
            graphs.append(graph)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    train_folder = "csv/train_pre"
    train_dataset = CarDataset(preprocess_folder=train_folder, mlp=False, mpc_aug=True)
