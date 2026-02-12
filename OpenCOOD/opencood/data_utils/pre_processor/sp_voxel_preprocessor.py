"""
Transform points to voxels using sparse conv library.

This module provides a sparse voxel preprocessor for converting LiDAR point clouds
to voxel representations using the spconv library (supports both v1.x and v2.x).
"""

import sys

import numpy as np
import torch
from cumm import tensorview as tv
from opencood.data_utils.pre_processor.base_preprocessor import BasePreprocessor
from typing import Dict, List, Union, Any


class SpVoxelPreprocessor(BasePreprocessor):
    """
    Sparse voxel preprocessor for converting LiDAR point clouds to voxel representation.

    This preprocessor uses the sparse convolution library (spconv) to efficiently
    generate voxel representations from point clouds. It supports both spconv v1.x
    and v2.x.

    Parameters
    ----------
    preprocess_params : Dict[str, Any]
        Configuration dictionary containing:
        - 'cav_lidar_range': LiDAR detection range [xmin, ymin, zmin, xmax, ymax, zmax]
        - 'args': Dictionary with voxelization parameters
            - 'voxel_size': [dx, dy, dz] dimensions of each voxel
            - 'max_points_per_voxel': Maximum points allowed per voxel
            - 'max_voxel_train': Maximum voxels for training
            - 'max_voxel_test': Maximum voxels for testing/validation
    train : bool
        Whether the preprocessor is used for training (True) or testing (False).
        Determines the maximum number of voxels to generate.

    Attributes
    ----------
    spconv : int
        Version of spconv being used (1 for v1.x, 2 for v2.x).
    lidar_range : List[float]
        LiDAR detection range.
    voxel_size : List[float]
        Size of each voxel in [x, y, z] dimensions.
    max_points_per_voxel : int
        Maximum number of points per voxel.
    max_voxels : int
        Maximum number of voxels to generate.
    grid_size : np.ndarray
        Grid dimensions in voxel coordinates.
    voxel_generator : VoxelGenerator
        Spconv voxel generator instance.
    """

    def __init__(self, preprocess_params: Dict[str, Any], train: bool) -> None:
        super(SpVoxelPreprocessor, self).__init__(preprocess_params, train)
        self.spconv = 1
        try:
            # spconv v1.x
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except ImportError:
            # spconv v2.x
            from spconv.utils import Point2VoxelCPU3d as VoxelGenerator

            self.spconv = 2
        self.lidar_range = self.params["cav_lidar_range"]
        self.voxel_size = self.params["args"]["voxel_size"]
        self.max_points_per_voxel = self.params["args"]["max_points_per_voxel"]

        if train:
            self.max_voxels = self.params["args"]["max_voxel_train"]
        else:
            self.max_voxels = self.params["args"]["max_voxel_test"]

        grid_size = (np.array(self.lidar_range[3:6]) - np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # use sparse conv library to generate voxel
        if self.spconv == 1:
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size, point_cloud_range=self.lidar_range, max_num_points=self.max_points_per_voxel, max_voxels=self.max_voxels
            )
        else:
            self.voxel_generator = VoxelGenerator(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                max_num_points_per_voxel=self.max_points_per_voxel,
                num_point_features=4,
                max_num_voxels=self.max_voxels,
            )

    def preprocess(self, pcd_np: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert point cloud to sparse voxel representation.

        Parameters
        ----------
        pcd_np : np.ndarray
            Input point cloud with shape (N, 4) where columns are (x, y, z, intensity).

        Returns
        -------
        dict
            Dictionary containing:
            - voxel_features : np.ndarray
                Voxel features with shape (M, max_points_per_voxel, 4)
            - voxel_coords : np.ndarray
                Voxel coordinates with shape (M, 3)
            - voxel_num_points : np.ndarray
                Number of points in each voxel with shape (M,)
        """
        data_dict = {}
        if self.spconv == 1:
            voxel_output = self.voxel_generator.generate(pcd_np)
        else:
            pcd_tv = tv.from_numpy(pcd_np)
            voxel_output = self.voxel_generator.point_to_voxel(pcd_tv)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = voxel_output["voxels"], voxel_output["coordinates"], voxel_output["num_points_per_voxel"]
        else:
            voxels, coordinates, num_points = voxel_output

        if self.spconv == 2:
            voxels = voxels.numpy()
            coordinates = coordinates.numpy()
            num_points = num_points.numpy()

        data_dict["voxel_features"] = voxels
        data_dict["voxel_coords"] = coordinates
        data_dict["voxel_num_points"] = num_points

        return data_dict

    def collate_batch(self, batch: Union[List[Dict[str, np.ndarray]], Dict[str, List[np.ndarray]]]) -> Dict[str, torch.Tensor]:
        """
        Customized PyTorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            Either a list of dictionaries (each representing a frame) or
            a dictionary with lists as values.

        Returns
        -------
        dict
            Dictionary containing batched tensors:
            - voxel_features : torch.Tensor
                Concatenated voxel features
            - voxel_coords : torch.Tensor
                Concatenated voxel coordinates with batch indices
            - voxel_num_points : torch.Tensor
                Concatenated number of points per voxel

        Raises
        ------
        SystemExit
            If batch is neither a list nor a dictionary.
        """
        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit("Batch has to be a list or a dictionary")

    @staticmethod
    def collate_batch_list(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch when input is a list of dictionaries.

        Parameters
        ----------
        batch : list of dict
            List of dictionaries, where each dictionary represents a single frame
            containing 'voxel_features', 'voxel_coords', and 'voxel_num_points'.

        Returns
        -------
        dict
            Dictionary containing:
            - voxel_features : torch.Tensor
                Concatenated voxel features from all frames
            - voxel_coords : torch.Tensor
                Concatenated voxel coordinates with batch indices prepended
            - voxel_num_points : torch.Tensor
                Concatenated number of points per voxel from all frames
        """
        voxel_features_list: List[np.ndarray] = []
        voxel_num_points_list: List[np.ndarray] = []
        voxel_coords_list: List[np.ndarray] = []

        for i in range(len(batch)):
            voxel_features_list.append(batch[i]["voxel_features"])
            voxel_num_points_list.append(batch[i]["voxel_num_points"])
            coords = batch[i]["voxel_coords"]
            voxel_coords_list.append(np.pad(coords, ((0, 0), (1, 0)), mode="constant", constant_values=i))

        voxel_num_points_tensor = torch.from_numpy(np.concatenate(voxel_num_points_list))
        voxel_features_tensor = torch.from_numpy(np.concatenate(voxel_features_list))
        voxel_coords_tensor = torch.from_numpy(np.concatenate(voxel_coords_list))

        return {"voxel_features": voxel_features_tensor, "voxel_coords": voxel_coords_tensor, "voxel_num_points": voxel_num_points_tensor}

    @staticmethod
    def collate_batch_dict(batch: Dict[str, List[np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch when input is a dictionary with lists as values.

        Parameters
        ----------
        batch : dict
            Dictionary with keys 'voxel_features', 'voxel_coords', and 'voxel_num_points',
            where values are lists of numpy arrays.
            Example: {'voxel_features': [feature1, feature2, ..., feature_n]}

        Returns
        -------
        dict
            Dictionary containing:
            - voxel_features : torch.Tensor
                Concatenated voxel features from all frames
            - voxel_coords : torch.Tensor
                Concatenated voxel coordinates with batch indices prepended
            - voxel_num_points : torch.Tensor
                Concatenated number of points per voxel from all frames
        """
        voxel_features_tensor = torch.from_numpy(np.concatenate(batch["voxel_features"]))
        voxel_num_points_tensor = torch.from_numpy(np.concatenate(batch["voxel_num_points"]))
        coords = batch["voxel_coords"]
        voxel_coords_list: List[np.ndarray] = []

        for i in range(len(coords)):
            voxel_coords_list.append(np.pad(coords[i], ((0, 0), (1, 0)), mode="constant", constant_values=i))
        voxel_coords_tensor = torch.from_numpy(np.concatenate(voxel_coords_list))

        return {"voxel_features": voxel_features_tensor, "voxel_coords": voxel_coords_tensor, "voxel_num_points": voxel_num_points_tensor}
