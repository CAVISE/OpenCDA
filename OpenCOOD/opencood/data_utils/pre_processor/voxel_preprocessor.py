"""
Voxel preprocessor for LiDAR point cloud conversion.

This module provides manual voxelization of LiDAR point clouds for 3D object
detection in cooperative autonomous driving. For improved performance, consider
using sp_voxel_preprocessor instead.
"""

import sys

import numpy as np
from numpy.typing import NDArray
import torch

from opencood.data_utils.pre_processor.base_preprocessor import BasePreprocessor
from typing import Dict, List, Union, Any


class VoxelPreprocessor(BasePreprocessor):
    """
    Voxel preprocessor for converting LiDAR point clouds to voxel representation.
    
    This class performs voxelization of raw LiDAR data. For better performance,
    consider using sp_voxel_preprocessor instead.
    
    Parameters
    ----------
    preprocess_params : Dict[str, Any]
        Configuration dictionary for preprocessing containing:
        - cav_lidar_range : list
            LiDAR range boundaries
        - args : dict
            Voxelization parameters with keys:
            - vw : float
                Voxel width
            - vh : float
                Voxel height
            - vd : float
                Voxel depth
            - T : int
                Maximum number of points per voxel
    train : bool
        Boolean indicating training or evaluation mode.
    
    Attributes
    ----------
    lidar_range : list
        LiDAR range boundaries.
    vw : float
        Voxel width.
    vh : float
        Voxel height.
    vd : float
        Voxel depth.
    T : int
        Maximum number of points per voxel.
    """

    def __init__(self, preprocess_params: Dict[str, Any], train: bool):
        super(VoxelPreprocessor, self).__init__(preprocess_params, train)
        # TODO: add intermediate lidar range later
        self.lidar_range = self.params["cav_lidar_range"]

        self.vw = self.params["args"]["vw"]
        self.vh = self.params["args"]["vh"]
        self.vd = self.params["args"]["vd"]
        self.T = self.params["args"]["T"]

    def preprocess(self, pcd_np: NDArray[np.float64]) -> Dict[str, NDArray]:
        """
        Preprocess the lidar points by voxelization.
        
        Parameters
        ----------
        pcd_np : NDArray[np.float64]
            The raw lidar point cloud with shape (N, 4) where N is number of points.
        
        Returns
        -------
        Dict[str, NDArray]
            Dictionary containing:
            - voxel_features : NDArray
                Voxelized features with shape (num_voxels, T, 7)
            - voxel_coords : NDArray
                Voxel coordinates with shape (num_voxels, 3)
        """
        data_dict = {}

        # calculate the voxel coordinates
        voxel_coords = (
            pcd_np[:, :3] - np.floor(np.array([self.lidar_range[0], self.lidar_range[1], self.lidar_range[2]])) / (self.vw, self.vh, self.vd)
        ).astype(np.int32)

        # convert to (D, H, W) as the paper
        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)

        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.T, 7), dtype=np.float32)
            pts = pcd_np[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[: self.T, :]
                voxel_counts[i] = self.T

            # augment the points
            voxel[: pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
            voxel_features.append(voxel)

        data_dict["voxel_features"] = np.array(voxel_features)
        data_dict["voxel_coords"] = voxel_coords

        return data_dict

    def collate_batch(self, batch: Union[List[Dict[str, NDArray]], Dict[str, List[NDArray]]]) -> Dict[str, torch.Tensor]:
        """
        Customized PyTorch data loader collate function.
        
        Parameters
        ----------
        batch : Union[List[Dict[str, NDArray]], Dict[str, List[NDArray]]]
            Either a list of dictionaries (each representing a frame) or
            a dictionary with lists as values.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing batched tensors:
            - voxel_features : torch.Tensor
                Concatenated voxel features
            - voxel_coords : torch.Tensor
                Concatenated voxel coordinates with batch indices
        
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
    def collate_batch_list(batch: List[Dict[str, NDArray]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch when input is a list of dictionaries.
        
        Parameters
        ----------
        batch : List[Dict[str, NDArray]]
            List of dictionaries, where each dictionary represents a single frame
            containing 'voxel_features' and 'voxel_coords'.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - voxel_features : torch.Tensor
                Concatenated voxel features from all frames
            - voxel_coords : torch.Tensor
                Concatenated voxel coordinates with batch indices prepended
        """
        voxel_features = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]["voxel_features"])
            coords = batch[i]["voxel_coords"]
            voxel_coords.append(np.pad(coords, ((0, 0), (1, 0)), mode="constant", constant_values=i))

        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {"voxel_features": voxel_features, "voxel_coords": voxel_coords}

    @staticmethod
    def collate_batch_dict(batch: Dict[str, List[NDArray]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch when input is a dictionary with lists as values.
        
        Parameters
        ----------
        batch : Dict[str, List[NDArray]]
            Dictionary with keys 'voxel_features' and 'voxel_coords',
            where values are lists of numpy arrays.
            Example: {'voxel_features': [feature1, feature2, ..., feature_n]}
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - voxel_features : torch.Tensor
                Concatenated voxel features from all frames
            - voxel_coords : torch.Tensor
                Concatenated voxel coordinates with batch indices prepended
        """
        voxel_features = torch.from_numpy(np.concatenate(batch["voxel_features"]))
        coords = batch["voxel_coords"]
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(np.pad(coords[i], ((0, 0), (1, 0)), mode="constant", constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {"voxel_features": voxel_features, "voxel_coords": voxel_coords}