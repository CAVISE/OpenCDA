"""
Convert LiDAR point clouds to Bird's Eye View (BEV) representations.

This module provides a BEV preprocessor that projects 3D LiDAR point clouds
onto a 2D bird's eye view grid with occupancy and intensity information.
"""

import numpy as np
import torch
from opencood.data_utils.pre_processor.base_preprocessor import BasePreprocessor
from typing import Dict, List, Union, Any

class BevPreprocessor(BasePreprocessor):
    """
    Bird's Eye View (BEV) preprocessor for LiDAR point clouds.
    
    This preprocessor converts 3D LiDAR point clouds into 2D BEV representations
    by projecting points onto a bird's eye view grid with occupancy and intensity
    information across multiple height bins.
    
    Parameters
    ----------
    preprocess_params : Dict[str, Any]
        Configuration dictionary containing:
        - 'cav_lidar_range': LiDAR detection range [xmin, ymin, zmin, xmax, ymax, zmax]
        - 'geometry_param': Dictionary with BEV grid parameters
            - 'input_shape': Tuple[int, int, int] - (length, width, height_bins + 1)
            - 'L1', 'W1', 'H1': float - Origin coordinates of BEV grid
            - 'res': float - Resolution (voxel size) in meters
    train : bool
        Whether the preprocessor is used for training (True) or testing (False).
    
    Attributes
    ----------
    lidar_range : List[float]
        LiDAR detection range.
    geometry_param : Dict[str, Any]
        BEV grid geometry parameters.
    """
    
    def __init__(self, preprocess_params: Dict[str, Any], train: bool):
        super(BevPreprocessor, self).__init__(preprocess_params, train)
        self.lidar_range = self.params["cav_lidar_range"]
        self.geometry_param = preprocess_params["geometry_param"]

    def preprocess(self, pcd_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Preprocess the LiDAR point cloud to BEV representation.
        
        Converts raw 3D point cloud with intensity values into a 2D BEV grid
        where each cell contains occupancy information across height bins and
        averaged intensity values.

        Parameters
        ----------
        pcd_raw : NDArray
            Raw LiDAR point cloud with shape (N, 4).
            Each point is [x, y, z, intensity] where:
                - x, y, z : float
                    3D coordinates in meters.
                - intensity : float
                    Reflectivity/intensity value (0-1 range).

        Returns
        -------
        data_dict : dict of {str: NDArray}
            Dictionary containing:
                - bev_input : NDArray
                    BEV representation with shape (C, H, W) where:
                        - C : int
                            Number of channels (height bins + intensity).
                        - H, W : int
                            Height and width of BEV grid in voxels.
                    First C-1 channels are binary occupancy for each height bin,
                    last channel contains averaged intensity values.
        """
        bev = np.zeros(self.geometry_param["input_shape"], dtype=np.float32)
        intensity_map_count = np.zeros((bev.shape[0], bev.shape[1]), dtype=np.int64)
        bev_origin = np.array([self.geometry_param["L1"], self.geometry_param["W1"], self.geometry_param["H1"]]).reshape(1, -1)

        indices = ((pcd_raw[:, :3] - bev_origin) / self.geometry_param["res"]).astype(int)

        for i in range(indices.shape[0]):
            bev[indices[i, 0], indices[i, 1], indices[i, 2]] = 1
            bev[indices[i, 0], indices[i, 1], -1] += pcd_raw[i, 3]
            intensity_map_count[indices[i, 0], indices[i, 1]] += 1
        divide_mask = intensity_map_count != 0
        bev[divide_mask, -1] = np.divide(bev[divide_mask, -1], intensity_map_count[divide_mask])

        data_dict = {"bev_input": np.transpose(bev, (2, 0, 1))}
        return data_dict

    @staticmethod
    def collate_batch_list(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Customized pytorch data loader collate function..

        Parameters
        ----------
        batch : list of dict
            List of dictionaries, each representing a single frame.
            Each dictionary contains:
                - bev_input : NDArray
                    BEV representation with shape (C, H, W).

        Returns
        -------
        processed_batch : dict of {str: torch.Tensor}
            Dictionary containing:
                - bev_input : torch.Tensor
                    Batched BEV tensor with shape (B, C, H, W) where:
                        - B : int
                            Batch size (number of frames).
                        - C, H, W : int
                            Channels, height, width of each BEV.
        """
        bev_input_list = [x["bev_input"][np.newaxis, ...] for x in batch]
        processed_batch = {"bev_input": torch.from_numpy(np.concatenate(bev_input_list, axis=0))}
        return processed_batch

    @staticmethod
    def collate_batch_dict(batch: Dict[str, List[np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : dict of {str: list of NDArray}
            Dictionary where keys are data types and values are lists.
            Expected structure:
                - bev_input : list of NDArray
                    List of BEV representations, each with shape (C, H, W).
                    Each element represents a different CAV.

        Returns
        -------
        processed_batch : dict of {str: torch.Tensor}
            Dictionary containing:
                - bev_input : torch.Tensor
                    Batched BEV tensor with shape (N, C, H, W) where:
                        - N : int
                            Number of CAVs.
                        - C, H, W : int
                            Channels, height, width of each BEV.
        """
        bev_input_list = [x[np.newaxis, ...] for x in batch["bev_input"]]
        processed_batch = {"bev_input": torch.from_numpy(np.concatenate(bev_input_list, axis=0))}
        return processed_batch

    def collate_batch(self, batch: Union[List[Dict[str, np.ndarray]], Dict[str, List[np.ndarray]]]) -> Dict[str, torch.Tensor]:
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list of dict or dict of lists
            Batched data in one of two formats:
            
            Format 1 (list of dict): Single-frame batch
                List of dictionaries, each containing:
                    - bev_input : NDArray with shape (C, H, W)
            
            Format 2 (dict of lists): Multi-CAV batch
                Dictionary containing:
                    - bev_input : list of NDArray, each with shape (C, H, W)

        Returns
        -------
        processed_batch : dict of {str: torch.Tensor}
            Dictionary containing:
                - bev_input : torch.Tensor
                    Batched BEV tensor with shape (B, C, H, W).
        """
        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            raise NotImplementedError
