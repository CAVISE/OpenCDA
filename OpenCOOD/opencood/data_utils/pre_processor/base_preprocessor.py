"""
Functionality for downsampling and projecting points to BEV maps.

This module provides base preprocessor classes for lidar point cloud data.
"""

import numpy as np
from numpy.typing import NDArray

from opencood.utils import pcd_utils
from typing import Dict, Any

class BasePreprocessor:
    """
    Basic Lidar pre-processor.

    Parameters
    ----------
    preprocess_params : Dict[str, Any]
        The dictionary containing all parameters of the preprocessing.
    train : bool
        Train or test mode.
    
    Attributes
    ----------
    params : Dict[str, Any]
        Configuration parameters for preprocessing operations.
    train : bool
        Boolean flag indicating training or evaluation mode.
    """

    def __init__(self, preprocess_params: Dict[str, Any], train: bool):
        self.params = preprocess_params
        self.train = train

    def preprocess(self, pcd_np: NDArray[np.float64]) -> Dict[str, NDArray[np.float64]]:
        """
        Preprocess the lidar points by simple sampling.

        Parameters
        ----------
        pcd_np : NDArray[np.float64]
            The raw lidar.

        Returns
        -------
        Dict[str, NDArray[np.float64]]
            The output dictionary.
        """
        data_dict = {}
        sample_num = self.params["args"]["sample_num"]

        pcd_np = pcd_utils.downsample_lidar(pcd_np, sample_num)
        data_dict["downsample_lidar"] = pcd_np

        return data_dict

    def project_points_to_bev_map(self, 
                                points: NDArray[np.float64], 
                                ratio: float = 0.1) -> NDArray[np.float64]:
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : NDArray[np.float64]
            Point cloud array with shape (N, 3) or (N, 4).
        ratio : float, optional
            Discretization parameters. Default is 0.1.

        Returns
        -------
        NDArray[np.float64]
            BEV occupancy map including projected points with shape
            (img_row, img_col).
        """
        L1, W1, H1, L2, W2, H2 = self.params["cav_lidar_range"]
        img_row = int((L2 - L1) / ratio)
        img_col = int((W2 - W1) / ratio)
        bev_map = np.zeros((img_row, img_col))
        bev_origin = np.array([L1, W1, H1]).reshape(1, -1)
        # (N, 3)
        indices = ((points[:, :3] - bev_origin) / ratio).astype(int)
        mask = np.logical_and(indices[:, 0] > 0, indices[:, 0] < img_row)
        mask = np.logical_and(mask, np.logical_and(indices[:, 1] > 0, indices[:, 1] < img_col))
        indices = indices[mask, :]
        bev_map[indices[:, 0], indices[:, 1]] = 1
        return bev_map