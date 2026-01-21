"""
Common utilities.

This module provides various utility functions for data type checking and conversion,
as well as common operations used throughout the OpenCOOD project.
"""

from typing import Any, Dict, List, Tuple, Union, Hashable, Optional

import numpy as np
import torch
import numpy.typing as npt
from shapely.geometry import Polygon
from torch import Tensor


def check_numpy_to_torch(x: Union[npt.NDArray, Any]) -> Tuple[Union[Tensor, Any], bool]:
    """
    Check if input is a numpy array and convert it to a PyTorch tensor if it is.

    Parameters
    ----------
    x : np.ndarray or Any
        Input value which could be a numpy array or any other type.

    Returns
    -------
    converted : torch.Tensor or Any
        The input converted to a PyTorch tensor if it was a numpy array,
        otherwise the original input.
    is_converted : bool
        Boolean indicating whether a conversion was performed (True) or not (False).
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def check_contain_nan(
    x: Union[Dict[str, Any], List[Any], int, float, npt.NDArray[np.floating]]
) -> bool:
    """
    Recursively check if any value in a nested structure contains NaN.

    Parameters
    ----------
    x : dict, list, np.ndarray, int, or float
        Input which can be a dictionary, list, numpy array, or numeric value.

    Returns
    -------
    bool
        True if any NaN value is found in the input structure, False otherwise.

    Notes
    -----
    - For dictionaries, checks all values recursively.
    - For lists, checks all elements recursively.
    - For numpy arrays, uses np.any(np.isnan()).
    - For int or float, always returns False as they cannot be NaN in Python.
    """
    if isinstance(x, dict):
        return any(check_contain_nan(v) for k, v in x.items())
    if isinstance(x, list):
        return any(check_contain_nan(itm) for itm in x)
    if isinstance(x, int) or isinstance(x, float):
        return False
    if isinstance(x, np.ndarray):
        return bool(np.any(np.isnan(x)))
    return bool(torch.any(x.isnan()).detach().cpu().item())


def rotate_points_along_z(points: Union[npt.NDArray, torch.Tensor], angle: Union[npt.NDArray, torch.Tensor]) -> Union[npt.NDArray, torch.Tensor]:
    """
    Rotate points around the z-axis by given angles.

    Parameters
    ----------
    points : np.ndarray or torch.Tensor
        Input points to be rotated with shape (B, N, 3 + C) where:
        - B: batch size
        - N: number of points
        - 3: x, y, z coordinates
        - C: additional channels (not modified by rotation)
    angle : np.ndarray or torch.Tensor
        Rotation angles in radians with shape (B,), where B is the batch size.
        The rotation follows the right-hand rule (counter-clockwise when looking
        from positive z towards origin).

    Returns
    -------
    np.ndarray or torch.Tensor
        Rotated points with the same type and shape as input.
        The first 3 dimensions (x, y, z) are rotated around z-axis, while any
        additional channels remain unchanged.

    Notes
    -----
    - The function handles both numpy arrays and PyTorch tensors as input.
    - The function preserves the input type (numpy array or tensor).
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3].float(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def rotate_points_along_z_2d(points: Union[npt.NDArray, torch.Tensor], angle: Union[npt.NDArray, torch.Tensor]) -> Union[npt.NDArray, torch.Tensor]:
    """
    Rotate the points along z-axis in 2D.

    Parameters
    ----------
    points : torch.Tensor or np.ndarray
        Points with shape (N, 2).
    angle : torch.Tensor or np.ndarray
        Rotation angles with shape (N,).

    Returns
    -------
    torch.Tensor or np.ndarray
        Rotated points with shape (N, 2).
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    # (N, 2, 2)
    rot_matrix = torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2, 2).float()
    points_rot = torch.einsum("ik, ikj->ij", points.float(), rot_matrix)
    return points_rot.numpy() if is_numpy else points_rot


def compute_iou(box: Polygon, boxes: List[Polygon]) -> npt.NDArray:
    """
    Compute IoU between box and boxes list.

    Parameters
    ----------
    box : shapely.geometry.Polygon
        Bounding box Polygon.
    boxes : list of shapely.geometry.Polygon
        List of shapely.geometry.Polygon.

    Returns
    -------
    np.ndarray
        Array of IoU between box and boxes.
    """
    # Calculate intersection areas
    if np.any(np.array([box.union(b).area for b in boxes]) == 0):
        print("debug")
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)


def convert_format(boxes_array: npt.NDArray) -> List[Polygon]:
    """
    Convert boxes array to shapely.geometry.Polygon format.

    Parameters
    ----------
    boxes_array : np.ndarray
        Boxes array with shape (N, 4, 2) or (N, 8, 3).

    Returns
    -------
    list of Polygon
        List of converted shapely.geometry.Polygon objects.
    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return polygons


def torch_tensor_to_numpy(torch_tensor: torch.Tensor) -> npt.NDArray:
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor
        Input PyTorch tensor.

    Returns
    -------
    np.ndarray
        A numpy array.
    """
    return torch_tensor.numpy() if not torch_tensor.is_cuda else torch_tensor.cpu().detach().numpy()


def get_voxel_centers(voxel_coords: Any, downsample_times: Any, voxel_size: Any, point_cloud_range: Any) -> Any:
    """
    Calculate voxel center coordinates.

    Parameters
    ----------
    voxel_coords : np.ndarray
        Voxel coordinates with shape (N, 3).
    downsample_times : int
        Downsampling factor.
    voxel_size : np.ndarray
        Size of each voxel.
    point_cloud_range : np.ndarray
        Point cloud range.

    Returns
    -------
    np.ndarray
        Voxel center coordinates.
    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers
