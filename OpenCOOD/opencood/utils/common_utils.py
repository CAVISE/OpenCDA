"""
Common utilities
This module provides various utility functions for data type checking and conversion,
as well as common operations used throughout the OpenCOOD project.
"""

import numpy as np
import torch
from shapely.geometry import Polygon

from typing import Union, Any, Tuple, Dict, List
from numpy import ndarray
from torch import Tensor

def check_numpy_to_torch(x: Union[ndarray, Any]) -> Tuple[Union[Tensor, Any], bool]:
    """Check if input is a numpy array and convert it to a PyTorch tensor if it is.
    Args:
        x: Input value which could be a numpy array or any other type.
    Returns:
        Tuple containing:
            - The input converted to a PyTorch tensor if it was a numpy array, 
              otherwise the original input.
            - Boolean indicating whether a conversion was performed (True) or not (False).
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def check_contain_nan(x: Union[Dict[str, Any], List[Any], int, float, ndarray]) -> bool:
    """Recursively check if any value in a nested structure contains NaN.
    Args:
        x: Input which can be a dictionary, list, numpy array, or numeric value.
    Returns:
        bool: True if any NaN value is found in the input structure, False otherwise.
    Note:
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
        return np.any(np.isnan(x))
    return torch.any(x.isnan()).detach().cpu().item()


def rotate_points_along_z(points: Union[np.ndarray, torch.Tensor], 
                         angle: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Rotate points around the z-axis by given angles.
    Args:
        points (Union[np.ndarray, torch.Tensor]): Input points to be rotated. 
            Shape should be (B, N, 3 + C) where:
                - B: batch size
                - N: number of points
                - 3: x, y, z coordinates
                - C: additional channels (not modified by rotation)
        angle (Union[np.ndarray, torch.Tensor]): Rotation angles in radians. 
            Shape should be (B,), where B is the batch size.
            The rotation follows the right-hand rule (counter-clockwise when looking
            from positive z towards origin).
    Returns:
        Union[np.ndarray, torch.Tensor]: Rotated points with the same type and shape as input.
            The first 3 dimensions (x, y, z) are rotated around z-axis, while any
            additional channels remain unchanged.
    Note:
        - The function handles both numpy arrays and PyTorch tensors as input.
        - The rotation is performed in-place for efficiency.
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


def rotate_points_along_z_2d(points: Union[np.ndarray, torch.Tensor], 
                            angle: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Rorate the points along z-axis.
    Parameters
    ----------
    points : torch.Tensor / np.ndarray
        (N, 2).
    angle : torch.Tensor / np.ndarray
        (N,)

    Returns
    -------
    points_rot : torch.Tensor / np.ndarray
        Rorated points with shape (N, 2)

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    # (N, 2, 2)
    rot_matrix = torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2, 2).float()
    points_rot = torch.einsum("ik, ikj->ij", points.float(), rot_matrix)
    return points_rot.numpy() if is_numpy else points_rot


def remove_ego_from_objects(objects: Dict[str, Any], ego_id: int):
    """
    Avoid adding ego vehicle to the object dictionary.

    Parameters
    ----------
    objects : dict
        The dictionary contained all objects.

    ego_id : int
        Ego id.
    """
    if ego_id in objects:
        del objects[ego_id]


def retrieve_ego_id(base_data_dict: Dict[str, Any]) -> str:
    """
    Retrieve the ego vehicle id from sample(origin format).

    Parameters
    ----------
    base_data_dict : dict
        Data sample in origin format.

    Returns
    -------
    ego_id : str
        The id of ego vehicle.
    """
    ego_id = None

    for cav_id, cav_content in base_data_dict.items():
        if cav_content["ego"]:
            ego_id = cav_id
            break
    return ego_id


def compute_iou(box: Polygon, boxes: List[Polygon]) -> np.ndarray:
    """
    Compute iou between box and boxes list
    Parameters
    ----------
    box : shapely.geometry.Polygon
        Bounding box Polygon.

    boxes : list
        List of shapely.geometry.Polygon.

    Returns
    -------
    iou : np.ndarray
        Array of iou between box and boxes.

    """
    # Calculate intersection areas
    if np.any(np.array([box.union(b).area for b in boxes]) == 0):
        print("debug")
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)


def convert_format(boxes_array: np.ndarray) -> List[Polygon]:
    """
    Convert boxes array to shapely.geometry.Polygon format.
    Parameters
    ----------
    boxes_array : np.ndarray
        (N, 4, 2) or (N, 8, 3).

    Returns
    -------
    List[Polygon]:
        list of converted shapely.geometry.Polygon object.

    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return polygons


def torch_tensor_to_numpy(torch_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor

    Returns
    -------
    np.ndarray:
        A numpy array.
    """
    return torch_tensor.numpy() if not torch_tensor.is_cuda else torch_tensor.cpu().detach().numpy()


def get_voxel_centers(voxel_coords: np.ndarray, downsample_times: int, voxel_size: np.ndarray, point_cloud_range: np.ndarray) -> np.ndarray:
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:
        np.ndarray:
    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers
