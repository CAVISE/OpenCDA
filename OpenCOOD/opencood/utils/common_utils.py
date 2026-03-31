"""
Common utilities
"""

import numpy as np
import torch
from shapely.geometry import Polygon


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def check_contain_nan(x):
    if isinstance(x, dict):
        return any(check_contain_nan(v) for k, v in x.items())
    if isinstance(x, list):
        return any(check_contain_nan(itm) for itm in x)
    if isinstance(x, int) or isinstance(x, float):
        return False
    if isinstance(x, np.ndarray):
        return np.any(np.isnan(x))
    return torch.any(x.isnan()).detach().cpu().item()


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), radians, angle along z-axis, angle increases x ==> y
    Returns:

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


def rotate_points_along_z_2d(points, angle):
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


def compute_iou(box, boxes):
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


def convert_format(boxes_array):
    """
    Convert boxes array to shapely.geometry.Polygon format.
    Parameters
    ----------
    boxes_array : np.ndarray
        (N, 4, 2) or (N, 8, 3) or (N, 8) or (N, 7) or (7,) or (8,).

    Returns
    -------
        list of converted shapely.geometry.Polygon object.

    """
    # Ensure input is at least 2D
    if boxes_array.ndim == 1:
        boxes_array = boxes_array[np.newaxis, :]
    
    # Handle case where input is (N, 8) - reshape to (N, 4, 2)
    if boxes_array.ndim == 2 and boxes_array.shape[1] == 8:
        boxes_array = boxes_array.reshape(-1, 4, 2)
    # Handle case where input is (N, 7) - convert from [x, y, z, length, width, height, yaw] to corners
    elif boxes_array.ndim == 2 and boxes_array.shape[1] == 7:
        x = boxes_array[:, 0]
        y = boxes_array[:, 1]
        l = boxes_array[:, 3]
        w = boxes_array[:, 4]
        yaw = boxes_array[:, 6]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        half_l = l / 2.0
        half_w = w / 2.0
        corners = np.zeros((boxes_array.shape[0], 4, 2))
        # front left
        corners[:, 0, 0] = x + half_l * cos_yaw - half_w * sin_yaw
        corners[:, 0, 1] = y + half_l * sin_yaw + half_w * cos_yaw
        # front right
        corners[:, 1, 0] = x + half_l * cos_yaw + half_w * sin_yaw
        corners[:, 1, 1] = y + half_l * sin_yaw - half_w * cos_yaw
        # rear right
        corners[:, 2, 0] = x - half_l * cos_yaw + half_w * sin_yaw
        corners[:, 2, 1] = y - half_l * sin_yaw - half_w * cos_yaw
        # rear left
        corners[:, 3, 0] = x - half_l * cos_yaw - half_w * sin_yaw
        corners[:, 3, 1] = y - half_l * sin_yaw + half_w * cos_yaw
        boxes_array = corners
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)


def torch_tensor_to_numpy(torch_tensor):
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor

    Returns
    -------
    A numpy array.
    """
    if isinstance(torch_tensor, np.ndarray):
        return torch_tensor
    return torch_tensor.numpy() if not torch_tensor.is_cuda else torch_tensor.cpu().detach().numpy()


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers
