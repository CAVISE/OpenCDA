"""
Utility functions related to point cloud processing.

This module provides various utilities for working with point cloud data,
including reading, filtering, transforming, and downsampling operations.
"""

from typing import List

import numpy as np
import numpy.typing as npt
import open3d as o3d


def pcd_to_np(pcd_file: str) -> npt.NDArray[np.float32]:
    """
    Read PCD file and return numpy array.

    Parameters
    ----------
    pcd_file : str
        Path to the PCD file containing the point cloud.

    Returns
    -------
    np.ndarray
        The lidar data in numpy format with shape (n, 4), where the last
        column represents intensity.
    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def mask_points_by_range(points: npt.NDArray[np.floating], limit_range: List[float]) -> npt.NDArray[np.floating]:
    """
    Remove lidar points outside the specified boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.
    limit_range : list of float
        Boundary limits as [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns
    -------
    np.ndarray
        Filtered lidar points within the specified range.
    """
    mask = (
        (points[:, 0] > limit_range[0])
        & (points[:, 0] < limit_range[3])
        & (points[:, 1] > limit_range[1])
        & (points[:, 1] < limit_range[4])
        & (points[:, 2] > limit_range[2])
        & (points[:, 2] < limit_range[5])
    )

    points = points[mask]

    return points


def mask_ego_points(points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Remove lidar points corresponding to the ego vehicle itself.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    Returns
    -------
    np.ndarray
        Filtered lidar points with ego vehicle points removed.
    """
    mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1)
    points = points[np.logical_not(mask)]

    return points


def shuffle_points(points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Randomly shuffle the order of points.

    Parameters
    ----------
    points : np.ndarray
        Input points with shape (n, m).

    Returns
    -------
    np.ndarray
        Shuffled points with same shape as input.
    """
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]

    return points


def projected_lidar_stack(projected_lidar_list: List[npt.NDArray[np.floating]]) -> npt.NDArray[np.floating]:
    """
    Stack all projected lidar point clouds together.

    Parameters
    ----------
    projected_lidar_list : list of np.ndarray
        List containing projected lidar point clouds.

    Returns
    -------
    np.ndarray
        Vertically stacked lidar data from all point clouds.
    """
    stack_lidar = []
    for lidar_data in projected_lidar_list:
        stack_lidar.append(lidar_data)

    return np.vstack(stack_lidar)


def downsample_lidar(pcd_np: npt.NDArray[np.floating], num: int) -> npt.NDArray[np.floating]:
    """
    Downsample lidar points to a specified number.

    Parameters
    ----------
    pcd_np : np.ndarray
        The lidar points with shape (n, 4).
    num : int
        Target number of points after downsampling.

    Returns
    -------
    np.ndarray
        Downsampled lidar points with shape (num, 4).

    Raises
    ------
    AssertionError
        If the input has fewer points than the target number.
    """
    assert pcd_np.shape[0] >= num

    selected_index = np.random.choice((pcd_np.shape[0]), num, replace=False)
    pcd_np = pcd_np[selected_index]

    return pcd_np


def downsample_lidar_minimum(pcd_np_list: List[npt.NDArray[np.floating]]) -> List[npt.NDArray[np.floating]]:
    """
    Downsample all point clouds to match the minimum point count.

    Given a list of point clouds, finds the one with minimum number of points
    and downsamples all others to match that count.

    Parameters
    ----------
    pcd_np_list : list of np.ndarray
        List of point cloud numpy arrays, each with shape (n, 4).

    Returns
    -------
    list of np.ndarray
        List of downsampled point clouds, all with the same number of points.
    """
    minimum: int = np.iinfo(np.int64).max

    for i in range(len(pcd_np_list)):
        num = pcd_np_list[i].shape[0]
        minimum = num if minimum > num else minimum

    for i, pcd_np in enumerate(pcd_np_list):
        pcd_np_list[i] = downsample_lidar(pcd_np, minimum)

    return pcd_np_list
