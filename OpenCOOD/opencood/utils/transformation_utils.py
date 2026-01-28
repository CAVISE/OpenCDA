"""
Transformation utilities for coordinate system conversions.

This module provides functions for transforming coordinates between different
reference frames, including conversions between vehicle coordinate systems
and continuous/discretized coordinate spaces.
"""

from typing import List

import numpy as np
import numpy.typing as npt


def x_to_world(pose: List[float]) -> npt.NDArray[np.floating]:
    """
    Create transformation matrix from x-coordinate system to carla world system.

    Parameters
    ----------
    pose : list of float
        Vehicle pose as [x, y, z, roll, yaw, pitch].

    Returns
    -------
    np.ndarray
        The 4x4 transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1: List[float], x2: List[float]) -> npt.NDArray[np.floating]:
    """
    Compute transformation matrix from coordinate system x1 to x2.

    Parameters
    ----------
    x1 : list of float
        The pose of x1 under world coordinates.
    x2 : list of float
        The pose of x2 under world coordinates.

    Returns
    -------
    np.ndarray
        The transformation matrix from x1 to x2.
    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def dist_to_continuous(
    p_dist: npt.NDArray[np.floating], displacement_dist: npt.NDArray[np.floating], res: float, downsample_rate: int
) -> npt.NDArray[np.floating]:
    """
    Convert points from discretized format to continuous space for BEV representation.

    Parameters
    ----------
    p_dist : np.ndarray
        Points in discretized coordinates.
    displacement_dist : np.ndarray
        Discretized coordinates of bottom left origin.
    res : float
        Discretization resolution.
    downsample_rate : int
        Downsampling rate.

    Returns
    -------
    np.ndarray
        Points in continuous coordinates.
    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous
