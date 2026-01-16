"""
Data augmentation utilities for 3D object detection.

This module provides geometric transformation functions for augmenting LiDAR
point clouds and 3D bounding boxes, including flipping, rotation, and scaling
operations.
"""

import numpy as np
from numpy.typing import NDArray
from opencood.utils import common_utils
from typing import Tuple


def random_flip_along_x(
    gt_boxes: NDArray[np.float64], 
    points: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Randomly flip the point cloud and bounding boxes along the X-axis.
    
    Parameters
    ----------
    gt_boxes : NDArray[np.float64]
        Ground truth boxes of shape (N, 7 + C) where:
        - N : number of bounding boxes
        - 7 : [x, y, z, dx, dy, dz, heading]
        - C : optional velocity components [vx, vy, ...]
    points : NDArray[np.float64]
        Point cloud of shape (M, 3 + C) where:
        - M : number of points
        - 3 : x, y, z coordinates
        - C : additional features per point
    
    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        gt_boxes : NDArray[np.float64]
            Flipped ground truth boxes of shape (N, 7 + C).
        points : NDArray[np.float64]
            Flipped points of shape (M, 3 + C).
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(
    gt_boxes: NDArray[np.float64], 
    points: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Flip the point cloud and bounding boxes along the Y-axis.
    
    Parameters
    ----------
    gt_boxes : NDArray[np.float64]
        Ground truth boxes of shape (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]].
    points : NDArray[np.float64]
        Point cloud of shape (M, 3 + C).
    
    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        gt_boxes : NDArray[np.float64]
            Flipped ground truth boxes.
        points : NDArray[np.float64]
            Flipped points.
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(
    gt_boxes: NDArray[np.float64], 
    points: NDArray[np.float64], 
    rot_range: Tuple[float, float]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply global rotation to point cloud and bounding boxes.
    
    Parameters
    ----------
    gt_boxes : NDArray[np.float64]
        Ground truth boxes of shape (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]].
    points : NDArray[np.float64]
        Point cloud of shape (M, 3 + C).
    rot_range : Tuple[float, float]
        Rotation angle range [min, max].
    
    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        gt_boxes : NDArray[np.float64]
            Rotated ground truth boxes.
        points : NDArray[np.float64]
            Rotated points.
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]

    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation]),
        )[0][:, 0:2]

    return gt_boxes, points


def global_scaling(
    gt_boxes: NDArray[np.float64], 
    points: NDArray[np.float64], 
    scale_range: Tuple[float, float]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply global scaling to point cloud and bounding boxes.
    
    Parameters
    ----------
    gt_boxes : NDArray[np.float64]
        Ground truth boxes of shape (N, 7), [x, y, z, dx, dy, dz, heading].
    points : NDArray[np.float64]
        Point cloud of shape (M, 3 + C).
    scale_range : Tuple[float, float]
        Scale factor range [min, max].
    
    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        gt_boxes : NDArray[np.float64]
            Scaled ground truth boxes.
        points : NDArray[np.float64]
            Scaled points.
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    return gt_boxes, points