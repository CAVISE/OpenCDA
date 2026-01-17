"""
Matplotlib-based 2D visualization utilities for point clouds and bounding boxes.

This module provides functions for drawing point clouds and bounding boxes
in 2D bird's eye view using matplotlib.
"""

from typing import Optional, List, Any, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def draw_box_plt(
    boxes_dec: Union[npt.NDArray[np.floating], Any], ax: plt.Axes, color: Optional[Any] = None, linewidth_scale: float = 1.0
) -> plt.Axes:
    """
    Draw bounding boxes in a given matplotlib axes.

    Parameters
    ----------
    boxes_dec : np.ndarray or torch.Tensor
        Bounding boxes with shape (N, 5) or (N, 7) in metric units.
        Format is [x, y, dx, dy, theta] or [x, y, z, dx, dy, dz, theta].
    ax : matplotlib.axes.Axes
        Matplotlib axes object to draw on.
    color : Any, optional
        Color specification for the boxes. Default is None.
    linewidth_scale : float, optional
        Scale factor for line width. Default is 1.0.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with drawn boxes.
    """
    if not len(boxes_dec) > 0:
        return ax
    boxes_np = boxes_dec
    if not isinstance(boxes_np, np.ndarray):
        boxes_np = boxes_np.cpu().detach().numpy()
    if boxes_np.shape[-1] > 5:
        boxes_np = boxes_np[:, [0, 1, 3, 4, 6]]
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    theta = boxes_np[:, 4:5]
    # bl, fl, fr, br
    corners = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).transpose(2, 0, 1)
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1] - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1] - y[:, None]) * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_x, new_y], axis=2)
    for corner in corners:
        ax.plot(corner[[0, 1, 2, 3, 0], 0], corner[[0, 1, 2, 3, 0], 1], color=color, linewidth=0.5 * linewidth_scale)
        # draw front line (
        ax.plot(corner[[2, 3], 0], corner[[2, 3], 1], color=color, linewidth=2 * linewidth_scale)
    return ax


def draw_points_pred_gt_boxes_plt_2d(
    pc_range: List[float],
    points: Optional[npt.NDArray[np.floating]] = None,
    boxes_pred: Optional[Union[npt.NDArray[np.floating], Any]] = None,
    boxes_gt: Optional[Union[npt.NDArray[np.floating], Any]] = None,
) -> None:
    """
    Draw points, predicted boxes, and ground truth boxes in a 2D plot.

    Parameters
    ----------
    pc_range : list of float
        Point cloud range as [x_min, y_min, z_min, x_max, y_max, z_max] defining the plot area.
    points : np.ndarray or None, optional
        Point cloud data with shape (N, 3) or (N, 4). Default is None.
    boxes_pred : np.ndarray or torch.Tensor or None, optional
        Predicted bounding boxes with shape (M, 5) or (M, 7). Default is None.
    boxes_gt : np.ndarray or torch.Tensor or None, optional
        Ground truth bounding boxes with shape (K, 5) or (K, 7). Default is None.
    """
    ax = plt.figure(figsize=(14, 4)).add_subplot(1, 1, 1)
    ax.set_aspect("equal", "box")
    ax.set(xlim=(pc_range[0], pc_range[3]), ylim=(pc_range[1], pc_range[4]))
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], "y.", markersize=0.3)
    if (boxes_gt is not None) and len(boxes_gt) > 0:
        ax = draw_box_plt(boxes_gt, ax, color="green")
    if (boxes_pred is not None) and len(boxes_pred) > 0:
        ax = draw_box_plt(boxes_pred, ax, color="red")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
    plt.close()


def draw_points_boxes_plt_2d(
    ax: plt.Axes,
    pc_range: List[float],
    points: Optional[npt.NDArray[np.floating]] = None,
    boxes: Optional[Union[npt.NDArray[np.floating], Any]] = None,
    color: Optional[Any] = None,
) -> plt.Axes:
    """
    Draw points and boxes in a given matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to draw on.
    pc_range : list of float
        Point cloud range as [x_min, y_min, z_min, x_max, y_max, z_max].
    points : np.ndarray or None, optional
        Point cloud data with shape (N, 3) or (N, 4). Default is None.
    boxes : np.ndarray or torch.Tensor or None, optional
        Bounding boxes with shape (M, 5) or (M, 7). Default is None.
    color : Any, optional
        Color specification for the points and boxes. Default is None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with drawn points and boxes.
    """
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], ".", markersize=0.3, color=color)
    if (boxes is not None) and len(boxes) > 0:
        ax = draw_box_plt(boxes, ax, color=color)

    return ax
