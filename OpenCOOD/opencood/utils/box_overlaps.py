"""Vectorized NumPy operations for axis-aligned 2D boxes."""

from __future__ import annotations

import numpy as np


def bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Compute pairwise intersection over union for axis-aligned 2D boxes.

    Parameters
    ----------
    boxes : numpy.ndarray
        Boxes with shape ``(N, 4)`` represented as
        ``(x_min, y_min, x_max, y_max)``.
    query_boxes : numpy.ndarray
        Query boxes with shape ``(K, 4)`` represented as
        ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    numpy.ndarray
        Pairwise IoU values with shape ``(N, K)`` and dtype ``float32``.

    Notes
    -----
    Widths and heights use the legacy OpenCOOD inclusive-coordinate
    convention and are therefore calculated with ``+1``.
    """
    box_count = boxes.shape[0]
    query_count = query_boxes.shape[0]
    if box_count == 0 or query_count == 0:
        return np.zeros((box_count, query_count), dtype=np.float32)

    intersection_width = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2])
    intersection_width -= np.maximum(boxes[:, None, 0], query_boxes[None, :, 0])
    intersection_width += 1
    np.maximum(intersection_width, 0, out=intersection_width)

    intersection_height = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3])
    intersection_height -= np.maximum(boxes[:, None, 1], query_boxes[None, :, 1])
    intersection_height += 1
    np.maximum(intersection_height, 0, out=intersection_height)

    intersection = intersection_width * intersection_height
    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    union = box_areas[:, None] + query_areas[None, :] - intersection

    overlaps = np.zeros((box_count, query_count), dtype=np.float32)
    np.divide(intersection, union, out=overlaps, where=intersection > 0)
    return overlaps
