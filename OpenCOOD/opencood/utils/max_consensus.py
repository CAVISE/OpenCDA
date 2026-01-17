"""
Maximum consensus algorithm for point cloud registration.

This module implements hierarchical maximum consensus methods for aligning
point clouds using transformation search and nearest neighbor matching.
"""

from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors


def max_consunsus_hierarchical(
    pointsl: npt.NDArray[np.floating],
    pointsr: npt.NDArray[np.floating],
    loc_l: npt.NDArray[np.floating],
    loc_r: npt.NDArray[np.floating],
    resolution: Optional[npt.NDArray[np.floating]] = None,
    radius: float = 1,
    point_labels: Optional[Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]] = None,
    label_weights: Optional[npt.NDArray[np.floating]] = None,
    **kwargs: Any,
) -> Tuple[Optional[npt.NDArray[np.floating]], Optional[npt.NDArray[np.floating]], Optional[npt.NDArray[np.floating]]]:
    """
    Perform hierarchical maximum consensus for point cloud registration.

    Parameters
    ----------
    pointsl : np.ndarray
        Source point cloud with shape (N, 3).
    pointsr : np.ndarray
        Target point cloud with shape (M, 3).
    loc_l : np.ndarray
        Location of source point cloud with shape (1, 3).
    loc_r : np.ndarray
        Location of target point cloud with shape (1, 3).
    resolution : np.ndarray, optional
        Resolution for transformation search.
    radius : float, default=1
        Search radius for nearest neighbor matching.
    point_labels : tuple of np.ndarray, optional
        Tuple of point labels for source and target.
    label_weights : np.ndarray, optional
        Weights for different point labels.
    **kwargs : Any
        Additional keyword arguments:
        - search_range : Search range for transformation.
        - min_cons : Minimum consensus threshold.
        - min_match_acc_points : Minimum number of matched points.

    Returns
    -------
    T : np.ndarray or None
        Transformation matrix with shape (3, 3), or None if no good match.
    tf_local : np.ndarray or None
        Local transformation parameters, or None if no good match.
    pointsr_out : np.ndarray or None
        Transformed target points, or None if no good match.
    """
    max_err = kwargs["search_range"]  # np.array([1, 1, 6])
    min_cons = kwargs["min_cons"]
    min_match_acc_points = kwargs["min_match_acc_points"]
    pointsl_out, pointsr_out, T, tf_local, cons, matched_pointsl, matched_pointsr = max_consensus2(
        pointsl,
        pointsr,
        -max_err,
        max_err,
        resolution,
        radius,
        loc_l,
        loc_r,
        point_labels=point_labels,
        label_weights=label_weights,
    )

    if matched_pointsl is not None and len(matched_pointsl) > min_match_acc_points:
        T, tf = estimate_tf_2d(matched_pointsl, matched_pointsr, pointsl, pointsr_out)
        tf_local = tf
        tf_local[:2] = tf_local[:2] = tf_local[:2] - loc_r[0, :2] + loc_l[0, :2]
        pointsr_homo = np.concatenate([pointsr, np.ones((len(pointsr), 1))], axis=1).T
        pointsr_out = (T @ pointsr_homo).T
    else:
        return None, None, None

    if cons < min_cons:
        return None, None, None
    return T, tf_local, pointsr_out


def max_consensus2(
    pointsl: npt.NDArray[np.floating],
    pointsr: npt.NDArray[np.floating],
    xyr_min: npt.NDArray[np.floating],
    xyr_max: npt.NDArray[np.floating],
    resolution: npt.NDArray[np.floating],
    radius: float,
    loc_l: Optional[npt.NDArray[np.floating]] = None,
    loc_r: Optional[npt.NDArray[np.floating]] = None,
    point_labels: Optional[Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]] = None,
    label_weights: Optional[npt.NDArray[np.floating]] = None,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    Optional[npt.NDArray[np.floating]],
    Optional[npt.NDArray[np.floating]],
    float,
    Optional[npt.NDArray[np.floating]],
    Optional[npt.NDArray[np.floating]],
]:
    """
    Perform maximum consensus matching between two point clouds.

    Parameters
    ----------
    pointsl : np.ndarray
        Source point cloud with shape (N, 3).
    pointsr : np.ndarray
        Target point cloud with shape (M, 3).
    xyr_min : np.ndarray
        Minimum bounds for transformation search (dx, dy, dθ).
    xyr_max : np.ndarray
        Maximum bounds for transformation search (dx, dy, dθ).
    resolution : np.ndarray
        Resolution for transformation search.
    radius : float
        Search radius for nearest neighbor matching.
    loc_l : np.ndarray, optional
        Location of source point cloud.
    loc_r : np.ndarray, optional
        Location of target point cloud.
    point_labels : tuple of np.ndarray, optional
        Tuple of point labels for source and target.
    label_weights : np.ndarray, optional
        Weights for different point labels.

    Returns
    -------
    pointl_out : np.ndarray
        Transformed source points.
    pointr_out : np.ndarray
        Transformed target points.
    match_T : np.ndarray or None
        Best transformation matrix.
    match_tf_local : np.ndarray or None
        Local transformation parameters.
    cur_cons : float
        Consensus score.
    matched_pointsl : np.ndarray or None
        Matched source points.
    matched_pointsr : np.ndarray or None
        Matched target points.
    """
    tf_matrices, tf_params, tf_params_local = construct_tfs(xyr_min, xyr_max, resolution, loc_l, loc_r)
    rotl, _, _ = construct_tfs(xyr_min[2:], xyr_max[2:], resolution[2:])
    pointr_homo = np.concatenate([pointsr, np.ones((len(pointsr), 1))], axis=1).T
    # pointl_homo = np.concatenate([pointsl, np.ones((len(pointsl), 1))], axis=1).T
    pointr_transformed = np.einsum("...ij, ...jk", tf_matrices, np.tile(pointr_homo, (len(tf_matrices), 1, 1))).transpose(0, 2, 1)
    pointr_transformed_s = pointr_transformed.reshape(-1, 3)[:, :2]
    cur_cons = 0
    pointl_out = pointsl
    pointr_out = pointsr
    match_T, match_tf_local, matched_pointsl, matched_pointsr = None, None, None, None
    # r1 = 0
    for R in rotl[:, :2, :2]:
        pointl_transformed = np.einsum("ij, jk", R, pointsl.T).T
        nbrs = NearestNeighbors(n_neighbors=1, radius=radius, algorithm="auto").fit(pointl_transformed)
        distances, indices = nbrs.kneighbors(pointr_transformed_s)
        mask = distances < radius
        lbll, lblr = point_labels
        plus = (np.logical_and(lbll[indices] > 2, mask)).reshape(len(tf_matrices), len(pointsr))
        mask = mask.reshape(len(tf_matrices), len(pointsr))
        pointr_consensus = mask.sum(axis=1) + plus.sum(axis=1) * label_weights[-1]
        best_match = np.argmax(pointr_consensus)
        match_consensus = pointr_consensus[best_match]
        if match_consensus > cur_cons:
            pointr_out = pointr_transformed[best_match]
            match_T = tf_matrices[best_match]
            match_tf_local = tf_params_local[best_match]
            accurate_points_mask = plus[best_match]
            selected_indices = indices.reshape(len(tf_matrices), len(pointsr))[best_match][accurate_points_mask]
            matched_pointsl = pointsl[selected_indices]
            matched_pointsr = pointsr[accurate_points_mask]
            # r1 = np.arctan2(R[1, 0], R[0, 0])
            pointl_out = pointl_transformed
            cur_cons = match_consensus
    return (
        pointl_out,
        pointr_out,
        match_T,
        match_tf_local,
        cur_cons,
        matched_pointsl,
        matched_pointsr,
    )


def max_consensus1(
    pointsl: npt.NDArray[np.floating],
    pointsr: npt.NDArray[np.floating],
    xyr_min: npt.NDArray[np.floating],
    xyr_max: npt.NDArray[np.floating],
    resolution: npt.NDArray[np.floating],
    radius: float,
    loc_l: Optional[npt.NDArray[np.floating]] = None,
    loc_r: Optional[npt.NDArray[np.floating]] = None,
    point_labels: Optional[Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]] = None,
    label_weights: Optional[npt.NDArray[np.floating]] = None,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Alternative implementation of maximum consensus matching.

    Parameters
    ----------
    pointsl : np.ndarray
        Source point cloud with shape (N, 3).
    pointsr : np.ndarray
        Target point cloud with shape (M, 3).
    xyr_min : np.ndarray
        Minimum bounds for transformation search.
    xyr_max : np.ndarray
        Maximum bounds for transformation search.
    resolution : np.ndarray
        Resolution for transformation search.
    radius : float
        Search radius for nearest neighbor matching.
    loc_l : np.ndarray, optional
        Location of source point cloud.
    loc_r : np.ndarray, optional
        Location of target point cloud.
    point_labels : tuple of np.ndarray, optional
        Tuple of point labels for source and target.
    label_weights : np.ndarray, optional
        Weights for different point labels.

    Returns
    -------
    pointr_out : np.ndarray
        Transformed target points.
    match_T : np.ndarray
        Best transformation matrix.
    match_tf_local : np.ndarray
        Local transformation parameters.
    match_consensus : float
        Consensus score.
    matched_pointsl : np.ndarray
        Matched source points.
    matched_pointsr : np.ndarray
        Matched target points.
    """
    tf_matrices, tf_params, tf_params_local = construct_tfs(xyr_min, xyr_max, resolution, loc_l, loc_r)
    pointr_homo = np.concatenate([pointsr, np.ones((len(pointsr), 1))], axis=1).T
    pointr_transformed = np.einsum("...ij, ...jk", tf_matrices, np.tile(pointr_homo, (len(tf_matrices), 1, 1))).transpose(0, 2, 1)
    pointr_transformed_s = pointr_transformed.reshape(-1, 3)[:, :2]

    nbrs = NearestNeighbors(n_neighbors=1, radius=radius, algorithm="auto").fit(pointsl)
    distances, indices = nbrs.kneighbors(pointr_transformed_s)
    mask = distances < radius
    lbll, lblr = point_labels
    plus = (np.logical_and(lbll[indices] > 2, mask)).reshape(len(tf_matrices), len(pointsr))
    mask = mask.reshape(len(tf_matrices), len(pointsr))
    pointr_consensus = mask.sum(axis=1) + plus.sum(axis=1) * label_weights[-1]
    best_match = np.argmax(pointr_consensus)
    match_consensus = pointr_consensus[best_match]
    pointr_out = pointr_transformed[best_match]
    _ = tf_params[best_match]  # match_tf
    match_T = tf_matrices[best_match]
    match_tf_local = tf_params_local[best_match]
    accurate_points_mask = plus[best_match]
    selected_indices = indices.reshape(len(tf_matrices), len(pointsr))[best_match][accurate_points_mask]
    matched_pointsl = pointsl[selected_indices]
    matched_pointsr = pointsr[accurate_points_mask]
    return (
        pointr_out,
        match_T,
        match_tf_local,
        match_consensus,
        matched_pointsl,
        matched_pointsr,
    )


def construct_tfs(
    xyr_min: npt.NDArray[np.floating],
    xyr_max: npt.NDArray[np.floating],
    resolution: npt.NDArray[np.floating],
    loc_l: Optional[npt.NDArray[np.floating]] = None,
    loc_r: Optional[npt.NDArray[np.floating]] = None,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Construct transformation matrices for maximum consensus search.

    Parameters
    ----------
    xyr_min : np.ndarray
        Minimum bounds for transformation search.
    xyr_max : np.ndarray
        Maximum bounds for transformation search.
    resolution : np.ndarray
        Resolution for transformation search.
    loc_l : np.ndarray, optional
        Source location.
    loc_r : np.ndarray, optional
        Target location.

    Returns
    -------
    tfs : np.ndarray
        Transformation matrices.
    tf_parames : np.ndarray
        Transformation parameters.
    tf_parames_local : np.ndarray
        Local transformation parameters.
    """
    input = [np.arange(xyr_min[i], xyr_max[i], resolution[i]) for i in range(len(xyr_min))]
    grid = np.meshgrid(*input)
    grid = [a.reshape(-1) for a in grid]
    tf_parames_local = np.stack(grid, axis=1)
    tf_parames_local[:, -1] = tf_parames_local[:, -1] / 180 * np.pi
    tf_parames = np.copy(tf_parames_local)
    if loc_r is not None:
        tf_parames[:, :-1] = tf_parames_local[:, :2] + loc_r[:, :2] - loc_l[:, :2]
    sina = np.sin(tf_parames[:, -1])
    cosa = np.cos(tf_parames[:, -1])
    zeros = np.zeros(len(tf_parames), dtype=sina.dtype)
    ones = np.ones(len(tf_parames), dtype=sina.dtype)
    x = tf_parames[:, 0] if len(xyr_min) > 1 else zeros
    y = tf_parames[:, 1] if len(xyr_min) > 1 else zeros
    tfs = np.array([[cosa, -sina, x], [sina, cosa, y], [zeros, zeros, ones]]).transpose(2, 0, 1)
    return tfs, tf_parames, tf_parames_local


def estimate_tf_2d(
    pointsr: npt.NDArray[np.floating], pointsl: npt.NDArray[np.floating], pointsl_all: npt.NDArray[np.floating], pointsr_all: npt.NDArray[np.floating]
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Estimate 2D transformation between matched point sets.

    Parameters
    ----------
    pointsr : np.ndarray
        Source points with shape (N, 2) or (N, 3).
    pointsl : np.ndarray
        Target points with shape (N, 2) or (N, 3).
    pointsl_all : np.ndarray
        All source points (unused, for API compatibility).
    pointsr_all : np.ndarray
        All target points (unused, for API compatibility).

    Returns
    -------
    T : np.ndarray
        Homogeneous transformation matrix with shape (3, 3).
    tf_params : np.ndarray
        Transformation parameters [tx, ty, θ].
    """
    # 1 reduce by the center of mass
    l_mean = pointsl.mean(axis=0)
    r_mean = pointsr.mean(axis=0)
    l_reduced = pointsl - l_mean
    r_reduced = pointsr - r_mean
    # 2 compute the rotation
    Sxx = (l_reduced[:, 0] * r_reduced[:, 0]).sum()
    Syy = (l_reduced[:, 1] * r_reduced[:, 1]).sum()
    Sxy = (l_reduced[:, 0] * r_reduced[:, 1]).sum()
    Syx = (l_reduced[:, 1] * r_reduced[:, 0]).sum()
    theta = np.arctan2(Sxy - Syx, Sxx + Syy)  # / np.pi * 180
    sa = np.sin(theta)
    ca = np.cos(theta)
    T = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    t = r_mean.reshape(2, 1) - T[:2, :2] @ l_mean.reshape(2, 1)
    # T = T.T
    T[:2, 2:] = t
    return T, np.array([*t.squeeze(), theta])
