import numpy as np
from sklearn.neighbors import NearestNeighbors

from typing import Optional, Any, Tuple

def max_consunsus_hierarchical(
    pointsl: np.ndarray,
    pointsr: np.ndarray,
    loc_l: np.ndarray,
    loc_r: np.ndarray,
    resolution: Optional[np.ndarray] = None,
    radius: float = 1,
    point_labels: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    label_weights: Optional[np.ndarray] = None,
    **kwargs: Any
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Perform hierarchical maximum consensus for point cloud registration.
    
    Args:
        pointsl: Source point cloud (N x 3).
        pointsr: Target point cloud (M x 3).
        loc_l: Location of source point cloud (1 x 3).
        loc_r: Location of target point cloud (1 x 3).
        resolution: Resolution for transformation search.
        radius: Search radius for nearest neighbor matching.
        point_labels: Tuple of point labels for source and target.
        label_weights: Weights for different point labels.
        **kwargs: Additional arguments:
            - search_range: Search range for transformation.
            - min_cons: Minimum consensus threshold.
            - min_match_acc_points: Minimum number of matched points.
            
    Returns:
        Tuple containing:
            - Transformation matrix (3x3) or None if no good match
            - Local transformation parameters or None
            - Transformed target points or None
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
    pointsl: np.ndarray,
    pointsr: np.ndarray,
    xyr_min: np.ndarray,
    xyr_max: np.ndarray,
    resolution: np.ndarray,
    radius: float,
    loc_l: Optional[np.ndarray] = None,
    loc_r: Optional[np.ndarray] = None,
    point_labels: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    label_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Perform maximum consensus matching between two point clouds.
    
    Args:
        pointsl: Source point cloud (N x 3).
        pointsr: Target point cloud (M x 3).
        xyr_min: Minimum bounds for transformation search (dx, dy, dθ).
        xyr_max: Maximum bounds for transformation search (dx, dy, dθ).
        resolution: Resolution for transformation search.
        radius: Search radius for nearest neighbor matching.
        loc_l: Optional location of source point cloud.
        loc_r: Optional location of target point cloud.
        point_labels: Optional tuple of point labels for source and target.
        label_weights: Optional weights for different point labels.
        
    Returns:
        Tuple containing:
            - Transformed source points
            - Transformed target points
            - Best transformation matrix
            - Local transformation parameters
            - Consensus score
            - Matched source points
            - Matched target points
    """
    tf_matrices, tf_params, tf_params_local = construct_tfs(xyr_min, xyr_max, resolotion, loc_l, loc_r)
    rotl, _, _ = construct_tfs(xyr_min[2:], xyr_max[2:], resolotion[2:])
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
    pointsl: np.ndarray,
    pointsr: np.ndarray,
    xyr_min: np.ndarray,
    xyr_max: np.ndarray,
    resolution: np.ndarray,
    radius: float,
    loc_l: Optional[np.ndarray] = None,
    loc_r: Optional[np.ndarray] = None,
    point_labels: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    label_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Alternative implementation of maximum consensus matching.
    """
    tf_matrices, tf_params, tf_params_local = construct_tfs(xyr_min, xyr_max, resolotion, loc_l, loc_r)
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
    xyr_min: np.ndarray,
    xyr_max: np.ndarray,
    resolution: np.ndarray,
    loc_l: Optional[np.ndarray] = None,
    loc_r: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct transformation matrices for maximum consensus.
    
    Args:
        xyr_min: Minimum bounds for transformation search.
        xyr_max: Maximum bounds for transformation search.
        resolution: Resolution for transformation search.
        loc_l: Optional source location.
        loc_r: Optional target location.
        
    Returns:
        Tuple of:
            - Transformation matrices
            - Transformation parameters
            - Local transformation parameters
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
    pointsr: np.ndarray,
    pointsl: np.ndarray,
    pointsl_all: np.ndarray,
    pointsr_all: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate 2D transformation between matched point sets.
    
    Args:
        pointsr: Source points (N x 2 or 3).
        pointsl: Target points (N x 2 or 3).
        pointsl_all: All source points (unused, for API compatibility).
        pointsr_all: All target points (unused, for API compatibility).
        
    Returns:
        Tuple of:
            - 3x3 homogeneous transformation matrix
            - Transformation parameters [tx, ty, θ]
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
