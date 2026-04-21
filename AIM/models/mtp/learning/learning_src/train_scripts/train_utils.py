import torch

from AIM.models.mtp.learning.learning_src.data_scripts.data_config import config
from AIM.models.mtp.learning.learning_src.data_scripts.preprocess_utils import (
    transform_sumo2carla_yaw,
    transform_coords,
    normalize_yaw,
    denormalize_coords,
)


def my_get_yaw(
    start_cos: torch.Tensor,
    start_sin: torch.Tensor,
    last_cos: torch.Tensor,
    last_sin: torch.Tensor,
    pos: torch.Tensor,
    yaw_keys: torch.Tensor,
    yaw_vals: torch.Tensor,
    map_boundaries: torch.Tensor,
) -> torch.Tensor:
    """
    get yaw angle based on position and route information

    :param start_cos: cosine of start yaw angle
    :param start_sin: sine of start yaw angle
    :param last_cos: cosine of last yaw angle
    :param last_sin: sine of last yaw angle
    :param pos: position tensor in carla coordinates
    :param yaw_keys: yaw dictionary keys
    :param yaw_vals: yaw dictionary values (route information)
    :param map_boundaries: map boundary values

    :return: yaw angles in normalized carla coordinate system
    """
    idxs = 10 * torch.atan2(start_sin, start_cos) + torch.atan2(last_sin, last_cos)
    yaw_keys_exp = yaw_keys.view(*([1] * idxs.dim()), *yaw_keys.shape)

    mask = torch.abs(idxs.unsqueeze(-1) - yaw_keys_exp) < 1e-5
    positions = mask.int().argmax(dim=-1)
    route = yaw_vals[positions]

    pos_copy = pos.clone()  # carla coords
    if config.data_processing.normalize_data:
        denormalize_coords(pos_copy, map_boundaries)

    transform_coords(pos_copy)  # sumo coords
    deltas = pos_copy.unsqueeze(-2) - route.float()[..., :-1]
    dists = torch.norm(deltas, dim=-1)

    min_idx = torch.argmin(dists, dim=-1)

    min_mask = torch.zeros_like(route[..., 0], dtype=torch.bool)
    min_mask.scatter_(-1, min_idx.unsqueeze(-1), True)
    min_mask_exp = min_mask.unsqueeze(-1).expand_as(route)

    yaws_deg = (route * min_mask_exp).sum(dim=-2)[..., -1]
    yaws_rad = torch.deg2rad(yaws_deg)
    yaws_rad_carla = transform_sumo2carla_yaw(yaws_rad)

    if config.data_processing.normalize_data:
        yaws_rad_carla = normalize_yaw(yaws_rad_carla)
    return yaws_rad_carla.float()


def my_get_yaw_new(
    start_cos: torch.Tensor,
    start_sin: torch.Tensor,
    last_cos: torch.Tensor,
    last_sin: torch.Tensor,
    pos: torch.Tensor,
    map_lane_reprs: torch.Tensor,
    map_lane_repr_masks: torch.Tensor,
) -> torch.Tensor:
    """
    get yaw angle based on position and route information

    :param start_cos: cosine of start yaw angle
    :param start_sin: sine of start yaw angle
    :param last_cos: cosine of last yaw angle
    :param last_sin: sine of last yaw angle
    :param pos: position tensor in carla coordinates

    :return: yaw angles in normalized carla coordinate system
    """
    start_cos_diff = start_cos.unsqueeze(-1) - map_lane_reprs[..., 0, 2].unsqueeze(1)
    start_sin_diff = start_sin.unsqueeze(-1) - map_lane_reprs[..., 0, 3].unsqueeze(1)

    last_cos_diff = last_cos.unsqueeze(-1) - map_lane_reprs[..., -1, 2].unsqueeze(1)
    last_sin_diff = last_sin.unsqueeze(-1) - map_lane_reprs[..., -1, 3].unsqueeze(1)
    yaw_diff = torch.abs(start_cos_diff) + torch.abs(start_sin_diff) + torch.abs(last_cos_diff) + torch.abs(last_sin_diff)
    INF = 1e6
    YAW_FIT_EPS = 5e-2
    lane_mask = map_lane_repr_masks[..., 0].unsqueeze(1).expand(-1, yaw_diff.shape[1], -1)
    yaw_diff[~lane_mask] = INF

    yaw_fit_mask = yaw_diff < YAW_FIT_EPS
    has_fit = yaw_fit_mask.any(dim=-1, keepdim=True)
    lane_select_mask = torch.where(has_fit, yaw_fit_mask, lane_mask)

    pos_diff = ((pos.unsqueeze(-2).unsqueeze(-2) - map_lane_reprs[..., [0, 1]].unsqueeze(1)) ** 2).sum(dim=-1)
    pos_diff[~lane_select_mask.unsqueeze(-1).expand_as(pos_diff)] = INF
    pos_diff_view = pos_diff.reshape((pos_diff.shape[0], pos_diff.shape[1], pos_diff.shape[2] * pos_diff.shape[3]))
    _, pos_diff_view_inds = pos_diff_view.min(dim=-1)

    map_lane_reprs_view = map_lane_reprs.view(map_lane_reprs.shape[0], map_lane_reprs.shape[1] * map_lane_reprs.shape[2], map_lane_reprs.shape[3])
    map_lane_reprs_gathered = torch.gather(
        map_lane_reprs_view, dim=1, index=pos_diff_view_inds.unsqueeze(-1).expand(-1, -1, config.object_map.object_vector_size + 2)
    )
    yaws_rad_carla = torch.atan2(map_lane_reprs_gathered[..., -1], map_lane_reprs_gathered[..., -2])

    if config.data_processing.normalize_data:
        yaws_rad_carla = normalize_yaw(yaws_rad_carla)

    return yaws_rad_carla.float()


def my_get_speed(dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """
    calculate speed from coordinate deltas

    :param dx: x coordinate delta
    :param dy: y coordinate delta

    :return: speed values
    """
    speed = (dx**2 + dy**2) ** 0.5 * config.temporal.sample_rate
    return speed
