import torch

from AIM.models.mtp.learning.learning_src.data_scripts.data_config import (
    NORMALIZE_DATA,
    SAMPLE_RATE,
    COLLECT_DATA_RADIUS,
)
from AIM.models.mtp.learning.learning_src.data_scripts.preprocess_utils import (
    transform_sumo2carla_yaw,
    transform_coords,
    normalize_yaw,
    denormalize_coords,
)


def my_get_yaw(
    start_cos, start_sin, last_cos, last_sin, pos: torch.Tensor, yaw_keys: torch.Tensor, yaw_vals: torch.Tensor, map_boundaries: torch.Tensor
):
    idxs = 10 * torch.atan2(start_sin, start_cos) + torch.atan2(last_sin, last_cos)
    yaw_keys_exp = yaw_keys.view(*([1] * idxs.dim()), *yaw_keys.shape)

    mask = torch.abs(idxs.unsqueeze(-1) - yaw_keys_exp) < 1e-3
    positions = mask.int().argmax(dim=-1)
    route = yaw_vals[positions]

    pos_copy = pos.clone()  # carla coords
    if NORMALIZE_DATA:
        denormalize_coords(pos_copy, COLLECT_DATA_RADIUS * map_boundaries)

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

    if NORMALIZE_DATA:
        yaws_rad_carla = normalize_yaw(yaws_rad_carla)
    return yaws_rad_carla.float()


def my_get_speed(dx: torch.Tensor, dy: torch.Tensor):
    speed = (dx**2 + dy**2) ** 0.5 * SAMPLE_RATE
    # vehicle_max_speed = VEHICLE_MAX_SPEED if not NORMALIZE_DATA else normalize_speed(VEHICLE_MAX_SPEED, VEHICLE_MAX_SPEED)
    # mask = (speed > vehicle_max_speed)
    # speed[mask] = vehicle_max_speed
    return speed
