import json
from functools import cache

import numpy as np


@cache
def load_path_to_intention_config(
    intention_config_path: str,
) -> dict:
    try:
        with open(intention_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise Exception(f"Файл '{intention_config_path}' не найден.")


@cache
def get_path_to_intention(
    intention_config_path: str,
) -> dict:
    config = load_path_to_intention_config(intention_config_path)
    if "paths" not in config:
        raise KeyError(f'There is no "paths" parameter in {intention_config_path}. Please specify it')
    return config["paths"]


@cache
def get_center_coodinates(
    intention_config_path: str,
) -> dict:
    config = load_path_to_intention_config(intention_config_path)
    if "center_coordinates" not in config:
        raise KeyError(
            f'There is no "center_coordinates" parameter in {intention_config_path}. Please specify it like this "center_coordinates": {{"x": 631.73, "y": 597.22}}'
        )
    return config["center_coordinates"]


def get_intention_from_vehicle_id(vehicle_id: str, intention_config_path: str) -> np.ndarray:
    """
    Parse the vehicle id to distinguish its intention.
    """
    path_to_intention = get_path_to_intention(intention_config_path)

    intention = np.zeros(4)

    from_path, to_path, _ = vehicle_id.split("_")
    intention_str = path_to_intention[f"{from_path}_{to_path}"]

    if intention_str == "left":
        intention[0] = 1
    elif intention_str == "straight":
        intention[1] = 1
    elif intention_str == "right":
        intention[2] = 1
    else:
        raise Exception(f'There is no "{from_path}_{to_path}" vehicle id in config')

    return intention
