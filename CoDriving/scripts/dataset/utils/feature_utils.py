import json
from functools import cache
from typing import Any, Dict
import numpy.typing as npt

import numpy as np


@cache
def load_path_to_intention_config(
    intention_config_path: str,
) -> Dict[str, Any]:
    """
    Load intentions configuration JSON and cache the result.

    Parameters
    ----------
    intention_config_path : str
        Path to the intentions JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object as a Python mapping.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        Not raised here; validation is done in helper functions.
    """
    try:
        with open(intention_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise Exception(f"Файл '{intention_config_path}' не найден.")


@cache
def get_path_to_intention(
    intention_config_path: str,
) -> Dict[str, str]:
    """
    Get mapping from route key ("from_to") to intention label.

    Parameters
    ----------
    intention_config_path : str
        Path to the intentions JSON file.

    Returns
    -------
    dict[str, str]
        Mapping like {"E4_E2": "left", ...}.

    Raises
    ------
    KeyError
        If "paths" key is missing.
    TypeError
        If "paths" is not a dict[str, str].
    """
    config = load_path_to_intention_config(intention_config_path)
    if "paths" not in config:
        raise KeyError(f'There is no "paths" parameter in {intention_config_path}. Please specify it')
    return config["paths"]


@cache
def get_center_coodinates(
    intention_config_path: str,
) -> Dict[str, float]:
    """
    Get center coordinates used for filtering/scene selection.

    Parameters
    ----------
    intention_config_path : str
        Path to the intentions JSON file.

    Returns
    -------
    dict[str, float]
        Dictionary with keys "x" and "y".

    Raises
    ------
    KeyError
        If "center_coordinates" key is missing.
    TypeError
        If the value is not a dict with numeric "x" and "y".
    """
    config = load_path_to_intention_config(intention_config_path)
    if "center_coordinates" not in config:
        raise KeyError(
            f'There is no "center_coordinates" parameter in {intention_config_path}. Please specify it like this "center_coordinates": {{"x": 631.73, "y": 597.22}}'
        )
    return config["center_coordinates"]


def get_intention_from_vehicle_id(vehicle_id: str, intention_config_path: str) -> npt.NDArray[np.float64]:
    """
    Parse a vehicle id and return its one-hot intention vector.

    Parameters
    ----------
    vehicle_id : str
        Vehicle id encoded as "<from_edge>_<to_edge>_<...>".
    intention_config_path : str
        Path to the intentions JSON file.

    Returns
    -------
    numpy.typing.NDArray[numpy.float64]
        One-hot intention vector of shape (4,):
        [left, straight, right, reserved].
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
