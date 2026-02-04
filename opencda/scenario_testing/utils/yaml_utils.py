"""
Used to load and write yaml files
"""

import re
from typing import Any, Dict, Tuple
import yaml
from datetime import datetime
from omegaconf import OmegaConf


def load_yaml(file: str) -> Dict[str, Any]:
    """
    Load yaml file and return a dictionary.
    Parameters
    ----------
    file : string
        yaml file path.

    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """

    stream = open(file, "r")
    loader = yaml.Loader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    param = yaml.load(stream, Loader=loader)

    # load current time for data dumping and evaluation
    current_time = datetime.now()
    current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")  # NOTE: current_time changes type from datetime to str

    param["current_time"] = current_time

    return param


def add_current_time(params: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Add current time to the params dictionary.
    """
    # load current time for data dumping and evaluation
    current_time = datetime.now()
    current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")  # NOTE: current_time changes type from datetime to str

    params["current_time"] = current_time

    return (
        params,
        current_time,
    )  # NOTE [mypy] Incompatible return type (got "tuple[dict[str, Any], datetime]", expected "tuple[dict[str, Any], str]").Caused by type mutation of 'current_time'


def save_yaml(data: Any, save_name: str) -> None:
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """
    if isinstance(data, dict):
        with open(save_name, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
    else:
        with open(save_name, "w") as f:
            OmegaConf.save(data, f)
