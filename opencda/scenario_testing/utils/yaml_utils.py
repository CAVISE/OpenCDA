"""
Used to load and write yaml files
"""

from __future__ import annotations

import re
from datetime import datetime
from os import PathLike
from typing import Any, TypeAlias, cast

import yaml  # type: ignore[import-untyped]
from omegaconf import OmegaConf

YamlDict: TypeAlias = dict[str, Any]


def load_yaml(file: str | PathLike[str]) -> YamlDict:
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
    with open(file, "r", encoding="utf-8") as stream:
        param = cast(YamlDict, yaml.load(stream, Loader=loader))

    # load current time for data dumping and evaluation
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    param["current_time"] = current_time

    return param


def add_current_time(params: YamlDict) -> tuple[YamlDict, str]:
    """
    Add current time to the params dictionary.
    """
    # load current time for data dumping and evaluation
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    params["current_time"] = current_time

    return params, current_time


def save_yaml(data: Any, save_name: str | PathLike[str]) -> None:
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
        with open(save_name, "w", encoding="utf-8") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
    else:
        with open(save_name, "w", encoding="utf-8") as f:
            OmegaConf.save(data, f)
