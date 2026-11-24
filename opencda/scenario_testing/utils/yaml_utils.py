"""
Used to load and write yaml files
"""

from __future__ import annotations

import re
from datetime import datetime
from os import PathLike
from typing import Any, TypeAlias, cast

import yaml
from omegaconf import OmegaConf

YamlDict: TypeAlias = dict[str, Any]


class _SafeConfigLoader(yaml.SafeLoader):
    """Read data-only config files with OpenCDA's scientific notation support."""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._checked_mappings: set[yaml.nodes.MappingNode] = set()

    def flatten_mapping(self, node: yaml.nodes.MappingNode) -> None:
        # Check the explicit keys before expanding YAML merges, which allow overrides.
        if node not in self._checked_mappings:
            self._checked_mappings.add(node)
            seen: set[Any] = set()
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    continue
                key = key_node.value if key_node.tag == "tag:yaml.org,2002:value" else self.construct_object(key_node)
                try:
                    duplicate = key in seen
                    seen.add(key)
                except TypeError:
                    continue  # The YAML constructor reports invalid mapping keys.
                if duplicate:
                    raise yaml.constructor.ConstructorError("while reading a mapping", node.start_mark, f"duplicate key {key!r}", key_node.start_mark)
        super().flatten_mapping(node)


def _add_float_resolver(loader: type[yaml.Loader] | type[yaml.SafeLoader]) -> None:
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


_add_float_resolver(_SafeConfigLoader)


def read_config_mapping(file: str | PathLike[str]) -> YamlDict:
    """Read a data-only YAML definition without adding scenario metadata."""
    with open(file, "r", encoding="utf-8") as stream:
        try:
            param = yaml.load(stream, Loader=_SafeConfigLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"{file}: {exc}") from exc
    if not isinstance(param, dict):
        raise ValueError(f"{file}: expected a top-level mapping.")
    return param


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
    _add_float_resolver(loader)
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
