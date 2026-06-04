import os
import yaml
from pathlib import Path
from types import SimpleNamespace

_MAIN_PATH = Path(__file__).resolve().parent


def dict_to_namespace_data_path_config(d, base_path=_MAIN_PATH, paths_key=False):
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict_to_namespace_data_path_config(v, base_path, paths_key=(k == "paths")))
        else:
            if paths_key and isinstance(v, str):
                setattr(ns, k, os.path.join(base_path, v))
            else:
                setattr(ns, k, v)
    return ns


_config_path = Path(__file__).with_suffix(".yaml")

with open(_config_path, "r") as f:
    cfg_dict = yaml.safe_load(f)

path_config = dict_to_namespace_data_path_config(cfg_dict)
