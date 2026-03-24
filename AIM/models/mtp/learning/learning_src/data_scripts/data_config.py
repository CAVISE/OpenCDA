import yaml
from pathlib import Path
from types import SimpleNamespace


def dict_to_namespace_data_config(d):
    ns = SimpleNamespace()

    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict_to_namespace_data_config(v))
        else:
            setattr(ns, k, v)
    return ns


_config_path = Path(__file__).with_suffix(".yaml")

with open(_config_path) as f:
    cfg_dict = yaml.safe_load(f)

config = dict_to_namespace_data_config(cfg_dict)
