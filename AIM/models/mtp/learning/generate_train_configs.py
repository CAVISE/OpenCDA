import os
import yaml
import copy
import shutil
from typing import Any

from .data_path_config import BASE_TRAIN_CONFIG_PATH, BASE_MODEL_CONFIG_PATH, EXPIREMENTS_TRAIN_CONFIG_PATH, EXPIREMENTS_MODELS_CONFIG_PATH


def read_base_config_file(config_file_path: str) -> dict[str, Any]:
    """
    read base config file from yaml

    :param config_file_path: path to yaml config file

    :return: dictionary with config parameters
    """
    try:
        with open(config_file_path) as config_file:
            dict_config = yaml.safe_load(config_file)

        return dict_config
    except Exception as error:
        print(error)


def write_config_file(config_file_path: str, dict_config: dict[str, Any]) -> None:
    """
    write config dictionary to yaml file

    :param config_file_path: path to output yaml file
    :param dict_config: dictionary with config parameters
    """
    try:
        with open(config_file_path, "w") as config_file:
            yaml.safe_dump(dict_config, config_file)

    except Exception as error:
        print(error)
        return None


def write_configs(
    base_config_path: str,
    config_params: dict[str, list[Any]],
    cofig_path: str,
    model_config: bool = False,
) -> None:
    """
    generate multiple config files from base config and parameter variations

    :param base_config_path: path to base config file
    :param config_params: dictionary mapping parameter names to lists of values
    :param cofig_path: base path for output config files (will be appended with index)
    :param model_config: flag if generating model configs (nested structure)
    """
    base_config = read_base_config_file(base_config_path)
    write_config_file(f"{cofig_path}0.yaml", base_config)

    ind = 1
    for key in config_params.keys():
        for param in config_params[key]:
            cur_config = copy.deepcopy(base_config)
            if not model_config:
                cur_config[key] = param
            else:
                model_key = list(cur_config.keys())
                cur_config[model_key[0]][key] = param

            write_config_file(f"{cofig_path}{ind}.yaml", cur_config)
            ind += 1


def generate_configs() -> None:
    """
    generate train and model config files for experiments
    """
    if os.path.exists(EXPIREMENTS_TRAIN_CONFIG_PATH):
        shutil.rmtree(EXPIREMENTS_TRAIN_CONFIG_PATH)

    if os.path.exists(EXPIREMENTS_MODELS_CONFIG_PATH):
        shutil.rmtree(EXPIREMENTS_MODELS_CONFIG_PATH)

    os.makedirs(EXPIREMENTS_TRAIN_CONFIG_PATH, exist_ok=True)
    os.makedirs(EXPIREMENTS_MODELS_CONFIG_PATH, exist_ok=True)

    train_config_params = {}
    # train_config_params = {
    #     "lr": [5e-5, 1e-4],
    #     "weight_decay": [1e-4, 5e-4],
    # }

    model_config_params = {}
    # model_config_params = {
    #     "cars_encoder_hidden_channels": [160, 256],
    #     "cross_attn_n_attn": [5, 7],
    # }

    write_configs(BASE_TRAIN_CONFIG_PATH, train_config_params, os.path.join(EXPIREMENTS_TRAIN_CONFIG_PATH, "train_config"))
    write_configs(BASE_MODEL_CONFIG_PATH, model_config_params, os.path.join(EXPIREMENTS_MODELS_CONFIG_PATH, "model_config"), model_config=True)


if __name__ == "__main__":
    generate_configs()
