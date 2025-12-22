import os
import yaml
import numpy as np

from data_path_config import BASE_TRAIN_CONFIG_PATH, BASE_MODEL_CONFIG_PATH, EXPIREMENTS_TRAIN_CONFIG_PATH, EXPIREMENTS_MODELS_CONFIG_PATH
from CoDriving.data_scripts.utils.base_utils import del_files_in_dir


def read_base_config_file(config_file_path: str):
    try:
        with open(config_file_path) as config_file:
            dict_config = yaml.safe_load(config_file)

        return dict_config
    except Exception as error:
        print(error)


def write_config_file(config_file_path: str, dict_config):
    try:
        with open(config_file_path, "w") as config_file:
            yaml.safe_dump(dict_config, config_file)

    except Exception as error:
        print(error)


def write_configs(
    base_config_path: str,
    config_params: dict,
    cofig_path: str,
    model_config=False,
):
    base_config = read_base_config_file(base_config_path)

    ind = 0
    for key in config_params.keys():
        for param in config_params[key]:
            cur_config = base_config.copy()
            if not model_config:
                cur_config[key] = param
            else:
                model_key = list(cur_config.keys())
                cur_config[model_key[0]][key] = param

            write_config_file(f"{cofig_path}{ind}.yaml", cur_config)
            ind += 1


def generate_configs():
    os.makedirs(EXPIREMENTS_TRAIN_CONFIG_PATH, exist_ok=True)
    os.makedirs(EXPIREMENTS_MODELS_CONFIG_PATH, exist_ok=True)
    del_files_in_dir(EXPIREMENTS_TRAIN_CONFIG_PATH)
    del_files_in_dir(EXPIREMENTS_MODELS_CONFIG_PATH)

    train_config_params = {
        "lr": np.logspace(-5, -2, num=4, dtype=float).tolist(),
        "batch_size": np.logspace(8, 10, base=2, num=3, dtype=int).tolist(),
        "optimizer": ["adam", "adamw", "sgd"],
        "weight_decay": np.logspace(-5, -2, num=4, dtype=float).tolist(),
        "collision_penalty_factor": np.linspace(400, 1000, 7, dtype=float).tolist(),
        "step_weights_factor": np.linspace(5, 10, 5, dtype=float).tolist(),
        "dist_threshold": np.linspace(2, 5, 5, dtype=float).tolist(),
    }
    model_config_params = {"hidden_channels": np.logspace(7, 9, num=3, base=2, dtype=int).tolist()}

    write_configs(BASE_TRAIN_CONFIG_PATH, train_config_params, os.path.join(EXPIREMENTS_TRAIN_CONFIG_PATH, "train_config"))
    write_configs(BASE_MODEL_CONFIG_PATH, model_config_params, os.path.join(EXPIREMENTS_MODELS_CONFIG_PATH, "model_config"), model_config=True)


if __name__ == "__main__":
    generate_configs()
