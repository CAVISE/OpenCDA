import os

from data_config import EXPIREMENTS_PATH, DATA_PATH
import add_path  # noqa: F401
from CoDriving.train_scripts.train_one_config import train_one_config


def train():
    train_config_path = os.path.join(EXPIREMENTS_PATH, "test", "train_config.yaml")
    model_config_path = os.path.join(EXPIREMENTS_PATH, "test", "model_config.yaml")
    train_one_config(train_config_path, model_config_path, EXPIREMENTS_PATH, DATA_PATH)


if __name__ == "__main__":
    train()
