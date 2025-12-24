import os
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parent
DATA_PATH = os.path.join(MAIN_PATH, "data")
DATA_VIZUALIZATION_PATH = os.path.join(DATA_PATH, "vizualization")
EXPIREMENTS_PATH = os.path.join(MAIN_PATH, "experements")

EXPIREMENTS_CONFIG_PATH = os.path.join(MAIN_PATH, "experements_configs")
EXPIREMENTS_TRAIN_CONFIG_PATH = os.path.join(EXPIREMENTS_CONFIG_PATH, "train_configs")
EXPIREMENTS_MODELS_CONFIG_PATH = os.path.join(EXPIREMENTS_CONFIG_PATH, "model_configs")

BASE_TRAIN_CONFIG_PATH = os.path.join(EXPIREMENTS_CONFIG_PATH, "train_config.yaml")
BASE_MODEL_CONFIG_PATH = os.path.join(EXPIREMENTS_CONFIG_PATH, "model_config.yaml")
LOGS_DIR_NAME = "logs"
