import os
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parent
DATA_PATH = os.path.join(MAIN_PATH, "data")
SUMO_GENED_MAPS_PATH = os.path.join(DATA_PATH, "sumo_gened_maps")
DATA_VIZUALIZATION_PATH = os.path.join(DATA_PATH, "vizualization")
EXPIREMENTS_PATH = os.path.join(MAIN_PATH, "experements")

EXPIREMENTS_CONFIG_PATH = os.path.join(MAIN_PATH, "experements_configs")
EXPIREMENTS_TRAIN_CONFIG_PATH = os.path.join(EXPIREMENTS_CONFIG_PATH, "train_configs")
EXPIREMENTS_MODELS_CONFIG_PATH = os.path.join(EXPIREMENTS_CONFIG_PATH, "model_configs")

BASE_TRAIN_CONFIG_PATH = os.path.join(EXPIREMENTS_CONFIG_PATH, "train_config.yaml")
BASE_MODEL_CONFIG_PATH = os.path.join(EXPIREMENTS_CONFIG_PATH, "model_config.yaml")
LOGS_DIR_NAME = "logs"

Y_X_DISTR_FILE = "y_x_distr_file.pkl"
Y_Y_DISTR_FILE = "y_y_distr_file.pkl"

START_POSITIONS_FILE = "start_positions.csv"
LAST_POSITIONS_FILE = "last_positions.csv"

YAW_DICT_PATH = os.path.join(MAIN_PATH, "../../../../opencda/assets/yaw_dict_10m.pkl")
