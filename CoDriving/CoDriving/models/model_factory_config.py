from pathlib import Path
import os

CURRENT_DIR = Path(__file__).parent
MODEL_LIST_PATH = os.path.join(CURRENT_DIR, "model_list.yaml")

FACTORY_YAML_MODELS_FIELD = "models"
FACTORY_YAML_DIR_FIELD = "dir"
FACTORY_YAML_MODULE_FIELD = "module"
FACTORY_YAML_CLASS_FIELD = "class"
FACTORY_YAML_BASECONFIG_FIELD = "base_config"
