import torch
import yaml
import importlib
import os
from huggingface_hub import PyTorchModelHubMixin

from .model_factory_config import (
    MODEL_LIST_PATH,
    FACTORY_YAML_MODELS_FIELD,
    FACTORY_YAML_CLASS_FIELD,
    FACTORY_YAML_MODULE_FIELD,
    CURRENT_DIR,
)


class IncorrectModelName(Exception):
    pass


class ModelHolder(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, models):
        super().__init__()
        self.models_list = torch.nn.ModuleList(models)

    def forward(self, *args):
        for model in self.models_list:
            x = model(*args)

        return x


class ModelFactory:
    try:
        with open(MODEL_LIST_PATH) as model_list_file:
            model_list_yaml = yaml.safe_load(model_list_file)
            model_list_config = model_list_yaml[FACTORY_YAML_MODELS_FIELD]
            model_list = list(model_list_config.keys())

    except Exception as error:
        print(error)

    @classmethod
    def get_model_list(cls):
        return cls.model_list

    @classmethod
    def create_model(cls, yaml_config_file_path):
        try:
            with open(yaml_config_file_path) as models_file:
                yaml_config = yaml.safe_load(models_file)
            models = []

            for model_key in yaml_config.keys():
                if model_key not in cls.model_list:
                    raise IncorrectModelName()

                class_name = cls.model_list_config[model_key][FACTORY_YAML_CLASS_FIELD]
                module_name = cls.model_list_config[model_key][FACTORY_YAML_MODULE_FIELD]
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)

                models.append(cls(**yaml_config[model_key]))

            return ModelHolder(models)
        except Exception as error:
            print(error)

    @classmethod
    def get_model_info(cls, yaml_config_file_path):
        try:
            with open(yaml_config_file_path) as models_file:
                yaml_config = yaml.safe_load(models_file)

            for model_key in yaml_config.keys():
                if model_key not in cls.model_list:
                    raise IncorrectModelName()

            return cls.model_list_config[model_key]
        except Exception as error:
            print(error)
            return None
