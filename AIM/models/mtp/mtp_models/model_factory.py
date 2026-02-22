import torch
import yaml
import importlib
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional, Any

from .model_factory_config import (
    MODEL_LIST_PATH,
    FACTORY_YAML_MODELS_FIELD,
    FACTORY_YAML_CLASS_FIELD,
    FACTORY_YAML_MODULE_FIELD,
)


class IncorrectModelName(Exception):
    """
    exception raised when model name is incorrect
    """

    pass


class ModelHolder(torch.nn.Module, PyTorchModelHubMixin):
    """
    holder class for multiple models that can be pushed to huggingface hub
    """

    def __init__(self, models: list[torch.nn.Module]) -> None:
        """
        initialize model holder

        :param models: list of model instances
        """
        super().__init__()
        self.models_list = torch.nn.ModuleList(models)

    def forward(self, *args: Any) -> Any:
        """
        forward pass through all models sequentially

        :param args: input arguments for models

        :return: output from last model
        """
        for model in self.models_list:
            x = model(*args)

        return x


class ModelFactory:
    """
    factory class for creating models from yaml configuration files
    """

    try:
        with open(MODEL_LIST_PATH) as model_list_file:
            model_list_yaml = yaml.safe_load(model_list_file)
            model_list_config = model_list_yaml[FACTORY_YAML_MODELS_FIELD]
            model_list = list(model_list_config.keys())

    except Exception as error:
        print(error)

    @classmethod
    def get_model_list(cls) -> list[str]:
        """
        get list of available model names

        :return: list of model names
        """
        return cls.model_list

    @classmethod
    def create_model(cls, yaml_config_file_path: str) -> Optional[ModelHolder]:
        """
        create model instance from yaml configuration file

        :param yaml_config_file_path: path to yaml configuration file

        :return: model holder instance or None on error
        """
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
                model_class = getattr(module, class_name)

                models.append(model_class(**yaml_config[model_key]))

            return ModelHolder(models)
        except Exception as error:
            print(error)
            return None

    @classmethod
    def get_model_info(cls, yaml_config_file_path: str) -> Optional[dict[str, Any]]:
        """
        get model information from yaml configuration file

        :param yaml_config_file_path: path to yaml configuration file

        :return: dictionary with model information or None on error
        """
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
