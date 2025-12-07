import torch
import yaml
import importlib
import os
from huggingface_hub import PyTorchModelHubMixin

from .model_factory_config import *


class IncorrectModelName(Exception):
  pass


class ModelHolder(torch.nn.Module, PyTorchModelHubMixin):
  def __init__(self, models):
    super().__init__()
    self.models_list = torch.nn.ModuleList(models)

  def forward(self, x, edge_index):
    for model in self.models_list:
      x = model(x, edge_index)

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
        if not model_key in cls.model_list:
          raise IncorrectModelName()

        class_name = cls.model_list_config[model_key][FACTORY_YAML_CLASS_FIELD]
        module_name = cls.model_list_config[model_key][FACTORY_YAML_MODULE_FIELD]
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        models.append(cls(**yaml_config[model_key]))

      return ModelHolder(models)
    except Exception as error:
      print(error)


if __name__ == '__main__':
  factory = ModelFactory()
  models = factory.get_model_list()

  model_config_path = os.path.join(CURRENT_DIR, 'all_models/GNN_mtl_gnn/GNN_mtl_gnn.yaml')
  model = factory.create_model(model_config_path)
  print(model)
