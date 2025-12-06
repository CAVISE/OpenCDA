import torch
import yaml
import importlib
import os

from model_factory_config import *


class IncorrectModelName(Exception):
  pass


class ModelHolder(torch.nn.Module):
  def __init__(self, models):
    super().__init__()
    self.models_list = torch.nn.ModuleList(models)

  def forward(self, x):
    for model in self.models_list:
      x = model(x)

    return x


class ModelFactory:
  def __init__(self):
    try:
      with open(MODEL_LIST_PATH) as model_list_file:
        model_list_yaml = yaml.safe_load(model_list_file)
        self.model_list_config = model_list_yaml[FACTORY_YAML_MODELS_FIELD]
        self.model_list = list(self.model_list_config.keys())

    except Exception as error:
      print(error)

  def get_model_list(self):
    return self.model_list

  def create_model(self, yaml_config_str: str):
    try:
      yaml_config = yaml.safe_load(yaml_config_str)
      models = []

      for model_key in yaml_config.keys():
        if not model_key in self.model_list:
          raise IncorrectModelName()

        class_name = self.model_list_config[model_key][FACTORY_YAML_CLASS_FIELD]
        module_name = self.model_list_config[model_key][FACTORY_YAML_MODULE_FIELD]
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        models.append(cls(**yaml_config[model_key]))

      return ModelHolder(models)
    except Exception as error:
      print(error)


if __name__ == '__main__':
  factory = ModelFactory()
  models = factory.get_model_list()

  with open(os.path.join(CURRENT_DIR, 'all_models/GNN_mtl_gnn/GNN_mtl_gnn.yaml')) as models_file:
    model = factory.create_model(models_file)
    print(model)
