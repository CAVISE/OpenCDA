import importlib
from .aim_model import AIMModel
from .registry import ModelRegistry

# init models
importlib.import_module("AIM.models")

get_model = ModelRegistry.get_model
list_models = ModelRegistry.list_models

__all__ = ["AIMModel", "get_model", "list_models"]
