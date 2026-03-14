import importlib
from .aim_model import AIMModelWrapper, AIMModel
from .registry import ModelRegistry

# init models
importlib.import_module("AIM.models")

get_model_wrapper = ModelRegistry.get_model_wrapper
list_model_wrappers = ModelRegistry.list_model_wrappers
list_models = ModelRegistry.list_models

__all__ = ["AIMModelWrapper", "AIMModel", "get_model_wrapper", "list_models", "list_model_wrappers"]
