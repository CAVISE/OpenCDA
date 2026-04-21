"""
AIM module public API.

Provides:
- AIMModel base class
- get_model factory function
- list_models utility
"""

import importlib
from .aim_model import AIMModelWrapper, AIMModel
from .registry import ModelRegistry

# Initialize model discovery
importlib.import_module("AIM.models")

get_model_wrapper = ModelRegistry.get_model_wrapper
list_model_wrappers = ModelRegistry.list_model_wrappers
list_models = ModelRegistry.list_models

__all__ = ["AIMModelWrapper", "AIMModel", "get_model_wrapper", "list_models", "list_model_wrappers"]
