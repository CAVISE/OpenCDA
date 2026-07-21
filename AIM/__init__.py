"""
AIM module public API.

Provides:
- AIMModel base class
- get_model factory function
- list_models utility
"""

import importlib
from .aim_model import AIMModel
from .registry import ModelRegistry

# Initialize model discovery
importlib.import_module("AIM.models")

list_models = ModelRegistry.list_models
get_model = ModelRegistry.get_model

__all__ = ["AIMModel", "list_models", "get_model"]
