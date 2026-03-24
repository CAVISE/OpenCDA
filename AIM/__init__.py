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

get_model = ModelRegistry.get_model
list_models = ModelRegistry.list_models

__all__ = ["AIMModel", "get_model", "list_models"]
