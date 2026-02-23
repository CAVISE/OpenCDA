"""
Automatic model package discovery.

Dynamically imports all subpackages inside AIM.models
to trigger model registration.
"""

import pkgutil
import importlib
from typing import None


def _discover_model_packages() -> None:
    """
    Discover and import all model subpackages.
    """
    for info in pkgutil.iter_modules(__path__, prefix="AIM.models."):
        if not info.ispkg:
            continue
        importlib.import_module(info.name)


_discover_model_packages()
