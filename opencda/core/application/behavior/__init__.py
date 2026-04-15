"""
Behavior applications public API.

Provides:
- BehaviorApplication protocol
- BehaviorApplicationRegistry runtime registry
- get_application_class lookup function
- create_application factory function
- list_applications utility
"""

import importlib

from .behavior_application_protocol import BehaviorApplication
from .registry import BehaviorApplicationRegistry

# Initialize builtin behavior application discovery.
importlib.import_module("opencda.core.application.behavior.applications")

get_application_class = BehaviorApplicationRegistry.get_application_class
create_application = BehaviorApplicationRegistry.create_application
list_applications = BehaviorApplicationRegistry.list_applications

__all__ = [
    "BehaviorApplication",
    "BehaviorApplicationRegistry",
    "create_application",
    "get_application_class",
    "list_applications",
]
