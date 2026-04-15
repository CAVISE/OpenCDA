"""
Behavior services public API.

Provides:
- BehaviorService protocol
- BehaviorServiceRegistry runtime registry
- get_service_class lookup function
- create_service factory function
- list_services utility
"""

import importlib

from .behavior_service_protocol import BehaviorService
from .registry import BehaviorServiceRegistry

# Initialize builtin behavior service discovery.
importlib.import_module("opencda.core.application.behavior.services")

get_service_class = BehaviorServiceRegistry.get_service_class
create_service = BehaviorServiceRegistry.create_service
list_services = BehaviorServiceRegistry.list_services

__all__ = [
    "BehaviorService",
    "BehaviorServiceRegistry",
    "create_service",
    "get_service_class",
    "list_services",
]
