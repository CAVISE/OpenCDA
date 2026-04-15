"""Registry for discoverable behavior service classes."""

import inspect
import logging
from typing import Any

from .behavior_service_protocol import BehaviorService, BehaviorServiceMessageT, BehaviorServiceResultT

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.registry")


class BehaviorServiceRegistry:
    """
    Registry for behavior service classes.

    Services are keyed by ``service_name`` and registered manually
    or by using ``BehaviorServiceRegistry.register`` as a decorator.
    """

    _registry: dict[str, type[BehaviorService[BehaviorServiceMessageT, BehaviorServiceResultT]]] = {}

    @classmethod
    def register(cls, service_cls: type[BehaviorService[Any, Any]]) -> type[BehaviorService[Any, Any]]:
        """Register a concrete behavior service class."""
        if inspect.isabstract(service_cls):
            raise ValueError(f"Cannot register abstract behavior service class '{service_cls.__name__}'.")

        if (service_name := getattr(service_cls, "service_name", None)) is None:
            raise ValueError(f"Behavior service class '{service_cls.__name__}' must define 'service_name'.")

        if service_name in cls._registry:
            raise ValueError(f"Duplicate behavior service registration for service='{service_name}'.")

        cls._registry[service_name] = service_cls
        logger.info("Registered behavior service class '%s' as '%s'.", service_cls.__name__, service_name)
        return service_cls

    @classmethod
    def get_service_class(cls, service_name: str) -> type[BehaviorService[Any, Any]]:
        """Return a behavior service class for the given service name."""
        if service_name not in cls._registry:
            available = cls.list_services()
            raise KeyError(f"Unknown behavior service '{service_name}'. Available: {available}")
        return cls._registry[service_name]

    @classmethod
    def create_service(cls, service_name: str, **kwargs: Any) -> BehaviorService[Any, Any]:
        """Instantiate a behavior service by name."""
        service_cls = cls.get_service_class(service_name=service_name)
        return service_cls(**kwargs)

    @classmethod
    def list_services(cls) -> list[str]:
        """List registered behavior services."""
        return sorted(list(cls._registry))
