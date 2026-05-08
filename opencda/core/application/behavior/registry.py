"""Registry for discoverable behavior service classes."""

import inspect
import logging
from typing import Any, TypeVar

from .behavior_service_protocol import BehaviorService

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.registry")

BehaviorServiceT = TypeVar("BehaviorServiceT", bound=BehaviorService[Any, Any])


class BehaviorServiceRegistry:
    """
    Registry for behavior service classes.

    Services are keyed by ``service_type`` and registered manually
    or by using ``BehaviorServiceRegistry.register`` as a decorator.
    """

    _registry: dict[str, type[BehaviorService[Any, Any]]] = {}

    @classmethod
    def register(cls, service_cls: type[BehaviorServiceT]) -> type[BehaviorServiceT]:
        """Register a concrete behavior service class."""
        if inspect.isabstract(service_cls):
            raise ValueError(f"Cannot register abstract behavior service class '{service_cls.__name__}'.")

        if (service_type := getattr(service_cls, "service_type", None)) is None:
            raise ValueError(f"Behavior service class '{service_cls.__name__}' must define 'service_type'.")

        if service_type in cls._registry:
            raise ValueError(f"Duplicate behavior service registration for service='{service_type}'.")

        cls._registry[service_type] = service_cls
        logger.info("Registered behavior service class '%s' as '%s'.", service_cls.__name__, service_type)
        return service_cls

    @classmethod
    def get_service_class(cls, service_type: str) -> type[BehaviorService[Any, Any]]:
        """Return a behavior service class for the given service name."""
        if service_type not in cls._registry:
            available = cls.list_services()
            raise KeyError(f"Unknown behavior service '{service_type}'. Available: {available}")
        return cls._registry[service_type]

    @classmethod
    def create_service(cls, service_type: str, **kwargs: Any) -> BehaviorService[Any, Any]:
        """Instantiate a behavior service by name."""
        service_cls = cls.get_service_class(service_type=service_type)
        return service_cls(**kwargs)

    @classmethod
    def list_services(cls) -> list[str]:
        """List registered behavior services."""
        return sorted(list(cls._registry))
