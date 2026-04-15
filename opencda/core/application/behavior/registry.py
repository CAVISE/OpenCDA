"""Registry for discoverable behavior application classes."""

import inspect
import logging
from typing import Any

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.registry")


class BehaviorApplicationRegistry:
    """
    Registry for behavior application classes.

    Applications are keyed by ``application_name`` and registered manually
    or by using ``BehaviorApplicationRegistry.register`` as a decorator.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, application_cls: type) -> type:
        """Register a concrete behavior application class."""
        if inspect.isabstract(application_cls):
            raise ValueError(f"Cannot register abstract behavior application class '{application_cls.__name__}'.")

        if (application_name := getattr(application_cls, "application_name", None)) is None:
            raise ValueError(f"Behavior application class '{application_cls.__name__}' must define 'application_name'.")

        if application_name in cls._registry:
            raise ValueError(f"Duplicate behavior application registration for application='{application_name}'.")

        cls._registry[application_name] = application_cls
        logger.info("Registered behavior application class '%s' as '%s'.", application_cls.__name__, application_name)
        return application_cls

    @classmethod
    def get_application_class(cls, application_name: str) -> type:
        """Return a behavior application class for the given application name."""
        if application_name not in cls._registry:
            available = cls.list_applications()
            raise KeyError(f"Unknown behavior application '{application_name}'. Available: {available}")
        return cls._registry[application_name]

    @classmethod
    def create_application(cls, application_name: str, **kwargs: Any) -> Any:
        """Instantiate a behavior application by name."""
        application_cls = cls.get_application_class(application_name=application_name)
        return application_cls(**kwargs)

    @classmethod
    def list_applications(cls) -> list[str]:
        """List registered behavior applications."""
        return list(cls._registry)
