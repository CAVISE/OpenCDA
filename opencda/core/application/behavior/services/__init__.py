"""Builtin behavior service modules."""

from importlib import import_module
import logging
import pkgutil

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.__init__")


def _import_builtin_behavior_services() -> None:
    """Import all builtin behavior service modules so they self-register."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        import_module(f"{__name__}.{module_info.name}")

        logger.info("Imported builtin behavior service module '%s'.", module_info.name)


_import_builtin_behavior_services()
