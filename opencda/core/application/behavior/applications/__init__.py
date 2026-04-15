"""Builtin behavior application modules."""

from importlib import import_module
import logging
import pkgutil

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.applications.__init__")


def _import_builtin_behavior_applications() -> None:
    """Import all builtin behavior application modules so they self-register."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        import_module(f"{__name__}.{module_info.name}")

        logger.info("Imported builtin behavior application module '%s'.", module_info.name)


_import_builtin_behavior_applications()
