"""Builtin metric modules."""

from importlib import import_module
import logging
import pkgutil

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.metrics.__init__")


def _import_builtin_metrics() -> None:
    """Import all builtin metric modules so they self-register."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        import_module(f"{__name__}.{module_info.name}")

        logger.info("Imported builtin metric module '%s'.", module_info.name)


_import_builtin_metrics()
