"""Builtin attack stage modules."""

from __future__ import annotations

from importlib import import_module
import logging
import pkgutil

logger = logging.getLogger("cavise.opencda.opencda.core.attack.adversary_framework.stages.__init__")


def _import_builtin_attack_stages() -> None:
    """Import all builtin attack stage modules so they self-register."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        import_module(f"{__name__}.{module_info.name}")
        logger.info("Imported builtin attack stage module '%s'.", module_info.name)


_import_builtin_attack_stages()
