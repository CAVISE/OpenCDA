"""Builtin attack modules."""

from __future__ import annotations

from importlib import import_module
import logging
import pkgutil

logger = logging.getLogger("cavise.opencda.opencda.core.attack.adversary_framework.attacks.__init__")


def _import_builtin_attacks() -> None:
    """Import all builtin attack modules so they self-register."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        import_module(f"{__name__}.{module_info.name}")
        logger.info("Imported builtin attack module '%s'.", module_info.name)


_import_builtin_attacks()
