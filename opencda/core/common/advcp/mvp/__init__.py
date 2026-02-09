"""
MVP (Minimum Viable Product) module for AdvCP.
This module contains attack, defense, perception, and data processing components.
"""

import logging

logger = logging.getLogger("cavise.mvp_init")
logger.debug("MVP module initialized")

# Debug: Check if subdirectories are accessible
try:
    from . import attack

    logger.debug("MVP attack submodule imported successfully")
except ImportError as e:
    logger.error(f"Failed to import MVP attack submodule: {e}")

try:
    from . import defense

    logger.debug("MVP defense submodule imported successfully")
except ImportError as e:
    logger.error(f"Failed to import MVP defense submodule: {e}")

try:
    from . import perception

    logger.debug("MVP perception submodule imported successfully")
except ImportError as e:
    logger.error(f"Failed to import MVP perception submodule: {e}")

try:
    from . import data

    logger.debug("MVP data submodule imported successfully")
except ImportError as e:
    logger.error(f"Failed to import MVP data submodule: {e}")

try:
    from . import tools

    logger.debug("MVP tools submodule imported successfully")
except ImportError as e:
    logger.error(f"Failed to import MVP tools submodule: {e}")
