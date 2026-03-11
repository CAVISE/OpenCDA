"""
AdvCollaborativePerception (AdvCP) module for OpenCDA.
This module integrates attack and defense mechanisms for collaborative perception in V2X networks.
"""

import os
import sys
from typing import Any

# Add the mvp package directory to sys.path to enable imports like 'from mvp.xxx'
# This is necessary because mvp is a subdirectory of this package but needs to be
# importable as a top-level module.
_advcp_dir = os.path.dirname(os.path.abspath(__file__))
if _advcp_dir not in sys.path:
    sys.path.insert(0, _advcp_dir)

# Lazy imports to avoid circular dependencies and missing modules
# Import directly when needed:
#   from opencda.core.common.advcp.advcp_manager import AdvCPManager
#   from opencda.core.common.advcp.advcp_visualization_manager import AdvCPVisualizationManager

__all__ = ["AdvCPManager", "AdvCPVisualizationManager"]


def __getattr__(name: str) -> Any:
    """Lazy import for AdvCP module components."""
    if name == "AdvCPManager":
        from .advcp_manager import AdvCPManager

        return AdvCPManager
    elif name == "AdvCPVisualizationManager":
        from .advcp_visualization_manager import AdvCPVisualizationManager

        return AdvCPVisualizationManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
