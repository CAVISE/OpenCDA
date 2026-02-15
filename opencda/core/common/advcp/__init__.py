"""
AdvCollaborativePerception (AdvCP) module for OpenCDA.
This module integrates attack and defense mechanisms for collaborative perception in V2X networks.
"""

from typing import Any

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
