"""
AdvCollaborativePerception (AdvCP) module for OpenCDA.
This module integrates attack and defense mechanisms for collaborative perception in V2X networks.
"""

from .advcp_manager import AdvCPManager
from .attack_handler import AttackHandler
from .defense_handler import DefenseHandler
from .data_simulator import DataSimulator
from .metrics_logger import MetricsLogger
from .visualization import Visualization
from .utils import AdvCPUtils