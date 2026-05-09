"""
AdvCP: Adversarial Cooperative Perception.

Public re-exports for the AdvCP attack family. The implementation lives
in the sibling modules:

- ``attack_helper``: shared utilities (config validation, attacker
  resolution, target box construction, mesh helpers).
- ``early_fusion_attack``: rewrites the attacker LiDAR point cloud.
- ``intermediate_fusion_attack``: gradient-based perturbation of the
  attacker spatial feature map.
- ``late_fusion_attack``: rewrites the attacker per-CAV detections.
- ``adv_coperception_model_manager``: top-level manager that wires the
  attack into the cooperative perception inference pipeline.

See the package README for the architecture overview, glossary, and
data flow diagram.
"""

from .attack_helper import AdvCPAttackHelper, AdvCPCarMeshHelper
from .adv_coperception_model_manager import AdvCoperceptionModelManager
from .early_fusion_attack import AdvCoperceptionEarlyFusionAttack
from .intermediate_fusion_attack import AdvCoperceptionIntermediateFusionAttack
from .late_fusion_attack import AdvCoperceptionLateFusionAttack

__all__ = [
    "AdvCPAttackHelper",
    "AdvCPCarMeshHelper",
    "AdvCoperceptionModelManager",
    "AdvCoperceptionEarlyFusionAttack",
    "AdvCoperceptionIntermediateFusionAttack",
    "AdvCoperceptionLateFusionAttack",
]
