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

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AdvCPAttackHelper": ("opencda.core.attack.advcp.attack_helper", "AdvCPAttackHelper"),
    "AdvCPCarMeshHelper": ("opencda.core.attack.advcp.attack_helper", "AdvCPCarMeshHelper"),
    "AdvCoperceptionModelManager": (
        "opencda.core.attack.advcp.adv_coperception_model_manager",
        "AdvCoperceptionModelManager",
    ),
    "AdvCoperceptionEarlyFusionAttack": (
        "opencda.core.attack.advcp.early_fusion_attack",
        "AdvCoperceptionEarlyFusionAttack",
    ),
    "AdvCoperceptionIntermediateFusionAttack": (
        "opencda.core.attack.advcp.intermediate_fusion_attack",
        "AdvCoperceptionIntermediateFusionAttack",
    ),
    "AdvCoperceptionLateFusionAttack": (
        "opencda.core.attack.advcp.late_fusion_attack",
        "AdvCoperceptionLateFusionAttack",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: DC02
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
