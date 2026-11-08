"""
AdvCP asset library shared by the attack runtime and the asset scripts.

Available modules
-----------------
asset_utils
    Core mesh, divide, and perturbation helpers (read, write, validate,
    transform), including the canonical adv-shape template used by the
    runtime.
runtime_assets
    :class:`~AdvCPRuntimeAssetHelper` for on-demand asset generation
    during simulation.

The command-line entry points (``generate_car_mesh``,
``generate_mesh_divide``, ``generate_remove_advshape_assets`` and
``validate_advcp_assets``) live in ``scripts/advcp`` and are run from the
repository root, e.g. ``python -m scripts.advcp.generate_car_mesh``.
"""
