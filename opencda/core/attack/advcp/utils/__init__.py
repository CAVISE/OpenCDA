"""
AdvCP asset generation and validation utilities.

Available modules
-----------------
generate_car_mesh
    CLI for generating a vehicle mesh (.ply) from a CARLA blueprint or
    external mesh input.
generate_mesh_divide
    CLI for generating spoof or removal mesh-division metadata (.pkl).
generate_remove_advshape_assets
    CLI for generating removal assets (.pkl and/or .npy) including
    mesh-division metadata and adversarial-shape perturbation tensors.
validate_advcp_assets
    CLI for validating .ply, .pkl, and .npy assets against the AdvCP
    pipeline requirements.
runtime_assets
    :class:`~AdvCPRuntimeAssetHelper` for on-demand asset generation
    during simulation.
asset_utils
    Core mesh, divide, and perturbation helpers (read, write, validate,
    transform).
"""

