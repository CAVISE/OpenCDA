"""
CLI for generating AdvCP mesh-division metadata (.pkl).

Generates vertex-group index arrays that partition a vehicle mesh into
spatial regions. These index groups are used by the AdvCP attack
pipeline to apply per-region adversarial perturbations during spoofing
or removal attacks.

Two modes are supported:

- ``spoof`` — produces 8 vertex groups (extremal faces and quadrants).
- ``remove`` — produces 10 vertex groups (finer spatial partitioning).

Usage examples
--------------

Generate spoof mesh-divide metadata::

    python -m opencda.core.attack.advcp.utils.generate_mesh_divide \\
        --mesh opencda/core/attack/advcp/3d_models/car_mesh_0200.ply \\
        --mode spoof \\
        --output opencda/core/attack/advcp/3d_models/spoof/car_mesh_divide.pkl

Generate removal mesh-divide metadata::

    python -m opencda.core.attack.advcp.utils.generate_mesh_divide \\
        --mesh opencda/core/attack/advcp/3d_models/car_mesh_0200.ply \\
        --mode remove \\
        --output opencda/core/attack/advcp/3d_models/remove/mesh_divide.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opencda.core.attack.advcp.utils.asset_utils import dump_divide_pickle, generate_divide_indices, read_mesh


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the mesh-divide generation script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with arguments for mesh path, divide mode,
        and output path.
    """
    parser = argparse.ArgumentParser(description="Generate AdvCP mesh-division metadata (.pkl).")
    parser.add_argument("--mesh", type=Path, required=True, help="Input mesh path (.ply or .obj).")
    parser.add_argument("--mode", choices=("spoof", "remove"), required=True, help="Target divide mode.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pkl path.")
    return parser


def main() -> None:
    """Entry point for the mesh-divide generation CLI.

    Parses arguments, reads the input mesh, generates vertex-group
    indices for the specified mode, serialises them to a pickle file,
    and prints a summary.
    """
    args = _build_parser().parse_args()
    mesh = read_mesh(args.mesh)
    divide = generate_divide_indices(mesh.vertices, mode=args.mode)
    dump_divide_pickle(divide, args.output)
    print(f"Generated {args.mode} mesh divide: {args.output} ({len(divide)} groups)")


if __name__ == "__main__":
    main()