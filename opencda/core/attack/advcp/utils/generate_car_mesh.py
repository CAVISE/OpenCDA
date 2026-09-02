"""
CLI for generating an AdvCP-compatible vehicle mesh (.ply).

Generates a bottom-centred, axis-aligned triangle mesh that can be used
as the ``car_mesh_path`` asset in AdvCP spoofing and removal attack
configurations.

The mesh can be constructed in two ways:

1. **From a CARLA blueprint** — a simple box mesh is generated using the
   blueprint's known dimensions (or user-supplied dimensions).
2. **From an external mesh file** — a ``.ply`` or ``.obj`` file is loaded,
   normalised to a bottom-centre frame, and scaled to the target dimensions.

Usage examples
--------------

Generate a mesh for the default Tesla Model 3 blueprint::

    python -m opencda.core.attack.advcp.utils.generate_car_mesh \\
        --output opencda/core/attack/advcp/3d_models/car_mesh_0200.ply

Generate a mesh for a specific blueprint with custom dimensions::

    python -m opencda.core.attack.advcp.utils.generate_car_mesh \\
        --vehicle-blueprint vehicle.audi.a2 \\
        --dimensions 3.7 1.7 1.55 \\
        --output /tmp/car_mesh_audi.ply

Generate a mesh from an external .obj file, preserving aspect ratio::

    python -m opencda.core.attack.advcp.utils.generate_car_mesh \\
        --mesh-input /path/to/vehicle.obj \\
        --dimensions 4.5 2.0 1.6 \\
        --preserve-aspect \\
        --output /tmp/car_mesh_external.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opencda.core.attack.advcp.utils.asset_utils import (
    blueprint_dimensions_m,
    copy_or_generate_mesh,
    parse_dimensions_arg,
    write_mesh,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the car-mesh generation script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with arguments for blueprint, mesh input,
        dimensions, aspect-preservation flag, and output path.
    """
    parser = argparse.ArgumentParser(description="Generate AdvCP-compatible vehicle mesh (.ply).")
    parser.add_argument("--vehicle-blueprint", type=str, default="vehicle.tesla.model3", help="CARLA blueprint id.")
    parser.add_argument("--mesh-input", type=Path, default=None, help="Optional external .ply/.obj mesh input.")
    parser.add_argument(
        "--dimensions",
        type=float,
        nargs=3,
        metavar=("LENGTH", "WIDTH", "HEIGHT"),
        default=None,
        help="Vehicle dimensions in meters. Defaults to blueprint dimensions.",
    )
    parser.add_argument("--preserve-aspect", action="store_true", help="Preserve source mesh proportions while scaling.")
    parser.add_argument("--output", type=Path, required=True, help="Output .ply path.")
    return parser


def main() -> None:
    """Entry point for the car-mesh generation CLI.

    Parses arguments, resolves the target dimensions (from CLI override
    or blueprint lookup), generates or copies the mesh, writes the
    result to the output path, and prints a summary.
    """
    args = _build_parser().parse_args()
    dimensions = parse_dimensions_arg(args.dimensions) or blueprint_dimensions_m(args.vehicle_blueprint)
    mesh = copy_or_generate_mesh(args.mesh_input, dimensions=dimensions, preserve_aspect=args.preserve_aspect)
    write_mesh(args.output, mesh)
    print(f"Generated car mesh: {args.output}")
    print(f"Dimensions (m): length={dimensions[0]:.3f}, width={dimensions[1]:.3f}, height={dimensions[2]:.3f}")


if __name__ == "__main__":
    main()