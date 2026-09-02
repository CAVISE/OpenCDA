"""
CLI for generating AdvCP removal mesh-division and adversarial-shape perturbation assets.

Supports generation of:

- ``mesh_divide.pkl`` (removal mode) — vertex-group indices derived from an
  AdvCP-shaped template mesh, or from a user-supplied mesh.
- ``mesh_perturb.npy`` — optional zero-initialised or random perturbation
  tensor that can be used as a warm-start or placeholder when ``advshape: true``.

Usage examples
--------------

Generate removal mesh-divide from an existing car mesh::

    python -m opencda.core.attack.advcp.utils.generate_remove_advshape_assets \\
        --mode divide \\
        --mesh opencda/core/attack/advcp/3d_models/car_mesh_0200.ply \\
        --output opencda/core/attack/advcp/3d_models/remove/mesh_divide.pkl

Generate removal mesh-divide from the built-in AdvCP template mesh::

    python -m opencda.core.attack.advcp.utils.generate_remove_advshape_assets \\
        --mode divide \\
        --output opencda/core/attack/advcp/3d_models/remove/mesh_divide.pkl

Generate a zero-initialised perturbation tensor::

    python -m opencda.core.attack.advcp.utils.generate_remove_advshape_assets \\
        --mode perturb \\
        --output opencda/core/attack/advcp/3d_models/remove/mesh_perturb.npy

Generate a random perturbation tensor (for testing / warm-start)::

    python -m opencda.core.attack.advcp.utils.generate_remove_advshape_assets \\
        --mode perturb \\
        --random \\
        --seed 42 \\
        --output opencda/core/attack/advcp/3d_models/remove/mesh_perturb.npy

Generate both divide and perturb in one call::

    python -m opencda.core.attack.advcp.utils.generate_remove_advshape_assets \\
        --mode both \\
        --mesh opencda/core/attack/advcp/3d_models/car_mesh_0200.ply \\
        --divide-output opencda/core/attack/advcp/3d_models/remove/mesh_divide.pkl \\
        --perturb-output opencda/core/attack/advcp/3d_models/remove/mesh_perturb.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from opencda.core.attack.advcp.utils.asset_utils import (
    MeshData,
    advshape_template_mesh,
    dump_divide_pickle,
    generate_divide_indices,
    read_mesh,
    save_perturbation,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the removal-asset generation script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with arguments for mode selection, mesh input,
        output paths, and perturbation options (randomisation, seed,
        scale).
    """
    parser = argparse.ArgumentParser(
        description="Generate AdvCP removal assets (mesh-divide .pkl and/or perturbation .npy)."
    )
    parser.add_argument(
        "--mode",
        choices=("divide", "perturb", "both"),
        required=True,
        help=(
            "'divide' – generate removal mesh-divide metadata (.pkl). "
            "'perturb' – generate adversarial-shape perturbation (.npy). "
            "'both' – generate both."
        ),
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help=(
            "Input mesh path (.ply or .obj). When omitted, the built-in AdvCP "
            "template mesh (subdivided box) is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path. Used by 'divide' and 'perturb' modes when a single output "
            "is produced. For 'both' mode, use --divide-output and --perturb-output instead."
        ),
    )
    parser.add_argument(
        "--divide-output",
        type=Path,
        default=None,
        help="Output path for removal mesh-divide .pkl (only used with --mode both).",
    )
    parser.add_argument(
        "--perturb-output",
        type=Path,
        default=None,
        help="Output path for perturbation .npy (only used with --mode both).",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate a random perturbation instead of zeros.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic perturbation generation.",
    )
    parser.add_argument(
        "--perturb-scale",
        type=float,
        default=0.5,
        help="Maximum per-vertex displacement (meters) when --random is set. Default: 0.5.",
    )
    return parser


def _resolve_mesh(mesh_input: Path | None) -> MeshData:
    """Read or build the source mesh for removal divide generation.

    Parameters
    ----------
    mesh_input : Path or None
        Optional path to an external ``.ply`` or ``.obj`` mesh. When
        ``None``, the built-in AdvCP template mesh is returned.

    Returns
    -------
    MeshData
        The source mesh to use for divide-index or perturbation generation.
    """
    if mesh_input is not None:
        return read_mesh(mesh_input)
    return advshape_template_mesh()


def _generate_divide(mesh_input: Path | None, output: Path) -> None:
    """Generate removal mesh-divide indices and write them to a pickle file.

    Parameters
    ----------
    mesh_input : Path or None
        Input mesh path, or ``None`` to use the built-in AdvCP template.
    output : Path
        Destination path for the ``.pkl`` file.
    """
    mesh = _resolve_mesh(mesh_input)
    divide = generate_divide_indices(mesh.vertices, mode="remove")
    dump_divide_pickle(divide, output)
    print(f"Generated removal mesh divide: {output} ({len(divide)} groups, {output.stat().st_size} bytes)")


def _generate_perturb(
    mesh_input: Path | None,
    output: Path,
    randomize: bool,
    seed: int | None,
    scale: float,
) -> None:
    """Generate a per-vertex perturbation tensor and write it to a ``.npy`` file.

    Parameters
    ----------
    mesh_input : Path or None
        Input mesh path, or ``None`` to use the built-in AdvCP template.
    output : Path
        Destination path for the ``.npy`` file.
    randomize : bool
        If ``True``, generate random perturbations in ``[-scale, scale]``.
        If ``False``, generate a zero-initialised tensor.
    seed : int or None
        Random seed for deterministic generation (only used when
        *randomize* is ``True``).
    scale : float
        Maximum per-vertex displacement in meters (only used when
        *randomize* is ``True``).
    """
    mesh = _resolve_mesh(mesh_input)
    vertex_count = mesh.vertices.shape[0]

    if randomize:
        rng = np.random.default_rng(seed)
        perturbation = rng.uniform(-scale, scale, size=(vertex_count, 3)).astype(np.float32)
        source_desc = f"random (scale={scale})"
    else:
        perturbation = np.zeros((vertex_count, 3), dtype=np.float32)
        source_desc = "zero-initialised"

    save_perturbation(output, perturbation)
    print(
        f"Generated {source_desc} perturbation: {output} "
        f"(shape={perturbation.shape}, {output.stat().st_size} bytes)"
    )


def main() -> None:
    """Entry point for the removal-asset generation CLI.

    Parses arguments and dispatches to the appropriate generation
    function based on the selected ``--mode``:

    - ``divide`` — generates only the removal mesh-divide ``.pkl``.
    - ``perturb`` — generates only the perturbation ``.npy``.
    - ``both`` — generates both assets in a single invocation.
    """
    args = _build_parser().parse_args()

    if args.mode == "divide":
        if args.output is None:
            raise ValueError("--output is required when --mode=divide.")
        _generate_divide(args.mesh, args.output)

    elif args.mode == "perturb":
        if args.output is None:
            raise ValueError("--output is required when --mode=perturb.")
        _generate_perturb(args.mesh, args.output, args.random, args.seed, args.perturb_scale)

    elif args.mode == "both":
        div_out = args.divide_output or args.output
        pert_out = args.perturb_output or args.output
        if div_out is None:
            raise ValueError("Either --divide-output or --output must be provided when --mode=both.")
        if pert_out is None:
            raise ValueError("Either --perturb-output or --output must be provided when --mode=both.")
        _generate_divide(args.mesh, div_out)
        _generate_perturb(args.mesh, pert_out, args.random, args.seed, args.perturb_scale)


if __name__ == "__main__":
    main()