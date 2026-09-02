"""
CLI for validating AdvCP 3D assets required by spoofing and removal attacks.

Checks:

- File existence and non-empty size.
- Mesh (.ply) format, vertex/face counts, coordinate frame, and scale.
- Spoof mesh-divide (.pkl) structure, index bounds, and compatibility with
  the car mesh vertex count.
- Removal mesh-divide (.pkl) structure, index bounds, and compatibility with
  the car mesh or AdvCP template mesh vertex count.
- Removal perturbation (.npy) shape, dtype, and finite values.

Exits with code 0 when all checks pass, or 1 with actionable error messages
when any check fails.

Usage examples
--------------

Validate all assets::

    python -m opencda.core.attack.advcp.utils.validate_advcp_assets \\
        --car-mesh opencda/core/attack/advcp/3d_models/car_mesh_0200.ply \\
        --spoof-divide opencda/core/attack/advcp/3d_models/spoof/car_mesh_divide.pkl \\
        --remove-divide opencda/core/attack/advcp/3d_models/remove/mesh_divide.pkl \\
        --remove-perturb opencda/core/attack/advcp/3d_models/remove/mesh_perturb.npy

Validate only the car mesh::

    python -m opencda.core.attack.advcp.utils.validate_advcp_assets \\
        --car-mesh opencda/core/attack/advcp/3d_models/car_mesh_0200.ply

Validate with an explicit expected vertex count::

    python -m opencda.core.attack.advcp.utils.validate_advcp_assets \\
        --car-mesh opencda/core/attack/advcp/3d_models/car_mesh_0200.ply \\
        --expected-vertices 148755
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from opencda.core.attack.advcp.utils.asset_utils import (
    load_divide_pickle,
    load_perturbation,
    read_mesh,
    validate_divide_indices,
    validate_mesh,
    validate_mesh_frame_and_scale,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the asset validation script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with arguments for each asset type (car mesh,
        spoof divide, remove divide, remove perturb) and optional
        expected vertex counts.
    """
    parser = argparse.ArgumentParser(
        description="Validate AdvCP 3D assets for spoofing and removal attacks."
    )
    parser.add_argument(
        "--car-mesh",
        type=Path,
        default=None,
        help="Path to the car mesh .ply file.",
    )
    parser.add_argument(
        "--spoof-divide",
        type=Path,
        default=None,
        help="Path to the spoof mesh-divide .pkl file.",
    )
    parser.add_argument(
        "--remove-divide",
        type=Path,
        default=None,
        help="Path to the removal mesh-divide .pkl file.",
    )
    parser.add_argument(
        "--remove-perturb",
        type=Path,
        default=None,
        help="Path to the removal perturbation .npy file.",
    )
    parser.add_argument(
        "--expected-vertices",
        type=int,
        default=None,
        help=(
            "Expected vertex count for the car mesh. When provided, the car mesh "
            "vertex count is checked against this value. When omitted, the check "
            "is skipped."
        ),
    )
    parser.add_argument(
        "--advshape-template-vertices",
        type=int,
        default=None,
        help=(
            "Expected vertex count for the AdvCP template mesh used by removal "
            "divide. When omitted, the removal divide is validated against the "
            "car mesh vertex count (if --car-mesh is provided), or the check is "
            "skipped."
        ),
    )
    return parser


def _check_file_exists(path: Path, label: str) -> None:
    """Assert that a file exists and is non-empty.

    Parameters
    ----------
    path : Path
        File path to check.
    label : str
        Human-readable label for the file (used in error messages).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty (0 bytes).
    """
    if not path.exists():
        raise FileNotFoundError(f"{label}: file not found at '{path}'.")
    if path.stat().st_size == 0:
        raise ValueError(f"{label}: file is empty (0 bytes).")


def _validate_car_mesh(path: Path, expected_vertices: int | None) -> int:
    """Validate a car mesh ``.ply`` file.

    Checks file existence, structural integrity, coordinate frame, and
    scale. Returns the vertex count for use in divide-index validation.

    Parameters
    ----------
    path : Path
        Path to the ``.ply`` file.
    expected_vertices : int or None
        Optional expected vertex count. When provided, the actual vertex
        count must match exactly.

    Returns
    -------
    int
        The number of vertices in the mesh.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, fails structural validation, frame/scale
        validation, or the vertex count does not match *expected_vertices*.
    """
    print(f"Validating car mesh: {path}")
    _check_file_exists(path, "Car mesh")
    mesh = read_mesh(path)
    validate_mesh(mesh, f"Car mesh '{path}'")
    validate_mesh_frame_and_scale(mesh, f"Car mesh '{path}'")
    vertex_count = mesh.vertices.shape[0]
    print(f"  Vertices: {vertex_count}")
    print(f"  Faces:    {mesh.faces.shape[0]}")
    print(f"  Bounds:   x=[{mesh.vertices[:, 0].min():.3f}, {mesh.vertices[:, 0].max():.3f}], "
          f"y=[{mesh.vertices[:, 1].min():.3f}, {mesh.vertices[:, 1].max():.3f}], "
          f"z=[{mesh.vertices[:, 2].min():.3f}, {mesh.vertices[:, 2].max():.3f}]")
    if expected_vertices is not None and vertex_count != expected_vertices:
        raise ValueError(
            f"Car mesh vertex count mismatch: expected {expected_vertices}, got {vertex_count}."
        )
    print("  [PASS]")
    return vertex_count


def _validate_spoof_divide(path: Path, car_vertex_count: int | None) -> None:
    """Validate a spoof mesh-divide ``.pkl`` file.

    Checks file existence, that the divide contains exactly 8 groups,
    and that all indices are valid for the given car mesh vertex count.

    Parameters
    ----------
    path : Path
        Path to the ``.pkl`` file.
    car_vertex_count : int or None
        Vertex count of the corresponding car mesh. When ``None``,
        index-bound validation is skipped.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, contains the wrong number of groups, or
        has out-of-bounds indices.
    """
    print(f"Validating spoof mesh divide: {path}")
    _check_file_exists(path, "Spoof mesh divide")
    indices = load_divide_pickle(path)
    if len(indices) != 8:
        raise ValueError(
            f"Spoof mesh divide must contain exactly 8 index groups, got {len(indices)}."
        )
    if car_vertex_count is not None:
        validate_divide_indices(indices, car_vertex_count, "Spoof mesh divide")
    print(f"  Groups: {len(indices)}")
    for i, g in enumerate(indices):
        print(f"    Group {i}: {g.shape[0]} indices, range [{g.min()}, {g.max()}]")
    print("  [PASS]")


def _validate_remove_divide(
    path: Path,
    car_vertex_count: int | None,
    template_vertex_count: int | None,
) -> None:
    """Validate a removal mesh-divide ``.pkl`` file.

    Checks file existence, that the divide contains exactly 10 groups,
    and that all indices are valid. The expected vertex count is
    resolved from *car_vertex_count* first, then *template_vertex_count*.

    Parameters
    ----------
    path : Path
        Path to the ``.pkl`` file.
    car_vertex_count : int or None
        Vertex count of the car mesh (used when available).
    template_vertex_count : int or None
        Expected vertex count for the AdvCP template mesh (used as
        fallback when *car_vertex_count* is ``None``).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, contains the wrong number of groups, or
        has out-of-bounds indices.
    """
    print(f"Validating removal mesh divide: {path}")
    _check_file_exists(path, "Removal mesh divide")
    indices = load_divide_pickle(path)
    if len(indices) != 10:
        raise ValueError(
            f"Removal mesh divide must contain exactly 10 index groups, got {len(indices)}."
        )
    # Determine the expected vertex count for validation
    expected_count = car_vertex_count or template_vertex_count
    if expected_count is not None:
        validate_divide_indices(indices, expected_count, "Removal mesh divide")
    print(f"  Groups: {len(indices)}")
    for i, g in enumerate(indices):
        print(f"    Group {i}: {g.shape[0]} indices, range [{g.min()}, {g.max()}]")
    print("  [PASS]")


def _validate_remove_perturb(path: Path, car_vertex_count: int | None) -> None:
    """Validate a removal perturbation ``.npy`` file.

    Checks file existence, that the perturbation has shape (N, 3),
    contains only finite values, and its vertex count matches the car
    mesh (when *car_vertex_count* is provided).

    Parameters
    ----------
    path : Path
        Path to the ``.npy`` file.
    car_vertex_count : int or None
        Vertex count of the corresponding car mesh. When provided, the
        perturbation's first dimension must match.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, has the wrong shape, contains non-finite
        values, or the vertex count does not match.
    """
    print(f"Validating removal perturbation: {path}")
    _check_file_exists(path, "Removal perturbation")
    perturbation = load_perturbation(path)
    print(f"  Shape: {perturbation.shape}")
    print(f"  Dtype: {perturbation.dtype}")
    print(f"  Range: [{perturbation.min():.6f}, {perturbation.max():.6f}]")
    if car_vertex_count is not None and perturbation.shape[0] != car_vertex_count:
        raise ValueError(
            f"Removal perturbation vertex count mismatch: car mesh has {car_vertex_count} "
            f"vertices, perturbation has {perturbation.shape[0]}."
        )
    print("  [PASS]")


def main() -> None:
    """Entry point for the AdvCP asset validation CLI.

    Parses command-line arguments, validates each specified asset in
    order (car mesh, spoof divide, removal divide, removal perturbation),
    and exits with code 0 on success or code 1 with error messages on
    failure.

    Validation is additive: all specified assets are checked, and
    all errors are reported before exiting.
    """
    args = _build_parser().parse_args()

    has_any_asset = any([
        args.car_mesh is not None,
        args.spoof_divide is not None,
        args.remove_divide is not None,
        args.remove_perturb is not None,
    ])
    if not has_any_asset:
        print("No assets specified. Use --car-mesh, --spoof-divide, --remove-divide, "
              "and/or --remove-perturb to select assets for validation.")
        sys.exit(1)

    errors: list[str] = []
    car_vertex_count: int | None = None

    # 1. Validate car mesh (if provided)
    if args.car_mesh is not None:
        try:
            car_vertex_count = _validate_car_mesh(args.car_mesh, args.expected_vertices)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))

    # 2. Validate spoof mesh divide (if provided)
    if args.spoof_divide is not None:
        try:
            _validate_spoof_divide(args.spoof_divide, car_vertex_count)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))

    # 3. Validate removal mesh divide (if provided)
    if args.remove_divide is not None:
        try:
            _validate_remove_divide(
                args.remove_divide,
                car_vertex_count,
                args.advshape_template_vertices,
            )
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))

    # 4. Validate removal perturbation (if provided)
    if args.remove_perturb is not None:
        try:
            _validate_remove_perturb(args.remove_perturb, car_vertex_count)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))

    if errors:
        print("\nValidation FAILED with the following errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("\nAll specified assets passed validation.")


if __name__ == "__main__":
    main()