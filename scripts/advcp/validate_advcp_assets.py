"""
CLI for validating AdvCP 3D assets required by spoofing and removal attacks.

Checks:

- File existence and non-empty size.
- Mesh (.ply) format, vertex/face counts, coordinate frame, and scale.
- Spoof mesh-divide (.pkl) structure, index bounds, and compatibility with
  the car mesh vertex count.
- Removal mesh-divide (.pkl) structure, index bounds, and non-empty mesh
  pieces of the AdvCP adv-shape template.
- Removal perturbation (.npy) shape, dtype, finite values, and vertex count
  of the AdvCP adv-shape template.

Removal assets are indexed against the adv-shape template used by the
attack runtime (:func:`advshape_template_mesh`, 98 vertices), not against
``--car-mesh``; ``--advshape-template-vertices`` overrides the expected
template vertex count.

Exits with code 0 when all checks pass, or 1 with actionable error messages
when any check fails.

Usage examples
--------------

Validate all assets::

    python -m scripts.advcp.validate_advcp_assets \\
        --car-mesh ../models/advcp/my-car/car_mesh_0200.ply \\
        --spoof-divide ../models/advcp/my-car/spoof/car_mesh_divide.pkl \\
        --remove-divide ../models/advcp/my-car/remove/mesh_divide.pkl \\
        --remove-perturb ../models/advcp/my-car/remove/mesh_perturb.npy

Validate only the car mesh::

    python -m scripts.advcp.validate_advcp_assets \\
        --car-mesh ../models/advcp/my-car/car_mesh_0200.ply

Validate with an explicit expected vertex count::

    python -m scripts.advcp.validate_advcp_assets \\
        --car-mesh ../models/advcp/my-car/car_mesh_0200.ply \\
        --expected-vertices 148755
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from opencda.core.attack.advcp.utils.asset_utils import (
    MeshData,
    advshape_template_mesh,
    load_divide_pickle,
    load_perturbation,
    read_mesh,
    validate_divide_indices,
    validate_divide_pieces,
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
    parser = argparse.ArgumentParser(description="Validate AdvCP 3D assets for spoofing and removal attacks.")
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
            "Expected vertex count of the adv-shape template that the removal "
            "divide and perturbation are indexed against. Defaults to the AdvCP "
            "runtime template (98 vertices). Takes precedence when provided."
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


def _validate_car_mesh(path: Path, expected_vertices: int | None) -> MeshData:
    """Validate a car mesh ``.ply`` file.

    Checks file existence, structural integrity, coordinate frame, and
    scale. Returns the parsed mesh for use in divide validation.

    Parameters
    ----------
    path : Path
        Path to the ``.ply`` file.
    expected_vertices : int or None
        Optional expected vertex count. When provided, the actual vertex
        count must match exactly.

    Returns
    -------
    MeshData
        The parsed mesh.

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
    print(
        f"  Bounds:   x=[{mesh.vertices[:, 0].min():.3f}, {mesh.vertices[:, 0].max():.3f}], "
        f"y=[{mesh.vertices[:, 1].min():.3f}, {mesh.vertices[:, 1].max():.3f}], "
        f"z=[{mesh.vertices[:, 2].min():.3f}, {mesh.vertices[:, 2].max():.3f}]"
    )
    if expected_vertices is not None and vertex_count != expected_vertices:
        raise ValueError(f"Car mesh vertex count mismatch: expected {expected_vertices}, got {vertex_count}.")
    print("  [PASS]")
    return mesh


def _validate_spoof_divide(path: Path, car_mesh: MeshData | None) -> None:
    """Validate a spoof mesh-divide ``.pkl`` file.

    Checks file existence, that the divide contains exactly 8 groups,
    that all indices are valid for the given car mesh, and that every
    group yields a non-empty mesh piece.

    Parameters
    ----------
    path : Path
        Path to the ``.pkl`` file.
    car_mesh : MeshData or None
        The corresponding car mesh. When ``None``, index-bound and
        mesh-piece validation are skipped.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, contains the wrong number of groups, has
        out-of-bounds indices, or a group retains no triangles.
    """
    print(f"Validating spoof mesh divide: {path}")
    _check_file_exists(path, "Spoof mesh divide")
    indices = load_divide_pickle(path)
    if len(indices) != 8:
        raise ValueError(f"Spoof mesh divide must contain exactly 8 index groups, got {len(indices)}.")
    if car_mesh is not None:
        validate_divide_indices(indices, car_mesh.vertices.shape[0], "Spoof mesh divide")
        validate_divide_pieces(indices, car_mesh.faces, "Spoof mesh divide")
    print(f"  Groups: {len(indices)}")
    for i, g in enumerate(indices):
        print(f"    Group {i}: {g.shape[0]} indices, range [{g.min()}, {g.max()}]")
    print("  [PASS]")


def _resolve_removal_template(template_vertices_override: int | None) -> tuple[int, MeshData | None]:
    """Resolve the template that removal assets are validated against.

    Parameters
    ----------
    template_vertices_override : int or None
        Value of ``--advshape-template-vertices``. Takes precedence over
        the runtime template when provided.

    Returns
    -------
    tuple
        ``(vertex_count, template_mesh)``. *template_mesh* is the runtime
        template when its vertex count is the one being validated, or
        ``None`` for a custom override (mesh pieces are then not checked).

    Raises
    ------
    ValueError
        If the override is not positive.
    """
    template = advshape_template_mesh()
    runtime_vertex_count = template.vertices.shape[0]
    if template_vertices_override is None or template_vertices_override == runtime_vertex_count:
        return runtime_vertex_count, template
    if template_vertices_override <= 0:
        raise ValueError(f"--advshape-template-vertices must be positive, got {template_vertices_override}.")
    print(
        f"Note: validating removal assets against {template_vertices_override} template vertices; "
        f"the AdvCP runtime always uses the {runtime_vertex_count}-vertex adv-shape template."
    )
    return template_vertices_override, None


def _validate_remove_divide(
    path: Path,
    template_vertex_count: int,
    template_mesh: MeshData | None,
) -> None:
    """Validate a removal mesh-divide ``.pkl`` file.

    Checks file existence, that the divide contains exactly 10 groups,
    that all indices are valid for the adv-shape template, and (when the
    template mesh is known) that every group yields a non-empty piece.

    Parameters
    ----------
    path : Path
        Path to the ``.pkl`` file.
    template_vertex_count : int
        Vertex count of the adv-shape template the indices refer to.
    template_mesh : MeshData or None
        The adv-shape template mesh, or ``None`` to skip the mesh-piece
        check.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, contains the wrong number of groups, has
        out-of-bounds indices, or a group retains no triangles.
    """
    print(f"Validating removal mesh divide: {path}")
    _check_file_exists(path, "Removal mesh divide")
    indices = load_divide_pickle(path)
    if len(indices) != 10:
        raise ValueError(f"Removal mesh divide must contain exactly 10 index groups, got {len(indices)}.")
    validate_divide_indices(indices, template_vertex_count, "Removal mesh divide")
    if template_mesh is not None:
        validate_divide_pieces(indices, template_mesh.faces, "Removal mesh divide")
    print(f"  Groups: {len(indices)}")
    for i, g in enumerate(indices):
        print(f"    Group {i}: {g.shape[0]} indices, range [{g.min()}, {g.max()}]")
    print("  [PASS]")


def _validate_remove_perturb(path: Path, template_vertex_count: int) -> None:
    """Validate a removal perturbation ``.npy`` file.

    Checks file existence, that the perturbation has shape (N, 3),
    contains only finite values, and that N matches the adv-shape
    template vertex count.

    Parameters
    ----------
    path : Path
        Path to the ``.npy`` file.
    template_vertex_count : int
        Vertex count of the adv-shape template the perturbation is
        applied to.

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
    if perturbation.shape[0] != template_vertex_count:
        raise ValueError(
            f"Removal perturbation vertex count mismatch: adv-shape template has {template_vertex_count} vertices, "
            f"perturbation has {perturbation.shape[0]}."
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

    has_any_asset = any(
        [
            args.car_mesh is not None,
            args.spoof_divide is not None,
            args.remove_divide is not None,
            args.remove_perturb is not None,
        ]
    )
    if not has_any_asset:
        print("No assets specified. Use --car-mesh, --spoof-divide, --remove-divide, and/or --remove-perturb to select assets for validation.")
        sys.exit(1)

    errors: list[str] = []
    car_mesh: MeshData | None = None

    # 1. Validate car mesh (if provided)
    if args.car_mesh is not None:
        try:
            car_mesh = _validate_car_mesh(args.car_mesh, args.expected_vertices)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))

    # 2. Validate spoof mesh divide (if provided)
    if args.spoof_divide is not None:
        try:
            _validate_spoof_divide(args.spoof_divide, car_mesh)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))

    # 3-4. Removal assets are indexed against the adv-shape template, not the car mesh
    if args.remove_divide is not None or args.remove_perturb is not None:
        try:
            template_vertex_count, template_mesh = _resolve_removal_template(args.advshape_template_vertices)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if args.remove_divide is not None:
                try:
                    _validate_remove_divide(args.remove_divide, template_vertex_count, template_mesh)
                except (FileNotFoundError, ValueError) as exc:
                    errors.append(str(exc))
            if args.remove_perturb is not None:
                try:
                    _validate_remove_perturb(args.remove_perturb, template_vertex_count)
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
