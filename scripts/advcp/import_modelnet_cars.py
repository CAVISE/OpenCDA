"""
CLI for converting ModelNet40 ``car`` meshes into AdvCP vehicle assets.

ModelNet40 (Wu et al., "3D ShapeNets: A Deep Representation for
Volumetric Shapes", CVPR 2015) ships 297 car CAD models as ``.off``
files, split into ``train`` (``car_0001`` .. ``car_0197``) and ``test``
(``car_0198`` .. ``car_0297``). The models share an upright frame
(Z up, vehicle length along Y) but not a common scale or polygon budget.

For every selected model the script:

1. reads the ``.off`` mesh;
2. rotates it so the vehicle length lies along +X (AdvCP/CARLA frame,
   Z up);
3. simplifies it to at most ``--max-faces`` triangles (vertex
   clustering);
4. moves it to the bottom-centre frame and scales it (by default
   uniformly, so its length matches the length of the ``base-car``
   bundle mesh ``car_mesh_0200.ply``);
5. writes ``<output-dir>/<prefix><model-id>/car_mesh.ply`` and the 8-group
   spoof divide ``<output-dir>/<prefix><model-id>/spoof/car_mesh_divide.pkl``,
   i.e. the same layout as the ``base-car`` bundle.

The divide uses :func:`octant_divide_indices` (face-centroid octants)
rather than the slab rule of ``generate_mesh_divide``: the slab rule is
designed for box-like meshes and leaves most groups of a real car mesh
empty. Removal divides are not generated, since removal attacks perturb
the adv-shape template, not the car mesh.

A ``manifest.json`` in ``--output-dir`` (or at ``--manifest``) lists the
source file, face counts and size of every converted model, so assets
can be referenced by their ModelNet40 id (e.g. ``car_0002``).

The target size comes from, in order of precedence: ``--dimensions``,
``--vehicle-blueprint``, ``--reference-mesh`` (default: the ``base-car``
mesh under ``--models-root``).

Usage examples
--------------

Convert every car into ``base-car``-style bundles of a models checkout::

    python -m scripts.advcp.import_modelnet_cars \\
        --modelnet-dir ~/Downloads/ModelNet40/car \\
        --models-root ../models-fork \\
        --output-dir ../models-fork/advcp \\
        --bundle-prefix modelnet40- \\
        --manifest modelnet40-car-manifest.json

Convert selected models with 3000 triangles each::

    python -m scripts.advcp.import_modelnet_cars \\
        --modelnet-dir ~/Downloads/ModelNet40/car \\
        --ids car_0002 car_0010 \\
        --max-faces 3000 \\
        --output-dir ../models-fork/advcp/modelnet40-car
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from opencda.core.plan.model_resolver import default_models_root

from opencda.core.attack.advcp.utils.asset_utils import (
    MeshData,
    read_mesh,
    blueprint_dimensions_m,
    decimate_vertex_clustering,
    divide_piece_face_counts,
    dump_divide_pickle,
    normalize_bottom_center,
    octant_divide_indices,
    parse_dimensions_arg,
    read_off,
    scale_to_dimensions,
    validate_divide_pieces,
    validate_mesh,
    write_mesh,
)

FIT_MODES = ("length", "box", "stretch")

REFERENCE_BUNDLE = Path("advcp") / "base-car" / "car_mesh_0200.ply"

MODELNET_CITATION = (
    "Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, J. Xiao. "
    "3D ShapeNets: A Deep Representation for Volumetric Shapes. CVPR 2015."
)


def align_length_to_x(mesh: MeshData, flip_forward: bool = False) -> MeshData:
    """Rotate an upright (Z-up) mesh about Z so its longer horizontal
    extent lies along X.

    Parameters
    ----------
    mesh : MeshData
        Z-up input mesh.
    flip_forward : bool, optional
        Additionally rotate by 180 degrees about Z (swap front and back).

    Returns
    -------
    MeshData
        Rotated mesh.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    size = vertices.max(axis=0) - vertices.min(axis=0)
    if size[1] > size[0]:
        vertices = np.column_stack([vertices[:, 1], -vertices[:, 0], vertices[:, 2]])
    if flip_forward:
        vertices[:, :2] *= -1.0
    return MeshData(vertices=vertices, faces=np.asarray(mesh.faces, dtype=np.int32))


def mesh_dimensions(path: Path) -> tuple[float, float, float]:
    """Return the bounding-box ``(length, width, height)`` of a mesh file.

    Parameters
    ----------
    path : Path
        Reference mesh (``.ply``, ``.obj`` or ``.off``) in the AdvCP frame.

    Returns
    -------
    tuple of float
        Extents along X, Y and Z.
    """
    vertices = read_mesh(path).vertices
    size = vertices.max(axis=0) - vertices.min(axis=0)
    return (float(size[0]), float(size[1]), float(size[2]))


def fit_to_dimensions(mesh: MeshData, dimensions: tuple[float, float, float], fit: str) -> MeshData:
    """Scale a mesh to the target dimensions and move it to the
    bottom-centre frame.

    Parameters
    ----------
    mesh : MeshData
        Input mesh (length along X, Z up).
    dimensions : tuple of float
        Target ``(length, width, height)`` in meters.
    fit : {"length", "box", "stretch"}
        ``length`` scales uniformly so the length matches; ``box`` scales
        uniformly so the mesh fits inside the dimensions; ``stretch``
        scales each axis to the exact dimensions.

    Returns
    -------
    MeshData
        Scaled, bottom-centred mesh.

    Raises
    ------
    ValueError
        If *fit* is unknown.
    """
    mesh = normalize_bottom_center(mesh)
    if fit == "box":
        return scale_to_dimensions(mesh, dimensions, preserve_aspect=True)
    if fit == "stretch":
        return scale_to_dimensions(mesh, dimensions, preserve_aspect=False)
    if fit != "length":
        raise ValueError(f"Unknown fit mode '{fit}'. Expected one of: {', '.join(FIT_MODES)}.")
    length = float(mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min())
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * (dimensions[0] / length)
    return normalize_bottom_center(MeshData(vertices=vertices, faces=mesh.faces))


def convert_model(
    source: Path,
    output_dir: Path,
    dimensions: tuple[float, float, float],
    max_faces: int,
    fit: str,
    flip_forward: bool,
    bundle_prefix: str = "",
) -> dict[str, object]:
    """Convert one ModelNet40 ``.off`` file into AdvCP assets.

    Parameters
    ----------
    source : Path
        Source ``.off`` file.
    output_dir : Path
        Root output directory; assets go to ``output_dir / <model id>``.
    dimensions : tuple of float
        Target ``(length, width, height)`` in meters.
    max_faces : int
        Upper bound for the number of triangles after simplification.
    fit : {"length", "box", "stretch"}
        Scaling mode, see :func:`fit_to_dimensions`.
    flip_forward : bool
        Whether to swap the front and back of the vehicle.
    bundle_prefix : str, optional
        Prefix of the per-model directory name (``<prefix><model id>``).

    Returns
    -------
    dict
        Manifest entry describing the converted model.

    Raises
    ------
    ValueError
        If the mesh is invalid or a divide group retains no triangles.
    """
    model_id = source.stem
    original = read_off(source)
    mesh = align_length_to_x(original, flip_forward=flip_forward)
    mesh = decimate_vertex_clustering(mesh, target_faces=max_faces)
    mesh = fit_to_dimensions(mesh, dimensions, fit)
    validate_mesh(mesh, f"converted ModelNet40 mesh '{model_id}'")
    divide = octant_divide_indices(mesh)
    validate_divide_pieces(divide, mesh.faces, f"ModelNet40 '{model_id}' spoof divide")

    model_dir = output_dir / f"{bundle_prefix}{model_id}"
    mesh_path = model_dir / "car_mesh.ply"
    divide_path = model_dir / "spoof" / "car_mesh_divide.pkl"
    write_mesh(mesh_path, mesh)
    dump_divide_pickle(divide, divide_path)

    size = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    return {
        "id": model_id,
        "bundle": model_dir.name,
        "split": source.parent.name,
        "source": str(source),
        "source_faces": int(original.faces.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "vertices": int(mesh.vertices.shape[0]),
        "size_m": [round(float(value), 3) for value in size],
        "car_mesh": str(mesh_path.relative_to(output_dir)),
        "spoof_divide": str(divide_path.relative_to(output_dir)),
        "divide_faces_per_group": divide_piece_face_counts(divide, mesh.faces),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the ModelNet40 import script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Convert ModelNet40 car meshes into AdvCP vehicle assets.")
    parser.add_argument(
        "--modelnet-dir",
        type=Path,
        required=True,
        help="ModelNet40 'car' category directory (containing train/ and test/).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Root directory for the generated assets.")
    parser.add_argument(
        "--bundle-prefix",
        type=str,
        default="",
        help="Prefix of each model directory, e.g. 'modelnet40-' gives <output-dir>/modelnet40-car_0002/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path; relative paths are resolved against --output-dir (default manifest.json).",
    )
    parser.add_argument("--ids", nargs="+", default=None, help="Model ids to convert (e.g. car_0002). Defaults to all.")
    parser.add_argument("--max-faces", type=int, default=5000, help="Maximum triangles per simplified mesh (default 5000).")
    parser.add_argument(
        "--vehicle-blueprint",
        type=str,
        default=None,
        help="CARLA blueprint whose approximate dimensions size the meshes (overrides --reference-mesh).",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Models checkout containing advcp/base-car (default: CAVISE_MODELS_ROOT or ../models).",
    )
    parser.add_argument(
        "--reference-mesh",
        type=Path,
        default=None,
        help="Mesh whose bounding box sizes the meshes (default: <models-root>/advcp/base-car/car_mesh_0200.ply).",
    )
    parser.add_argument(
        "--dimensions",
        type=float,
        nargs=3,
        metavar=("LENGTH", "WIDTH", "HEIGHT"),
        default=None,
        help="Target dimensions in meters (overrides --vehicle-blueprint and --reference-mesh).",
    )
    parser.add_argument(
        "--fit",
        choices=FIT_MODES,
        default="length",
        help="Scaling: match the length uniformly (default), fit inside the dimensions uniformly, or stretch per axis.",
    )
    parser.add_argument("--flip-forward", action="store_true", help="Rotate the models by 180 degrees about Z.")
    return parser


def main() -> None:
    """Entry point for the ModelNet40 import CLI."""
    parser = _build_parser()
    args = parser.parse_args()
    try:
        dimensions = parse_dimensions_arg(args.dimensions)
        if dimensions is None and args.vehicle_blueprint:
            dimensions = blueprint_dimensions_m(args.vehicle_blueprint)
        if dimensions is None:
            reference = args.reference_mesh
            if reference is None:
                models_root = args.models_root.expanduser() if args.models_root else default_models_root()
                reference = models_root / REFERENCE_BUNDLE
            reference = reference.expanduser()
            if not reference.is_file():
                parser.error(
                    f"Reference mesh not found: {reference}. Pass --models-root, --reference-mesh, "
                    "--vehicle-blueprint or --dimensions."
                )
            dimensions = mesh_dimensions(reference)
            print(f"Reference size from {reference}: " + " x ".join(f"{value:.3f}" for value in dimensions) + " m")
    except ValueError as exc:
        parser.error(str(exc))

    modelnet_dir = args.modelnet_dir.expanduser()
    sources = sorted(modelnet_dir.glob("*/*.off"), key=lambda path: path.stem)
    if args.ids:
        wanted = set(args.ids)
        sources = [path for path in sources if path.stem in wanted]
        missing = sorted(wanted - {path.stem for path in sources})
        if missing:
            parser.error(f"Model ids not found in {modelnet_dir}: {', '.join(missing)}")
    if not sources:
        parser.error(f"No .off files found under {modelnet_dir}/*/")

    output_dir = args.output_dir.expanduser()
    entries = []
    for index, source in enumerate(sources, start=1):
        try:
            entry = convert_model(
                source,
                output_dir,
                dimensions=dimensions,
                max_faces=args.max_faces,
                fit=args.fit,
                flip_forward=args.flip_forward,
                bundle_prefix=args.bundle_prefix,
            )
        except ValueError as exc:
            print(f"[{index}/{len(sources)}] {source.stem}: skipped ({exc})")
            continue
        entries.append(entry)
        size = " x ".join(f"{value:.2f}" for value in entry["size_m"])
        print(f"[{index}/{len(sources)}] {entry['bundle']}: {entry['source_faces']} -> {entry['faces']} faces, {size} m")

    manifest = {
        "dataset": "ModelNet40",
        "category": "car",
        "citation": MODELNET_CITATION,
        "dimensions_m": list(dimensions),
        "fit": args.fit,
        "max_faces": args.max_faces,
        "models": entries,
    }
    manifest_path = output_dir / (args.manifest.expanduser() if args.manifest else Path("manifest.json"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Converted {len(entries)}/{len(sources)} models.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
