"""
Core mesh, divide, and perturbation helpers for AdvCP asset generation.

This module provides the foundational data structures and operations used
by all AdvCP asset utilities:

- :class:`MeshData` — lightweight container for vertices and faces.
- Mesh construction: :func:`box_mesh`, :func:`subdivide_midpoint`,
  :func:`advshape_template_mesh`.
- Mesh I/O: :func:`read_mesh`, :func:`read_ply`, :func:`read_obj`,
  :func:`write_ascii_ply`, :func:`write_mesh`.
- Mesh transforms: :func:`normalize_bottom_center`,
  :func:`scale_to_dimensions`, :func:`copy_or_generate_mesh`.
- Divide-index generation and I/O: :func:`generate_divide_indices`,
  :func:`dump_divide_pickle`, :func:`load_divide_pickle`.
- Validation: :func:`validate_mesh`, :func:`validate_mesh_frame_and_scale`,
  :func:`validate_divide_indices`.
- Perturbation I/O: :func:`save_perturbation`, :func:`load_perturbation`.
- Blueprint dimension lookup: :func:`blueprint_dimensions_m`,
  :func:`parse_dimensions_arg`.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt


_BLUEPRINT_DIMENSIONS_M = {
    "vehicle.tesla.model3": (4.30, 1.91, 1.26),
    "vehicle.audi.a2": (3.70, 1.70, 1.55),
    "vehicle.lincoln.mkz_2017": (4.93, 1.86, 1.48),
    "vehicle.mercedes.coupe_2020": (4.69, 1.83, 1.40),
    "vehicle.dodge.charger_2020": (5.10, 1.90, 1.50),
}
_DEFAULT_DIMENSIONS_M = (4.30, 1.91, 1.26)


@dataclass(frozen=True)
class MeshData:
    """Lightweight container for a triangle mesh.

    Attributes
    ----------
    vertices : ndarray of float64, shape (N, 3)
        Vertex coordinates in 3D space.
    faces : ndarray of int32, shape (M, 3)
        Triangle face indices into the vertex array.
    """

    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int32]


def blueprint_dimensions_m(blueprint: str | None) -> tuple[float, float, float]:
    """Return the (length, width, height) in meters for a CARLA blueprint.

    Parameters
    ----------
    blueprint : str or None
        CARLA blueprint identifier (e.g. ``"vehicle.tesla.model3"``).
        When ``None`` or unknown, default Tesla Model 3 dimensions are
        returned.

    Returns
    -------
    tuple of float
        ``(length, width, height)`` in meters.
    """
    if blueprint is None:
        return _DEFAULT_DIMENSIONS_M
    return _BLUEPRINT_DIMENSIONS_M.get(blueprint, _DEFAULT_DIMENSIONS_M)


def parse_dimensions_arg(raw_dimensions: Iterable[float] | None) -> tuple[float, float, float] | None:
    """Parse and validate a user-supplied (length, width, height) triple.

    Parameters
    ----------
    raw_dimensions : iterable of float or None
        Raw dimension values, typically from a CLI argument.

    Returns
    -------
    tuple of float or None
        ``(length, width, height)`` when *raw_dimensions* is not ``None``,
        otherwise ``None``.

    Raises
    ------
    ValueError
        If the iterable does not contain exactly 3 values, or if any
        value is non-positive.
    """
    if raw_dimensions is None:
        return None
    dims = tuple(float(value) for value in raw_dimensions)
    if len(dims) != 3:
        raise ValueError("Dimensions must contain exactly 3 floats: length width height.")
    if any(value <= 0.0 for value in dims):
        raise ValueError(f"Dimensions must be positive, got {dims}.")
    return dims


def box_mesh(length: float, width: float, height: float) -> MeshData:
    """Build an axis-aligned box mesh with the given dimensions.

    The box is centred in XY and sits on the Z=0 plane (bottom face at
    Z=0, top face at Z=*height*).

    Parameters
    ----------
    length : float
        Size along the X axis (meters).
    width : float
        Size along the Y axis (meters).
    height : float
        Size along the Z axis (meters).

    Returns
    -------
    MeshData
        An 8-vertex, 12-face box mesh.
    """
    x0, x1 = -length / 2.0, length / 2.0
    y0, y1 = -width / 2.0, width / 2.0
    z0, z1 = 0.0, height
    vertices = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 6, 5],
            [4, 7, 6],  # top
            [0, 5, 1],
            [0, 4, 5],  # -y
            [1, 6, 2],
            [1, 5, 6],  # +x
            [2, 7, 3],
            [2, 6, 7],  # +y
            [3, 4, 0],
            [3, 7, 4],  # -x
        ],
        dtype=np.int32,
    )
    return MeshData(vertices=vertices, faces=faces)


def normalize_bottom_center(mesh: MeshData) -> MeshData:
    """Translate the mesh so its bottom face sits on Z=0 and its XY
    centroid is at the origin.

    Parameters
    ----------
    mesh : MeshData
        Input mesh.

    Returns
    -------
    MeshData
        A new mesh with the bottom-centre coordinate frame.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center_xy = (mins[:2] + maxs[:2]) / 2.0
    vertices[:, 0] -= center_xy[0]
    vertices[:, 1] -= center_xy[1]
    vertices[:, 2] -= mins[2]
    return MeshData(vertices=vertices, faces=np.asarray(mesh.faces, dtype=np.int32))


def scale_to_dimensions(mesh: MeshData, dimensions: tuple[float, float, float], preserve_aspect: bool = False) -> MeshData:
    """Scale a mesh to the target (length, width, height).

    The mesh is first normalised to a bottom-centre frame, then scaled
    along each axis independently (or uniformly when *preserve_aspect*
    is ``True``).

    Parameters
    ----------
    mesh : MeshData
        Input mesh.
    dimensions : tuple of float
        Target ``(length, width, height)`` in meters.
    preserve_aspect : bool, optional
        If ``True``, the smallest axis-aligned scale factor is applied
        uniformly to all three axes, preserving the original proportions.
        Defaults to ``False``.

    Returns
    -------
    MeshData
        Scaled mesh in bottom-centre frame.

    Raises
    ------
    ValueError
        If the mesh bounding box is degenerate (any axis span <= 1e-9).
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    size = maxs - mins
    if np.any(size <= 1e-9):
        raise ValueError("Cannot scale mesh with degenerate bounding-box size.")
    target = np.asarray(dimensions, dtype=np.float64)
    if preserve_aspect:
        scale = float(np.min(target / size))
        scales = np.asarray([scale, scale, scale], dtype=np.float64)
    else:
        scales = target / size
    vertices = (vertices - mins) * scales + mins
    return normalize_bottom_center(MeshData(vertices=vertices, faces=np.asarray(mesh.faces, dtype=np.int32)))


def subdivide_midpoint(mesh: MeshData, levels: int) -> MeshData:
    """Subdivide a triangle mesh using midpoint insertion.

    Each subdivision level splits every triangle into 4 smaller
    triangles by inserting a vertex at the midpoint of each edge.
    Shared edges reuse the same midpoint vertex.

    Parameters
    ----------
    mesh : MeshData
        Input mesh.
    levels : int
        Number of subdivision passes. ``0`` returns the mesh unchanged.

    Returns
    -------
    MeshData
        Subdivided mesh with increased vertex and face density.
    """
    if levels <= 0:
        return mesh

    vertices = [tuple(vertex) for vertex in np.asarray(mesh.vertices, dtype=np.float64)]
    faces = [tuple(int(index) for index in face) for face in np.asarray(mesh.faces, dtype=np.int32)]

    for _ in range(levels):
        edge_midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces: list[tuple[int, int, int]] = []

        def midpoint_index(a: int, b: int) -> int:
            key = (a, b) if a <= b else (b, a)
            if key in edge_midpoint_cache:
                return edge_midpoint_cache[key]
            midpoint = (np.asarray(vertices[a], dtype=np.float64) + np.asarray(vertices[b], dtype=np.float64)) / 2.0
            vertices.append((float(midpoint[0]), float(midpoint[1]), float(midpoint[2])))
            idx = len(vertices) - 1
            edge_midpoint_cache[key] = idx
            return idx

        for i0, i1, i2 in faces:
            m01 = midpoint_index(i0, i1)
            m12 = midpoint_index(i1, i2)
            m20 = midpoint_index(i2, i0)
            new_faces.extend(
                [
                    (i0, m01, m20),
                    (i1, m12, m01),
                    (i2, m20, m12),
                    (m01, m12, m20),
                ]
            )
        faces = new_faces

    return MeshData(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int32))


def advshape_template_mesh() -> MeshData:
    """Return the default AdvCP adversarial-shape template mesh.

    The template is a 4.9 x 2.5 x 2.0 meter box subdivided twice via
    midpoint insertion, producing a mesh with approximately 98 vertices
    and 192 faces. This provides enough geometric resolution for
    per-vertex adversarial perturbations.

    Returns
    -------
    MeshData
        The AdvCP template mesh.
    """
    template = box_mesh(4.9, 2.5, 2.0)
    return subdivide_midpoint(template, levels=2)


def generate_divide_indices(vertices: npt.NDArray[np.float64], mode: str) -> list[npt.NDArray[np.int32]]:
    """Generate vertex-group indices for spoofing or removal attacks.

    The mesh bounding box is partitioned into spatial regions. Each
    region is defined by thresholding vertex coordinates against the
    bounding-box extents and midpoints. The number and layout of groups
    depends on *mode*:

    - ``"spoof"`` — 8 groups (extremal faces and quadrants).
    - ``"remove"`` — 10 groups (finer spatial partitioning).

    Parameters
    ----------
    vertices : ndarray of float64, shape (N, 3)
        Mesh vertex coordinates.
    mode : {"spoof", "remove"}
        Target divide mode.

    Returns
    -------
    list of ndarray of int32
        Each element is a 1-D array of vertex indices belonging to one
        spatial group. Empty groups are replaced with all vertex indices
        as a fallback.

    Raises
    ------
    ValueError
        If *mode* is not one of ``"spoof"`` or ``"remove"``.
    """
    mode_normalized = mode.strip().lower()
    if mode_normalized not in {"spoof", "remove"}:
        raise ValueError(f"Unsupported divide mode '{mode}'. Expected one of: spoof, remove.")

    coords = np.asarray(vertices, dtype=np.float64)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    mids = (mins + maxs) / 2.0
    eps = np.maximum((maxs - mins) * 0.02, 1e-3)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    groups: list[npt.NDArray[np.int32]]

    if mode_normalized == "spoof":
        groups = [
            np.argwhere(x >= maxs[0] - eps[0]).reshape(-1),
            np.argwhere(x <= mins[0] + eps[0]).reshape(-1),
            np.argwhere(y >= maxs[1] - eps[1]).reshape(-1),
            np.argwhere(y <= mins[1] + eps[1]).reshape(-1),
            np.argwhere(z >= maxs[2] - eps[2]).reshape(-1),
            np.argwhere(z <= mins[2] + eps[2]).reshape(-1),
            np.argwhere(np.logical_and(x >= mids[0], z >= mids[2])).reshape(-1),
            np.argwhere(np.logical_and(x < mids[0], z >= mids[2])).reshape(-1),
        ]
    else:
        groups = [
            np.argwhere(x >= maxs[0] - eps[0]).reshape(-1),
            np.argwhere(x <= mins[0] + eps[0]).reshape(-1),
            np.argwhere(np.logical_and(x >= mids[0], y >= maxs[1] - eps[1])).reshape(-1),
            np.argwhere(np.logical_and(x < mids[0], y >= maxs[1] - eps[1])).reshape(-1),
            np.argwhere(np.logical_and(x >= mids[0], y <= mins[1] + eps[1])).reshape(-1),
            np.argwhere(np.logical_and(x < mids[0], y <= mins[1] + eps[1])).reshape(-1),
            np.argwhere(np.logical_and(x >= mids[0], z >= maxs[2] - eps[2])).reshape(-1),
            np.argwhere(np.logical_and(x < mids[0], z >= maxs[2] - eps[2])).reshape(-1),
            np.argwhere(np.logical_and(x >= mids[0], z <= mins[2] + eps[2])).reshape(-1),
            np.argwhere(np.logical_and(x < mids[0], z <= mins[2] + eps[2])).reshape(-1),
        ]

    all_indices = np.arange(coords.shape[0], dtype=np.int32)
    normalized_groups: list[npt.NDArray[np.int32]] = []
    for group in groups:
        if group.size == 0:
            normalized_groups.append(all_indices.copy())
        else:
            normalized_groups.append(np.asarray(group, dtype=np.int32))
    return normalized_groups


def dump_divide_pickle(indices: list[npt.NDArray[np.int32]], output_path: Path) -> None:
    """Serialize a list of vertex-index groups to a pickle file.

    The parent directory is created automatically if it does not exist.

    Parameters
    ----------
    indices : list of ndarray of int32
        Vertex-group index arrays.
    output_path : Path
        Destination ``.pkl`` path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [np.asarray(group, dtype=np.int32) for group in indices]
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_divide_pickle(path: Path) -> list[npt.NDArray[np.int32]]:
    """Load a list of vertex-index groups from a pickle file.

    Parameters
    ----------
    path : Path
        Source ``.pkl`` path.

    Returns
    -------
    list of ndarray of int32
        Vertex-group index arrays.

    Raises
    ------
    ValueError
        If the loaded object is not a ``list``.
    """
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Mesh divide asset must be a list, got {type(raw)!r}.")
    return [np.asarray(group, dtype=np.int32).reshape(-1) for group in raw]


def validate_divide_indices(indices: list[npt.NDArray[np.int32]], vertex_count: int, name: str) -> None:
    """Validate a list of vertex-index groups.

    Checks that every group is a non-empty 1-D array whose entries are
    valid indices into a vertex array of size *vertex_count*.

    Parameters
    ----------
    indices : list of ndarray of int32
        Vertex-group index arrays.
    vertex_count : int
        Number of vertices in the source mesh.
    name : str
        Human-readable name for error messages.

    Raises
    ------
    ValueError
        If any group is empty, not 1-D, or contains out-of-bounds indices.
    """
    if len(indices) == 0:
        raise ValueError(f"{name} must contain at least one mesh index group.")
    for group_index, group in enumerate(indices):
        if group.ndim != 1:
            raise ValueError(f"{name}[{group_index}] must be a 1-D index array.")
        if group.size == 0:
            raise ValueError(f"{name}[{group_index}] is empty.")
        if np.any(group < 0) or np.any(group >= vertex_count):
            min_index = int(group.min()) if group.size else -1
            max_index = int(group.max()) if group.size else -1
            raise ValueError(
                f"{name}[{group_index}] has invalid vertex indices: min={min_index}, max={max_index}, vertex_count={vertex_count}."
            )


def write_ascii_ply(mesh: MeshData, output_path: Path) -> None:
    """Write a mesh to an ASCII PLY file.

    The parent directory is created automatically if it does not exist.
    Vertices are stored as ``float`` properties and faces as
    ``uchar int`` lists.

    Parameters
    ----------
    mesh : MeshData
        Mesh to write.
    output_path : Path
        Destination ``.ply`` path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for x, y, z in vertices:
            handle.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
        for i0, i1, i2 in faces:
            handle.write(f"3 {int(i0)} {int(i1)} {int(i2)}\n")


def read_mesh(path: Path) -> MeshData:
    """Read a mesh from a PLY or OBJ file.

    The format is inferred from the file extension.

    Parameters
    ----------
    path : Path
        Source file path (``.ply`` or ``.obj``).

    Returns
    -------
    MeshData
        Parsed mesh.

    Raises
    ------
    ValueError
        If the file extension is not ``.ply`` or ``.obj``.
    """
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return read_ply(path)
    if suffix == ".obj":
        return read_obj(path)
    raise ValueError(f"Unsupported mesh format '{suffix}'. Expected .ply or .obj.")


def read_obj(path: Path) -> MeshData:
    """Read a Wavefront OBJ file into a MeshData.

    Supports ``v`` (vertex) and ``f`` (face) lines. Face entries with
    texture/normal indices (``v/t/n`` or ``v//n``) are handled by
    extracting only the vertex index. Polygonal faces with more than 3
    vertices are triangulated via fan triangulation.

    Parameters
    ----------
    path : Path
        Source ``.obj`` path.

    Returns
    -------
    MeshData
        Parsed mesh.

    Raises
    ------
    ValueError
        If the mesh fails validation after parsing.
    """
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                raw_indices = []
                for part in parts:
                    token = part.split("/")[0]
                    raw_indices.append(int(token) - 1)
                for idx in range(1, len(raw_indices) - 1):
                    faces.append((raw_indices[0], raw_indices[idx], raw_indices[idx + 1]))
    mesh = MeshData(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int32))
    validate_mesh(mesh, f"OBJ mesh '{path}'")
    return mesh


def _parse_ply_header(path: Path) -> tuple[list[str], str, int, int, int]:
    """Parse the header of a PLY file.

    Reads header lines until ``end_header``, then returns the header
    metadata and the byte offset where vertex data begins.

    Parameters
    ----------
    path : Path
        Source ``.ply`` path.

    Returns
    -------
    tuple
        ``(header_lines, format_line, vertex_count, face_count, data_offset)``.

    Raises
    ------
    ValueError
        If the file does not start with ``ply``, does not declare a
        format, or ends before ``end_header``.
    """
    header_lines: list[str] = []
    with path.open("rb") as handle:
        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise ValueError(f"PLY file '{path}' ended before 'end_header'.")
            line = line_bytes.decode("ascii", errors="strict").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        data_offset = handle.tell()

    if not header_lines or header_lines[0] != "ply":
        raise ValueError(f"File '{path}' is not a valid PLY file (missing 'ply' magic).")

    format_line = next((line for line in header_lines if line.startswith("format ")), None)
    if format_line is None:
        raise ValueError(f"PLY file '{path}' does not declare a format.")

    vertex_count = 0
    face_count = 0
    for line in header_lines:
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face "):
            face_count = int(line.split()[-1])
    return header_lines, format_line, vertex_count, face_count, data_offset


def read_ply(path: Path) -> MeshData:
    """Read a PLY file (ASCII or binary little-endian) into a MeshData.

    Parameters
    ----------
    path : Path
        Source ``.ply`` path.

    Returns
    -------
    MeshData
        Parsed mesh.

    Raises
    ------
    ValueError
        If the file has no vertices/faces, uses an unsupported format,
        or fails validation.
    """
    header_lines, format_line, vertex_count, face_count, data_offset = _parse_ply_header(path)
    if vertex_count <= 0 or face_count <= 0:
        raise ValueError(f"PLY file '{path}' must contain non-empty vertex and face elements.")

    if format_line.startswith("format ascii"):
        return _read_ascii_ply(path, vertex_count, face_count, header_lines)
    if format_line.startswith("format binary_little_endian"):
        return _read_binary_ply(path, vertex_count, face_count, data_offset)
    raise ValueError(f"Unsupported PLY format in '{path}': {format_line}.")


def _read_ascii_ply(path: Path, vertex_count: int, face_count: int, header_lines: list[str]) -> MeshData:
    """Read an ASCII-format PLY file body.

    Parameters
    ----------
    path : Path
        Source ``.ply`` path (used for error messages only).
    vertex_count : int
        Number of vertices declared in the header.
    face_count : int
        Number of faces declared in the header.
    header_lines : list of str
        Parsed header lines (used to skip past the header).

    Returns
    -------
    MeshData
        Parsed mesh.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break

        vertices: list[tuple[float, float, float]] = []
        for _ in range(vertex_count):
            parts = handle.readline().split()
            if len(parts) < 3:
                raise ValueError(f"Malformed vertex entry in '{path}'.")
            vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))

        faces: list[tuple[int, int, int]] = []
        for _ in range(face_count):
            parts = handle.readline().split()
            if len(parts) < 4:
                raise ValueError(f"Malformed face entry in '{path}'.")
            n = int(parts[0])
            if n < 3:
                continue
            raw_indices = [int(token) for token in parts[1 : n + 1]]
            for idx in range(1, len(raw_indices) - 1):
                faces.append((raw_indices[0], raw_indices[idx], raw_indices[idx + 1]))

    mesh = MeshData(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int32))
    validate_mesh(mesh, f"PLY mesh '{path}'")
    return mesh


def _read_binary_ply(path: Path, vertex_count: int, face_count: int, data_offset: int) -> MeshData:
    """Read a binary little-endian PLY file body.

    Vertices are expected as three consecutive ``double`` values per
    vertex. Faces use a 1-byte vertex count followed by ``N * 4`` bytes
    of ``uint32`` indices.

    Parameters
    ----------
    path : Path
        Source ``.ply`` path (used for error messages only).
    vertex_count : int
        Number of vertices declared in the header.
    face_count : int
        Number of faces declared in the header.
    data_offset : int
        Byte offset where vertex data begins (after ``end_header``).

    Returns
    -------
    MeshData
        Parsed mesh.
    """
    vertices = np.empty((vertex_count, 3), dtype=np.float64)
    faces: list[tuple[int, int, int]] = []

    with path.open("rb") as handle:
        handle.seek(data_offset)
        vertex_struct = struct.Struct("<ddd")
        for index in range(vertex_count):
            raw = handle.read(vertex_struct.size)
            if len(raw) != vertex_struct.size:
                raise ValueError(f"Unexpected EOF while reading vertices from '{path}'.")
            vertices[index] = vertex_struct.unpack(raw)

        for _ in range(face_count):
            raw_count = handle.read(1)
            if len(raw_count) != 1:
                raise ValueError(f"Unexpected EOF while reading faces from '{path}'.")
            n = struct.unpack("<B", raw_count)[0]
            raw_indices = handle.read(4 * n)
            if len(raw_indices) != 4 * n:
                raise ValueError(f"Unexpected EOF while reading face indices from '{path}'.")
            indices = struct.unpack(f"<{n}I", raw_indices)
            if n < 3:
                continue
            for idx in range(1, n - 1):
                faces.append((int(indices[0]), int(indices[idx]), int(indices[idx + 1])))

    mesh = MeshData(vertices=vertices, faces=np.asarray(faces, dtype=np.int32))
    validate_mesh(mesh, f"PLY mesh '{path}'")
    return mesh


def validate_mesh(mesh: MeshData, name: str) -> None:
    """Validate the structural integrity of a mesh.

    Checks that vertices and faces have the correct shape, that there
    are enough vertices/faces to form a valid mesh, that all vertex
    coordinates are finite, and that face indices are within bounds.

    Parameters
    ----------
    mesh : MeshData
        Mesh to validate.
    name : str
        Human-readable name for error messages.

    Raises
    ------
    ValueError
        If any structural check fails.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{name} vertices must have shape (N, 3), got {vertices.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{name} faces must have shape (M, 3), got {faces.shape}.")
    if vertices.shape[0] < 4:
        raise ValueError(f"{name} has too few vertices ({vertices.shape[0]}).")
    if faces.shape[0] < 4:
        raise ValueError(f"{name} has too few faces ({faces.shape[0]}).")
    if np.any(~np.isfinite(vertices)):
        raise ValueError(f"{name} contains non-finite vertices.")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError(f"{name} face indices are out of bounds for {vertices.shape[0]} vertices.")


def validate_mesh_frame_and_scale(mesh: MeshData, name: str) -> None:
    """Validate that a mesh uses a compatible coordinate frame and scale.

    Checks that the bounding box is not degenerate, that the mesh is
    not too large (incompatible scale), that the Z-origin is near the
    bottom of the mesh (bottom-centred frame), and that the XY centroid
    is near the origin.

    Parameters
    ----------
    mesh : MeshData
        Mesh to validate.
    name : str
        Human-readable name for error messages.

    Raises
    ------
    ValueError
        If any frame or scale check fails.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    size = maxs - mins

    if np.any(size <= 0.05):
        raise ValueError(f"{name} bounding-box is degenerate (size={size.tolist()}).")
    if np.any(size > 20.0):
        raise ValueError(f"{name} appears to use an incompatible scale (size={size.tolist()} meters).")
    if abs(float(mins[2])) > 2.0:
        raise ValueError(f"{name} z-origin is incompatible with bottom-centered frame (min_z={mins[2]:.3f}).")
    center_xy = (mins[:2] + maxs[:2]) / 2.0
    if float(np.linalg.norm(center_xy)) > 5.0:
        raise ValueError(
            f"{name} XY center is far from origin (center=({center_xy[0]:.3f}, {center_xy[1]:.3f})); expected near bottom-center frame."
        )


def save_perturbation(path: Path, perturbation: npt.NDArray[np.float64]) -> None:
    """Save a per-vertex perturbation tensor to a ``.npy`` file.

    The parent directory is created automatically if it does not exist.
    The array is cast to ``float32`` before saving.

    Parameters
    ----------
    path : Path
        Destination ``.npy`` path.
    perturbation : ndarray of float64 or float32, shape (N, 3)
        Per-vertex displacement vectors.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(perturbation, dtype=np.float32))


def load_perturbation(path: Path) -> npt.NDArray[np.float32]:
    """Load a per-vertex perturbation tensor from a ``.npy`` file.

    Parameters
    ----------
    path : Path
        Source ``.npy`` path.

    Returns
    -------
    ndarray of float32, shape (N, 3)
        Per-vertex displacement vectors.

    Raises
    ------
    ValueError
        If the array does not have shape (N, 3) or contains non-finite
        values.
    """
    perturbation = np.load(path)
    array = np.asarray(perturbation, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Perturbation asset '{path}' must have shape (N, 3), got {array.shape}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"Perturbation asset '{path}' contains non-finite values.")
    return array


def write_mesh(path: Path, mesh: MeshData) -> None:
    """Write a mesh to a PLY file (convenience wrapper).

    Delegates to :func:`write_ascii_ply`.

    Parameters
    ----------
    path : Path
        Destination ``.ply`` path.
    mesh : MeshData
        Mesh to write.
    """
    write_ascii_ply(mesh, path)


def copy_or_generate_mesh(mesh_input_path: Path | None, dimensions: tuple[float, float, float], preserve_aspect: bool) -> MeshData:
    """Return a mesh for the given dimensions, either by reading an
    external file or by constructing a simple box.

    When *mesh_input_path* is ``None``, a box mesh with the target
    dimensions is returned. Otherwise the external mesh is read,
    normalised to a bottom-centre frame, and scaled to the target
    dimensions.

    Parameters
    ----------
    mesh_input_path : Path or None
        Optional path to an external ``.ply`` or ``.obj`` mesh.
    dimensions : tuple of float
        Target ``(length, width, height)`` in meters.
    preserve_aspect : bool
        Whether to preserve the source mesh's aspect ratio during
        scaling.

    Returns
    -------
    MeshData
        The resulting mesh in bottom-centre frame.
    """
    if mesh_input_path is None:
        return box_mesh(*dimensions)
    mesh = read_mesh(mesh_input_path)
    mesh = normalize_bottom_center(mesh)
    return scale_to_dimensions(mesh, dimensions, preserve_aspect=preserve_aspect)


def dump_generation_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Serialize generation metadata to a pickle file.

    The parent directory is created automatically if it does not exist.
    This can be used to record the parameters used to generate an asset
    (blueprint, dimensions, source file, timestamp, etc.) for
    reproducibility.

    Parameters
    ----------
    path : Path
        Destination ``.pkl`` path.
    metadata : dict
        Arbitrary key-value metadata to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(metadata, handle)