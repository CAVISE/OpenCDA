"""
Unit tests for the AdvCP asset generation and validation utilities.

Tests cover:

- :class:`MeshData` construction and properties.
- Mesh generation: :func:`box_mesh`, :func:`subdivide_midpoint`,
  :func:`advshape_template_mesh`.
- Mesh transforms: :func:`normalize_bottom_center`,
  :func:`scale_to_dimensions`, :func:`copy_or_generate_mesh`.
- Divide-index generation: :func:`generate_divide_indices` for both
  ``"spoof"`` and ``"remove"`` modes.
- Mesh I/O round-trips: :func:`write_ascii_ply` / :func:`read_mesh`
  for ASCII PLY, binary PLY, and OBJ formats.
- Pickle I/O round-trips: :func:`dump_divide_pickle` /
  :func:`load_divide_pickle`.
- Perturbation I/O round-trips: :func:`save_perturbation` /
  :func:`load_perturbation`.
- Validation: :func:`validate_mesh`, :func:`validate_mesh_frame_and_scale`,
  :func:`validate_divide_indices` (pass and fail cases).
- Blueprint dimension lookup: :func:`blueprint_dimensions_m`,
  :func:`parse_dimensions_arg`.
- Runtime asset helper: :class:`AdvCPRuntimeAssetHelper` spoof and
  removal asset generation (skipped on Python < 3.10 due to
  ``TypeAlias`` dependency in ``types.py``).

CLI smoke tests for the scripts in ``scripts/advcp`` live in
``test/test_advcp_asset_scripts.py``.

All tests use temporary directories and do not modify the repository.
"""

from __future__ import annotations

import hashlib
import pickle
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from opencda.core.attack.advcp.utils.asset_utils import (
    MeshData,
    advshape_default_divide,
    advshape_template_mesh,
    blueprint_dimensions_m,
    box_mesh,
    copy_or_generate_mesh,
    divide_piece_face_counts,
    dump_divide_pickle,
    dump_generation_metadata,
    generate_divide_indices,
    load_divide_pickle,
    load_perturbation,
    normalize_bottom_center,
    parse_dimensions_arg,
    read_mesh,
    read_obj,
    read_ply,
    save_perturbation,
    scale_to_dimensions,
    subdivide_midpoint,
    validate_divide_indices,
    validate_divide_pieces,
    validate_mesh,
    validate_mesh_frame_and_scale,
    write_ascii_ply,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def tmp_output() -> Path:
    """Yield a temporary directory path for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_box_mesh() -> MeshData:
    """A standard 4.3 x 1.91 x 1.26 m box mesh (Tesla Model 3)."""
    return box_mesh(4.3, 1.91, 1.26)


@pytest.fixture
def sample_car_mesh() -> MeshData:
    """The default generated (subdivided) car mesh for Tesla Model 3 dimensions."""
    return copy_or_generate_mesh(None, dimensions=(4.3, 1.91, 1.26), preserve_aspect=False)


@pytest.fixture
def sample_vertices() -> np.ndarray:
    """8 vertices of a unit cube at the origin."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def minimal_valid_mesh() -> MeshData:
    """A tetrahedron with 4 vertices and 4 faces — the minimum for validation."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]], dtype=np.int32)
    return MeshData(vertices=vertices, faces=faces)


# =========================================================================
# MeshData
# =========================================================================


class TestMeshData:
    """MeshData construction and basic properties."""

    def test_construct(self, sample_box_mesh: MeshData) -> None:
        assert sample_box_mesh.vertices.shape == (8, 3)
        assert sample_box_mesh.faces.shape == (12, 3)
        assert sample_box_mesh.vertices.dtype == np.float64
        assert sample_box_mesh.faces.dtype == np.int32

    def test_immutable(self, sample_box_mesh: MeshData) -> None:
        with pytest.raises((TypeError, AttributeError)):
            sample_box_mesh.vertices = np.zeros((4, 3))  # type: ignore[misc]


# =========================================================================
# Blueprint dimensions
# =========================================================================


class TestBlueprintDimensions:
    """CARLA blueprint dimension lookup."""

    def test_known_blueprint(self) -> None:
        assert blueprint_dimensions_m("vehicle.tesla.model3") == (4.30, 1.91, 1.26)
        assert blueprint_dimensions_m("vehicle.audi.a2") == (3.70, 1.70, 1.55)

    def test_unknown_blueprint_falls_back_to_default(self) -> None:
        dims = blueprint_dimensions_m("vehicle.unknown.foo")
        assert dims == (4.30, 1.91, 1.26)

    def test_none_blueprint_falls_back_to_default(self) -> None:
        dims = blueprint_dimensions_m(None)
        assert dims == (4.30, 1.91, 1.26)


class TestParseDimensionsArg:
    """Dimension argument parsing."""

    def test_valid(self) -> None:
        assert parse_dimensions_arg([4.0, 2.0, 1.5]) == (4.0, 2.0, 1.5)

    def test_none(self) -> None:
        assert parse_dimensions_arg(None) is None

    def test_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="exactly 3"):
            parse_dimensions_arg([1.0, 2.0])

    def test_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            parse_dimensions_arg([4.0, -1.0, 1.5])


# =========================================================================
# Mesh generation
# =========================================================================


class TestBoxMesh:
    """Box mesh construction."""

    def test_dimensions(self) -> None:
        mesh = box_mesh(4.0, 2.0, 1.5)
        mins = mesh.vertices.min(axis=0)
        maxs = mesh.vertices.max(axis=0)
        size = maxs - mins
        np.testing.assert_allclose(size, [4.0, 2.0, 1.5], atol=1e-10)

    def test_bottom_at_zero(self) -> None:
        mesh = box_mesh(4.0, 2.0, 1.5)
        assert mesh.vertices[:, 2].min() == 0.0

    def test_xy_centered(self) -> None:
        mesh = box_mesh(4.0, 2.0, 1.5)
        center_xy = mesh.vertices.mean(axis=0)[:2]
        np.testing.assert_allclose(center_xy, [0.0, 0.0], atol=1e-10)

    def test_vertex_count(self, sample_box_mesh: MeshData) -> None:
        assert sample_box_mesh.vertices.shape[0] == 8

    def test_face_count(self, sample_box_mesh: MeshData) -> None:
        assert sample_box_mesh.faces.shape[0] == 12


class TestSubdivideMidpoint:
    """Midpoint subdivision."""

    def test_zero_levels_returns_original(self, sample_box_mesh: MeshData) -> None:
        result = subdivide_midpoint(sample_box_mesh, levels=0)
        assert result.vertices.shape == sample_box_mesh.vertices.shape
        assert result.faces.shape == sample_box_mesh.faces.shape

    def test_one_level_quadruples_faces(self, sample_box_mesh: MeshData) -> None:
        result = subdivide_midpoint(sample_box_mesh, levels=1)
        # 12 faces * 4 = 48
        assert result.faces.shape[0] == 48

    def test_two_levels(self, sample_box_mesh: MeshData) -> None:
        result = subdivide_midpoint(sample_box_mesh, levels=2)
        # 12 * 4 * 4 = 192
        assert result.faces.shape[0] == 192

    def test_vertices_are_finite(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        result = subdivide_midpoint(mesh, levels=3)
        assert np.all(np.isfinite(result.vertices))


class TestAdvShapeTemplateMesh:
    """AdvCP adversarial-shape template mesh."""

    def test_returns_valid_mesh(self) -> None:
        mesh = advshape_template_mesh()
        validate_mesh(mesh, "advshape_template")

    def test_vertex_count(self) -> None:
        mesh = advshape_template_mesh()
        assert 50 < mesh.vertices.shape[0] < 500

    def test_face_count(self) -> None:
        mesh = advshape_template_mesh()
        assert mesh.faces.shape[0] == 192  # 12 * 4 * 4


class TestOpen3DCompatibility:
    """Index compatibility with the Open3D meshes built by the AdvCP runtime.

    Open3D is mocked in the unit-test environment, so the expected values
    below were captured from ``open3d`` 0.18.0 and 0.19.0
    (``TriangleMesh.create_box`` + ``subdivide_midpoint``).
    """

    # Open3D ``create_box(1, 1, 1)`` vertex and triangle layout.
    O3D_UNIT_BOX_VERTICES = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    O3D_UNIT_BOX_TRIANGLES = [
        [4, 7, 5],
        [4, 6, 7],
        [0, 2, 4],
        [2, 6, 4],
        [0, 1, 2],
        [1, 3, 2],
        [1, 5, 7],
        [1, 7, 3],
        [2, 3, 7],
        [2, 7, 6],
        [0, 4, 1],
        [1, 4, 5],
    ]
    # First vertices / triangles created by Open3D ``subdivide_midpoint(1)`` on the unit box.
    O3D_UNIT_BOX_SUBDIV1_NEW_VERTICES = [[0.5, 1.0, 0.5], [1.0, 1.0, 0.5], [0.5, 1.0, 0.0], [0.0, 1.0, 0.5], [0.5, 1.0, 1.0], [0.0, 0.0, 0.5]]
    O3D_UNIT_BOX_SUBDIV1_TRIANGLES = [[4, 8, 10], [8, 7, 9], [9, 5, 10], [8, 9, 10], [4, 11, 8], [11, 6, 12], [12, 7, 8], [11, 12, 8]]
    # sha256 of the Open3D adv-shape template (create_box(4.9, 2.5, 2.0) centred in XY, subdivide_midpoint(2)).
    O3D_ADVSHAPE_TEMPLATE_SHA256 = "18eadf9af6766859aa88b9ef1f51472aaac6dd7eb2aabf110d919d3d528cc113"

    @staticmethod
    def _shift_to_origin_corner(mesh: MeshData, length: float, width: float) -> np.ndarray:
        return mesh.vertices + np.array([length / 2.0, width / 2.0, 0.0])

    def test_box_layout_matches_open3d(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        np.testing.assert_allclose(self._shift_to_origin_corner(mesh, 1.0, 1.0), self.O3D_UNIT_BOX_VERTICES)
        np.testing.assert_array_equal(mesh.faces, self.O3D_UNIT_BOX_TRIANGLES)

    def test_box_normals_point_outward(self, sample_box_mesh: MeshData) -> None:
        tri = sample_box_mesh.vertices[sample_box_mesh.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        center = np.array([0.0, 0.0, sample_box_mesh.vertices[:, 2].max() / 2.0])
        outward = tri.mean(axis=1) - center
        assert np.all(np.sum(normals * outward, axis=1) > 0)

    def test_subdivision_order_matches_open3d(self) -> None:
        mesh = subdivide_midpoint(box_mesh(1.0, 1.0, 1.0), levels=1)
        vertices = self._shift_to_origin_corner(mesh, 1.0, 1.0)
        np.testing.assert_allclose(vertices[8:14], self.O3D_UNIT_BOX_SUBDIV1_NEW_VERTICES)
        np.testing.assert_array_equal(mesh.faces[:8], self.O3D_UNIT_BOX_SUBDIV1_TRIANGLES)

    def test_advshape_template_matches_open3d(self) -> None:
        template = advshape_template_mesh()
        digest = hashlib.sha256(np.round(template.vertices, 6).astype("<f8").tobytes() + template.faces.astype("<i4").tobytes()).hexdigest()
        assert digest == self.O3D_ADVSHAPE_TEMPLATE_SHA256

    def test_generated_remove_divide_matches_runtime_default(self) -> None:
        """Generated removal divide must equal the partition the runtime uses by default."""
        template = advshape_template_mesh()
        generated = generate_divide_indices(template.vertices, "remove")
        runtime_default = advshape_default_divide(template.vertices)
        assert len(generated) == len(runtime_default) == 10
        for generated_group, runtime_group in zip(generated, runtime_default):
            np.testing.assert_array_equal(generated_group, runtime_group)


# =========================================================================
# Mesh transforms
# =========================================================================


class TestNormalizeBottomCenter:
    """Bottom-centre normalisation."""

    def test_centers_xy(self) -> None:
        vertices = np.array([[-1.0, -2.0, 0.0], [3.0, 4.0, 5.0]], dtype=np.float64)
        faces = np.array([[0, 1, 0]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces)
        result = normalize_bottom_center(mesh)
        # XY center should be at origin
        center_xy = result.vertices.mean(axis=0)[:2]
        np.testing.assert_allclose(center_xy, [0.0, 0.0], atol=1e-10)

    def test_bottom_at_zero(self) -> None:
        vertices = np.array([[0.0, 0.0, 2.0], [1.0, 1.0, 5.0]], dtype=np.float64)
        faces = np.array([[0, 1, 0]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces)
        result = normalize_bottom_center(mesh)
        assert result.vertices[:, 2].min() == 0.0


class TestScaleToDimensions:
    """Mesh scaling."""

    def test_scale_to_target(self, sample_box_mesh: MeshData) -> None:
        target = (5.0, 2.0, 1.8)
        scaled = scale_to_dimensions(sample_box_mesh, target, preserve_aspect=False)
        size = scaled.vertices.max(axis=0) - scaled.vertices.min(axis=0)
        np.testing.assert_allclose(size, target, atol=1e-10)

    def test_preserve_aspect(self, sample_box_mesh: MeshData) -> None:
        """When preserve_aspect=True, the smallest scale factor is applied uniformly.

        For a 4.3 x 1.91 x 1.26 box scaled to (10, 10, 10):
        target/size = [10/4.3, 10/1.91, 10/1.26] ≈ [2.326, 5.236, 7.937]
        min = 2.326, so result ≈ [10.0, 4.44, 2.93].
        """
        target = (10.0, 10.0, 10.0)
        scaled = scale_to_dimensions(sample_box_mesh, target, preserve_aspect=True)
        size = scaled.vertices.max(axis=0) - scaled.vertices.min(axis=0)
        # The smallest original axis (Z=1.26) determines the uniform scale
        # scale = min(10/4.3, 10/1.91, 10/1.26) = 10/4.3 ≈ 2.326
        expected_scale = 10.0 / 4.3
        np.testing.assert_allclose(size[0], 4.3 * expected_scale, atol=1e-10)
        np.testing.assert_allclose(size[1], 1.91 * expected_scale, atol=1e-10)
        np.testing.assert_allclose(size[2], 1.26 * expected_scale, atol=1e-10)

    def test_degenerate_mesh_raises(self) -> None:
        vertices = np.array([[0.0, 0.0, 0.0], [1e-12, 0.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 0]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces)
        with pytest.raises(ValueError, match="degenerate"):
            scale_to_dimensions(mesh, (4.0, 2.0, 1.5))


class TestCopyOrGenerateMesh:
    """Mesh copy-or-generate logic."""

    def test_no_input_creates_box(self) -> None:
        mesh = copy_or_generate_mesh(None, dimensions=(4.0, 2.0, 1.5), preserve_aspect=False)
        size = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        np.testing.assert_allclose(size, [4.0, 2.0, 1.5], atol=1e-10)

    def test_with_ply_input(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        src = tmp_output / "source.ply"
        write_ascii_ply(sample_box_mesh, src)
        mesh = copy_or_generate_mesh(src, dimensions=(5.0, 2.5, 2.0), preserve_aspect=False)
        size = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        np.testing.assert_allclose(size, [5.0, 2.5, 2.0], atol=1e-10)


# =========================================================================
# Divide-index generation
# =========================================================================


class TestGenerateDivideIndices:
    """Vertex-group index generation."""

    def test_spoof_returns_8_groups(self, sample_vertices: np.ndarray) -> None:
        groups = generate_divide_indices(sample_vertices, "spoof")
        assert len(groups) == 8

    def test_remove_returns_10_groups(self, sample_vertices: np.ndarray) -> None:
        groups = generate_divide_indices(sample_vertices, "remove")
        assert len(groups) == 10

    def test_all_indices_are_valid(self, sample_vertices: np.ndarray) -> None:
        groups = generate_divide_indices(sample_vertices, "spoof")
        n = sample_vertices.shape[0]
        for g in groups:
            assert g.dtype == np.int32
            assert g.ndim == 1
            assert g.min() >= 0
            assert g.max() < n

    def test_invalid_mode_raises(self, sample_vertices: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unsupported divide mode"):
            generate_divide_indices(sample_vertices, "invalid")

    def test_spoof_groups_cover_all_vertices(self, sample_vertices: np.ndarray) -> None:
        groups = generate_divide_indices(sample_vertices, "spoof")
        all_idx = np.unique(np.concatenate(groups))
        assert all_idx.shape[0] == sample_vertices.shape[0]

    def test_remove_groups_cover_all_vertices(self, sample_vertices: np.ndarray) -> None:
        groups = generate_divide_indices(sample_vertices, "remove")
        all_idx = np.unique(np.concatenate(groups))
        assert all_idx.shape[0] == sample_vertices.shape[0]

    def test_empty_group_fallback(self) -> None:
        """When a group is empty, it should fall back to all indices."""
        # Single vertex at origin — most groups will be empty
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        groups = generate_divide_indices(vertices, "spoof")
        for g in groups:
            assert g.shape[0] == 1
            assert g[0] == 0


class TestDividePieces:
    """Every divide group must yield a non-empty mesh piece."""

    def test_default_car_mesh_spoof_pieces_non_empty(self, sample_car_mesh: MeshData) -> None:
        counts = divide_piece_face_counts(generate_divide_indices(sample_car_mesh.vertices, "spoof"), sample_car_mesh.faces)
        assert len(counts) == 8
        assert all(count > 0 for count in counts)

    def test_template_remove_pieces_non_empty(self) -> None:
        template = advshape_template_mesh()
        counts = divide_piece_face_counts(generate_divide_indices(template.vertices, "remove"), template.faces)
        assert counts == [32, 32, 16, 16, 16, 16, 16, 16, 16, 16]

    def test_bare_box_spoof_pieces_rejected(self, sample_box_mesh: MeshData) -> None:
        groups = generate_divide_indices(sample_box_mesh.vertices, "spoof")
        assert divide_piece_face_counts(groups, sample_box_mesh.faces)[6:] == [0, 0]
        with pytest.raises(ValueError, match=r"groups \[6, 7\] contain no complete triangles"):
            validate_divide_pieces(groups, sample_box_mesh.faces, "Spoof divide")

    def test_validate_passes(self, sample_car_mesh: MeshData) -> None:
        groups = generate_divide_indices(sample_car_mesh.vertices, "spoof")
        validate_divide_pieces(groups, sample_car_mesh.faces, "Spoof divide")

    def test_default_generated_mesh_is_subdivided(self, sample_car_mesh: MeshData) -> None:
        assert sample_car_mesh.vertices.shape == (98, 3)
        assert sample_car_mesh.faces.shape == (192, 3)


# =========================================================================
# Mesh I/O
# =========================================================================


class TestWriteAsciiPly:
    """ASCII PLY writing."""

    def test_writes_valid_ply(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        path = tmp_output / "test.ply"
        write_ascii_ply(sample_box_mesh, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_ply_header(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        path = tmp_output / "test.ply"
        write_ascii_ply(sample_box_mesh, path)
        with open(path, "r") as f:
            header = f.read(500)
        assert header.startswith("ply")
        assert "format ascii 1.0" in header
        assert "element vertex 8" in header
        assert "element face 12" in header

    def test_round_trip(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        path = tmp_output / "test.ply"
        write_ascii_ply(sample_box_mesh, path)
        mesh = read_mesh(path)
        assert mesh.vertices.shape == sample_box_mesh.vertices.shape
        assert mesh.faces.shape == sample_box_mesh.faces.shape
        np.testing.assert_allclose(mesh.vertices, sample_box_mesh.vertices, atol=1e-8)


class TestReadPly:
    """PLY reading (ASCII and binary)."""

    def test_read_ascii_ply(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        path = tmp_output / "ascii.ply"
        write_ascii_ply(sample_box_mesh, path)
        mesh = read_ply(path)
        assert mesh.vertices.shape == (8, 3)
        assert mesh.faces.shape == (12, 3)

    def test_read_binary_ply(self, tmp_output: Path) -> None:
        """Write a minimal binary PLY with 4 vertices and 4 faces, then read it back."""
        path = tmp_output / "binary.ply"
        # A tetrahedron: 4 vertices, 4 faces
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]], dtype=np.int32)
        with open(path, "wb") as f:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(b"element vertex 4\n")
            f.write(b"property double x\n")
            f.write(b"property double y\n")
            f.write(b"property double z\n")
            f.write(b"element face 4\n")
            f.write(b"property list uchar uint vertex_indices\n")
            f.write(b"end_header\n")
            for v in vertices:
                f.write(struct.pack("<ddd", *v))
            for face in faces:
                f.write(struct.pack("<BIII", 3, *face))
        mesh = read_ply(path)
        assert mesh.vertices.shape == (4, 3)
        assert mesh.faces.shape == (4, 3)
        np.testing.assert_allclose(mesh.vertices, vertices)

    def test_invalid_format_raises(self, tmp_output: Path) -> None:
        path = tmp_output / "bad.ply"
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\nend_header\n")
        with pytest.raises(ValueError, match="non-empty"):
            read_ply(path)


class TestReadPlyHeaderDriven:
    """PLY parsing driven by the declared property types and order."""

    @staticmethod
    def _write_binary_ply(
        path: Path,
        vertex_props: list[tuple[str, str]],
        vertex_rows: list[tuple[Any, ...]],
        faces: list[list[int]],
        byte_order: str = "<",
        count_type: str = "uchar",
        index_type: str = "int",
        face_extra: tuple[str, str] | None = None,
    ) -> None:
        codes = {"float": "f", "double": "d", "uchar": "B", "int": "i", "uint": "I", "ushort": "H", "short": "h"}
        fmt_name = "binary_little_endian" if byte_order == "<" else "binary_big_endian"
        header = ["ply", f"format {fmt_name} 1.0", "comment test", f"element vertex {len(vertex_rows)}"]
        header += [f"property {ptype} {pname}" for pname, ptype in vertex_props]
        header += [f"element face {len(faces)}", f"property list {count_type} {index_type} vertex_indices"]
        if face_extra is not None:
            header.append(f"property {face_extra[1]} {face_extra[0]}")
        header.append("end_header")
        with open(path, "wb") as handle:
            handle.write(("\n".join(header) + "\n").encode("ascii"))
            vertex_fmt = byte_order + "".join(codes[ptype] for _, ptype in vertex_props)
            for row in vertex_rows:
                handle.write(struct.pack(vertex_fmt, *row))
            for face in faces:
                handle.write(struct.pack(byte_order + codes[count_type], len(face)))
                handle.write(struct.pack(byte_order + codes[index_type] * len(face), *face))
                if face_extra is not None:
                    handle.write(struct.pack(byte_order + codes[face_extra[1]], 7))

    TETRA_XYZ = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    TETRA_FACES = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]]

    def test_binary_float32_with_normals_and_colors(self, tmp_output: Path) -> None:
        path = tmp_output / "float_normals_colors.ply"
        props = [("x", "float"), ("y", "float"), ("z", "float"), ("nx", "float"), ("ny", "float"), ("nz", "float")]
        props += [("red", "uchar"), ("green", "uchar"), ("blue", "uchar")]
        rows = [(*xyz, 0.0, 0.0, 1.0, 255, 128, 0) for xyz in self.TETRA_XYZ]
        self._write_binary_ply(path, props, rows, self.TETRA_FACES)
        mesh = read_ply(path)
        np.testing.assert_allclose(mesh.vertices, self.TETRA_XYZ)
        np.testing.assert_array_equal(mesh.faces, self.TETRA_FACES)

    def test_binary_big_endian(self, tmp_output: Path) -> None:
        path = tmp_output / "big_endian.ply"
        props = [("x", "double"), ("y", "double"), ("z", "double")]
        self._write_binary_ply(path, props, self.TETRA_XYZ, self.TETRA_FACES, byte_order=">", index_type="uint")
        mesh = read_ply(path)
        np.testing.assert_allclose(mesh.vertices, self.TETRA_XYZ)
        np.testing.assert_array_equal(mesh.faces, self.TETRA_FACES)

    def test_binary_property_order_and_face_extras(self, tmp_output: Path) -> None:
        """Coordinates declared out of order, extra face property after the index list."""
        path = tmp_output / "reordered.ply"
        props = [("z", "float"), ("confidence", "float"), ("x", "float"), ("y", "float")]
        rows = [(z, 0.5, x, y) for x, y, z in self.TETRA_XYZ]
        self._write_binary_ply(path, props, rows, self.TETRA_FACES, count_type="ushort", face_extra=("material", "uchar"))
        mesh = read_ply(path)
        np.testing.assert_allclose(mesh.vertices, self.TETRA_XYZ)
        np.testing.assert_array_equal(mesh.faces, self.TETRA_FACES)

    def test_binary_mixed_polygons_are_triangulated(self, tmp_output: Path) -> None:
        path = tmp_output / "quads.ply"
        props = [("x", "float"), ("y", "float"), ("z", "float")]
        xyz = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5, 1.0)]
        faces = [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
        self._write_binary_ply(path, props, xyz, faces)
        mesh = read_ply(path)
        assert mesh.faces.shape == (6, 3)
        np.testing.assert_array_equal(mesh.faces[:2], [[0, 1, 2], [0, 2, 3]])

    def test_ascii_polygons_and_extra_properties(self, tmp_output: Path) -> None:
        path = tmp_output / "ascii_quads.ply"
        with open(path, "w") as handle:
            handle.write("ply\nformat ascii 1.0\nelement vertex 5\nproperty float x\nproperty float y\nproperty float z\n")
            handle.write("property uchar red\nelement face 5\nproperty list uchar int vertex_indices\nproperty int flags\nend_header\n")
            handle.write("0 0 0 1\n1 0 0 2\n1 1 0 3\n0 1 0 4\n0.5 0.5 1 5\n")
            handle.write("4 0 1 2 3 9\n3 0 1 4 9\n3 1 2 4 9\n3 2 3 4 9\n3 3 0 4 9\n")
        mesh = read_ply(path)
        assert mesh.vertices.shape == (5, 3)
        assert mesh.faces.shape == (6, 3)

    def test_truncated_binary_raises(self, tmp_output: Path) -> None:
        path = tmp_output / "truncated.ply"
        props = [("x", "float"), ("y", "float"), ("z", "float")]
        self._write_binary_ply(path, props, self.TETRA_XYZ, self.TETRA_FACES)
        path.write_bytes(path.read_bytes()[:-6])
        with pytest.raises(ValueError, match="Unexpected EOF"):
            read_ply(path)

    def test_missing_coordinates_raises(self, tmp_output: Path) -> None:
        path = tmp_output / "no_z.ply"
        props = [("x", "float"), ("y", "float"), ("w", "float")]
        self._write_binary_ply(path, props, self.TETRA_XYZ, self.TETRA_FACES)
        with pytest.raises(ValueError, match="x, y and z"):
            read_ply(path)

    def test_unknown_property_type_raises(self, tmp_output: Path) -> None:
        path = tmp_output / "bad_type.ply"
        path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty quad x\nend_header\n0\n")
        with pytest.raises(ValueError, match="unsupported property type"):
            read_ply(path)


class TestReadObj:
    """OBJ file reading."""

    def test_read_simple_obj(self, tmp_output: Path) -> None:
        """A tetrahedron with 4 vertices and 4 faces."""
        path = tmp_output / "test.obj"
        with open(path, "w") as f:
            f.write("v 0 0 0\n")
            f.write("v 1 0 0\n")
            f.write("v 0 1 0\n")
            f.write("v 0 0 1\n")
            f.write("f 1 2 3\n")
            f.write("f 1 3 4\n")
            f.write("f 1 4 2\n")
            f.write("f 2 3 4\n")
        mesh = read_obj(path)
        assert mesh.vertices.shape == (4, 3)
        assert mesh.faces.shape == (4, 3)

    def test_read_obj_with_texture_coords(self, tmp_output: Path) -> None:
        """Face entries with texture coordinates (v/t) should be handled."""
        path = tmp_output / "tex.obj"
        with open(path, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n")
            f.write("f 1/1 2/2 3/3\n")
            f.write("f 1/1 3/3 4/4\n")
            f.write("f 1/1 4/4 2/2\n")
            f.write("f 2/2 3/3 4/4\n")
        mesh = read_obj(path)
        assert mesh.vertices.shape == (4, 3)
        assert mesh.faces.shape == (4, 3)

    def test_read_obj_with_normals(self, tmp_output: Path) -> None:
        """Face entries with texture and normal (v/t/n) should be handled."""
        path = tmp_output / "norm.obj"
        with open(path, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n")
            f.write("f 1/1/1 2/2/2 3/3/3\n")
            f.write("f 1/1/1 3/3/3 4/4/4\n")
            f.write("f 1/1/1 4/4/4 2/2/2\n")
            f.write("f 2/2/2 3/3/3 4/4/4\n")
        mesh = read_obj(path)
        assert mesh.vertices.shape == (4, 3)
        assert mesh.faces.shape == (4, 3)

    def test_read_obj_triangulates_quads(self, tmp_output: Path) -> None:
        """Quad faces should be triangulated into two triangles.

        A hexahedron (box) with 8 vertices and 6 quad faces → 12 triangles.
        """
        path = tmp_output / "quad.obj"
        with open(path, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n")
            f.write("v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n")
            f.write("f 1 2 3 4\n")  # bottom
            f.write("f 5 8 7 6\n")  # top
            f.write("f 1 5 6 2\n")  # front
            f.write("f 2 6 7 3\n")  # right
            f.write("f 3 7 8 4\n")  # back
            f.write("f 4 8 5 1\n")  # left
        mesh = read_obj(path)
        assert mesh.vertices.shape == (8, 3)
        # 6 quads → 12 triangles
        assert mesh.faces.shape[0] == 12


class TestReadMesh:
    """Format-dispatching mesh reader."""

    def test_unsupported_format_raises(self, tmp_output: Path) -> None:
        path = tmp_output / "test.fbx"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported mesh format"):
            read_mesh(path)


# =========================================================================
# Pickle I/O
# =========================================================================


class TestDividePickle:
    """Mesh-divide pickle serialisation."""

    def test_round_trip(self, sample_vertices: np.ndarray, tmp_output: Path) -> None:
        groups = generate_divide_indices(sample_vertices, "spoof")
        path = tmp_output / "divide.pkl"
        dump_divide_pickle(groups, path)
        loaded = load_divide_pickle(path)
        assert len(loaded) == len(groups)
        for a, b in zip(loaded, groups):
            np.testing.assert_array_equal(a, b)

    def test_invalid_pickle_raises(self, tmp_output: Path) -> None:
        path = tmp_output / "bad.pkl"
        with open(path, "wb") as f:
            pickle.dump("not_a_list", f)
        with pytest.raises(ValueError, match="must be a list"):
            load_divide_pickle(path)


# =========================================================================
# Perturbation I/O
# =========================================================================


class TestPerturbation:
    """Perturbation tensor save/load."""

    def test_round_trip(self, tmp_output: Path) -> None:
        path = tmp_output / "perturb.npy"
        original = np.random.randn(100, 3).astype(np.float32)
        save_perturbation(path, original)
        loaded = load_perturbation(path)
        np.testing.assert_allclose(loaded, original, atol=1e-6)

    def test_wrong_shape_raises(self, tmp_output: Path) -> None:
        path = tmp_output / "bad.npy"
        bad = np.zeros((100, 4), dtype=np.float32)
        np.save(path, bad)
        with pytest.raises(ValueError, match="must have shape"):
            load_perturbation(path)

    def test_non_finite_raises(self, tmp_output: Path) -> None:
        path = tmp_output / "nan.npy"
        bad = np.full((10, 3), np.nan, dtype=np.float32)
        np.save(path, bad)
        with pytest.raises(ValueError, match="non-finite"):
            load_perturbation(path)


# =========================================================================
# Validation
# =========================================================================


class TestValidateMesh:
    """Mesh structural validation."""

    def test_valid_mesh_passes(self, sample_box_mesh: MeshData) -> None:
        validate_mesh(sample_box_mesh, "test")  # should not raise

    def test_wrong_vertex_shape_raises(self) -> None:
        mesh = MeshData(vertices=np.zeros((10, 2)), faces=np.zeros((5, 3), dtype=np.int32))
        with pytest.raises(ValueError, match="must have shape"):
            validate_mesh(mesh, "test")

    def test_wrong_face_shape_raises(self) -> None:
        mesh = MeshData(vertices=np.zeros((10, 3)), faces=np.zeros((5, 4), dtype=np.int32))
        with pytest.raises(ValueError, match="must have shape"):
            validate_mesh(mesh, "test")

    def test_too_few_vertices_raises(self) -> None:
        mesh = MeshData(vertices=np.zeros((2, 3)), faces=np.zeros((4, 3), dtype=np.int32))
        with pytest.raises(ValueError, match="too few vertices"):
            validate_mesh(mesh, "test")

    def test_too_few_faces_raises(self) -> None:
        mesh = MeshData(vertices=np.zeros((10, 3)), faces=np.zeros((2, 3), dtype=np.int32))
        with pytest.raises(ValueError, match="too few faces"):
            validate_mesh(mesh, "test")

    def test_non_finite_vertices_raises(self, minimal_valid_mesh: MeshData) -> None:
        """Replace one vertex with NaN; should trigger the non-finite check."""
        vertices = minimal_valid_mesh.vertices.copy()
        vertices[0, 0] = np.nan
        mesh = MeshData(vertices=vertices, faces=minimal_valid_mesh.faces)
        with pytest.raises(ValueError, match="non-finite"):
            validate_mesh(mesh, "test")

    def test_out_of_bounds_faces_raises(self, minimal_valid_mesh: MeshData) -> None:
        """Face index 10 is out of bounds for 4 vertices."""
        faces = minimal_valid_mesh.faces.copy()
        faces[0, 0] = 10
        mesh = MeshData(vertices=minimal_valid_mesh.vertices, faces=faces)
        with pytest.raises(ValueError, match="out of bounds"):
            validate_mesh(mesh, "test")


class TestValidateMeshFrameAndScale:
    """Mesh coordinate frame and scale validation."""

    def test_valid_mesh_passes(self, sample_box_mesh: MeshData) -> None:
        validate_mesh_frame_and_scale(sample_box_mesh, "test")

    def test_degenerate_raises(self) -> None:
        vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces)
        with pytest.raises(ValueError, match="degenerate"):
            validate_mesh_frame_and_scale(mesh, "test")

    def test_too_large_raises(self) -> None:
        vertices = np.array([[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces)
        with pytest.raises(ValueError, match="incompatible scale"):
            validate_mesh_frame_and_scale(mesh, "test")

    def test_z_origin_not_bottom_raises(self) -> None:
        vertices = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5], [0, 0, 6]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces)
        with pytest.raises(ValueError, match="z-origin"):
            validate_mesh_frame_and_scale(mesh, "test")


class TestValidateDivideIndices:
    """Divide-index validation."""

    def test_valid_indices_passes(self, sample_vertices: np.ndarray) -> None:
        groups = generate_divide_indices(sample_vertices, "spoof")
        validate_divide_indices(groups, sample_vertices.shape[0], "test")

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            validate_divide_indices([], 10, "test")

    def test_out_of_bounds_raises(self) -> None:
        groups = [np.array([0, 1, 100], dtype=np.int32)]
        with pytest.raises(ValueError, match="invalid vertex indices"):
            validate_divide_indices(groups, 10, "test")

    def test_negative_index_raises(self) -> None:
        groups = [np.array([0, -1, 2], dtype=np.int32)]
        with pytest.raises(ValueError, match="invalid vertex indices"):
            validate_divide_indices(groups, 10, "test")

    def test_empty_group_raises(self) -> None:
        groups = [np.array([], dtype=np.int32)]
        with pytest.raises(ValueError, match="empty"):
            validate_divide_indices(groups, 10, "test")


# =========================================================================
# Generation metadata
# =========================================================================


class TestDumpGenerationMetadata:
    """Generation metadata serialisation."""

    def test_round_trip(self, tmp_output: Path) -> None:
        path = tmp_output / "meta.pkl"
        meta = {"blueprint": "vehicle.tesla.model3", "dimensions": (4.3, 1.91, 1.26)}
        dump_generation_metadata(path, meta)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == meta


# =========================================================================
# Runtime asset helper (Python >= 3.10 only due to TypeAlias in types.py)
# =========================================================================

_RUNTIME_ASSETS_AVAILABLE: bool = True
try:
    from opencda.core.attack.advcp.utils.runtime_assets import AdvCPRuntimeAssetHelper
except ImportError:
    _RUNTIME_ASSETS_AVAILABLE = False


@pytest.mark.skipif(
    not _RUNTIME_ASSETS_AVAILABLE,
    reason="AdvCPRuntimeAssetHelper requires Python >= 3.10 (TypeAlias in types.py)",
)
class TestAdvCPRuntimeAssetHelper:
    """Runtime asset generation helper."""

    def test_ensure_spoof_assets_generates_missing(self, tmp_output: Path) -> None:
        config: dict[str, Any] = {
            "car_mesh_path": str(tmp_output / "car_mesh.ply"),
            "car_mesh_divide_path": str(tmp_output / "spoof" / "car_mesh_divide.pkl"),
            "vehicle_blueprint": "vehicle.tesla.model3",
            "asset_runtime_generation": True,
        }
        mesh_path, divide_path = AdvCPRuntimeAssetHelper.ensure_spoof_assets(config)
        assert mesh_path.exists()
        assert divide_path.exists()
        divide = load_divide_pickle(divide_path)
        assert len(divide) == 8

    def test_ensure_spoof_assets_uses_cache(self, tmp_output: Path) -> None:
        mesh_path = tmp_output / "car_mesh.ply"
        divide_path = tmp_output / "spoof" / "car_mesh_divide.pkl"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        box = box_mesh(4.3, 1.91, 1.26)
        write_ascii_ply(box, mesh_path)
        dump_divide_pickle(generate_divide_indices(box.vertices, "spoof"), divide_path)
        mtime_before = mesh_path.stat().st_mtime
        config: dict[str, Any] = {
            "car_mesh_path": str(mesh_path),
            "car_mesh_divide_path": str(divide_path),
            "asset_runtime_generation": True,
        }
        AdvCPRuntimeAssetHelper.ensure_spoof_assets(config)
        assert mesh_path.stat().st_mtime == mtime_before

    def test_ensure_spoof_assets_disabled(self, tmp_output: Path) -> None:
        config: dict[str, Any] = {
            "car_mesh_path": str(tmp_output / "car_mesh.ply"),
            "car_mesh_divide_path": str(tmp_output / "spoof" / "car_mesh_divide.pkl"),
            "asset_runtime_generation": False,
        }
        mesh_path, divide_path = AdvCPRuntimeAssetHelper.ensure_spoof_assets(config)
        assert not mesh_path.exists()
        assert not divide_path.exists()

    def test_ensure_remove_advshape_assets_no_cache_dir(self) -> None:
        config: dict[str, Any] = {
            "asset_runtime_generation": True,
        }
        result = AdvCPRuntimeAssetHelper.ensure_remove_advshape_assets(config)
        assert result is None

    def test_ensure_remove_advshape_assets_generates(self, tmp_output: Path) -> None:
        cache_dir = tmp_output / "cache"
        config: dict[str, Any] = {
            "asset_runtime_generation": True,
            "asset_cache_dir": str(cache_dir),
            "remove_adv_shape_generate_zero_perturb": True,
        }
        result = AdvCPRuntimeAssetHelper.ensure_remove_advshape_assets(config)
        assert result is not None
        perturb_path, divide_path = result
        assert divide_path.exists()
        assert perturb_path.exists()
        divide = load_divide_pickle(divide_path)
        assert len(divide) == 10

    def test_ensure_remove_advshape_assets_updates_config(self, tmp_output: Path) -> None:
        cache_dir = tmp_output / "cache"
        config: dict[str, Any] = {
            "asset_runtime_generation": True,
            "asset_cache_dir": str(cache_dir),
        }
        AdvCPRuntimeAssetHelper.ensure_remove_advshape_assets(config)
        assert "remove_adv_shape_divide_path" in config
        assert Path(config["remove_adv_shape_divide_path"]).exists()


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_vertices_divide(self) -> None:
        vertices = np.empty((0, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            generate_divide_indices(vertices, "spoof")

    def test_single_vertex_divide(self) -> None:
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        groups = generate_divide_indices(vertices, "spoof")
        assert len(groups) == 8
        for g in groups:
            assert g.shape[0] == 1

    def test_ply_with_extra_properties(self, tmp_output: Path) -> None:
        """PLY with extra vertex properties (e.g. normal) should still be readable."""
        path = tmp_output / "extra.ply"
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex 4\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("element face 4\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            f.write("0 0 0 1 0 0\n")
            f.write("1 0 0 0 1 0\n")
            f.write("0 1 0 0 0 1\n")
            f.write("0 0 1 0 0 1\n")
            f.write("3 0 1 2\n")
            f.write("3 0 2 3\n")
            f.write("3 0 3 1\n")
            f.write("3 1 2 3\n")
        mesh = read_ply(path)
        assert mesh.vertices.shape == (4, 3)
        assert mesh.faces.shape == (4, 3)

    def test_scale_to_dimensions_preserve_aspect_smaller(self) -> None:
        """When preserve_aspect is True, the smallest axis determines the scale."""
        mesh = box_mesh(4.0, 2.0, 1.0)
        target = (8.0, 8.0, 8.0)
        scaled = scale_to_dimensions(mesh, target, preserve_aspect=True)
        size = scaled.vertices.max(axis=0) - scaled.vertices.min(axis=0)
        # target/size = [8/4, 8/2, 8/1] = [2, 4, 8], min = 2
        # uniform scale = 2, result = [8, 4, 2]
        np.testing.assert_allclose(size, [8.0, 4.0, 2.0], atol=1e-10)
