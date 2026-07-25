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
- CLI integration smoke tests for ``generate_car_mesh``,
  ``generate_mesh_divide``, ``generate_remove_advshape_assets``,
  and ``validate_advcp_assets``.
- Runtime asset helper: :class:`AdvCPRuntimeAssetHelper` spoof and
  removal asset generation (skipped on Python < 3.10 due to
  ``TypeAlias`` dependency in ``types.py``).

All tests use temporary directories and do not modify the repository.
"""

from __future__ import annotations

import pickle
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from opencda.core.attack.advcp.utils.asset_utils import (
    MeshData,
    advshape_template_mesh,
    blueprint_dimensions_m,
    box_mesh,
    copy_or_generate_mesh,
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
    validate_mesh,
    validate_mesh_frame_and_scale,
    write_ascii_ply,
    write_mesh,
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
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
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
# CLI integration smoke tests
# =========================================================================


class TestGenerateCarMeshCLI:
    """Smoke tests for generate_car_mesh CLI."""

    def test_generates_ply(self, tmp_output: Path) -> None:
        out = tmp_output / "car.ply"
        from opencda.core.attack.advcp.utils.generate_car_mesh import main
        sys.argv = ["generate_car_mesh", "--vehicle-blueprint", "vehicle.tesla.model3", "--output", str(out)]
        main()
        assert out.exists()
        mesh = read_mesh(out)
        assert mesh.vertices.shape[0] > 0

    def test_with_custom_dimensions(self, tmp_output: Path) -> None:
        out = tmp_output / "car.ply"
        sys.argv = [
            "generate_car_mesh",
            "--dimensions", "5.0", "2.0", "1.8",
            "--output", str(out),
        ]
        from opencda.core.attack.advcp.utils.generate_car_mesh import main
        main()
        mesh = read_mesh(out)
        size = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        np.testing.assert_allclose(size, [5.0, 2.0, 1.8], atol=1e-6)


class TestGenerateMeshDivideCLI:
    """Smoke tests for generate_mesh_divide CLI."""

    def test_generates_spoof_pkl(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_box_mesh, mesh_path)
        out = tmp_output / "divide.pkl"
        sys.argv = ["generate_mesh_divide", "--mesh", str(mesh_path), "--mode", "spoof", "--output", str(out)]
        from opencda.core.attack.advcp.utils.generate_mesh_divide import main
        main()
        assert out.exists()
        loaded = load_divide_pickle(out)
        assert len(loaded) == 8

    def test_generates_remove_pkl(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_box_mesh, mesh_path)
        out = tmp_output / "divide.pkl"
        sys.argv = ["generate_mesh_divide", "--mesh", str(mesh_path), "--mode", "remove", "--output", str(out)]
        from opencda.core.attack.advcp.utils.generate_mesh_divide import main
        main()
        loaded = load_divide_pickle(out)
        assert len(loaded) == 10


class TestGenerateRemoveAdvshapeAssetsCLI:
    """Smoke tests for generate_remove_advshape_assets CLI."""

    def test_generates_divide(self, tmp_output: Path) -> None:
        out = tmp_output / "divide.pkl"
        sys.argv = [
            "generate_remove_advshape_assets",
            "--mode", "divide",
            "--output", str(out),
        ]
        from opencda.core.attack.advcp.utils.generate_remove_advshape_assets import main
        main()
        assert out.exists()
        loaded = load_divide_pickle(out)
        assert len(loaded) == 10

    def test_generates_perturb(self, tmp_output: Path) -> None:
        out = tmp_output / "perturb.npy"
        sys.argv = [
            "generate_remove_advshape_assets",
            "--mode", "perturb",
            "--output", str(out),
        ]
        from opencda.core.attack.advcp.utils.generate_remove_advshape_assets import main
        main()
        assert out.exists()
        loaded = np.load(out)
        assert loaded.shape[1] == 3

    def test_generates_random_perturb(self, tmp_output: Path) -> None:
        out = tmp_output / "perturb.npy"
        sys.argv = [
            "generate_remove_advshape_assets",
            "--mode", "perturb",
            "--random",
            "--seed", "42",
            "--perturb-scale", "0.3",
            "--output", str(out),
        ]
        from opencda.core.attack.advcp.utils.generate_remove_advshape_assets import main
        main()
        loaded = np.load(out)
        assert loaded.min() >= -0.3
        assert loaded.max() <= 0.3

    def test_generates_both(self, tmp_output: Path) -> None:
        div_out = tmp_output / "divide.pkl"
        pert_out = tmp_output / "perturb.npy"
        sys.argv = [
            "generate_remove_advshape_assets",
            "--mode", "both",
            "--divide-output", str(div_out),
            "--perturb-output", str(pert_out),
        ]
        from opencda.core.attack.advcp.utils.generate_remove_advshape_assets import main
        main()
        assert div_out.exists()
        assert pert_out.exists()


class TestValidateAdvcpAssetsCLI:
    """Smoke tests for validate_advcp_assets CLI."""

    def test_validate_car_mesh_only(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_box_mesh, mesh_path)
        sys.argv = [
            "validate_advcp_assets",
            "--car-mesh", str(mesh_path),
        ]
        from opencda.core.attack.advcp.utils.validate_advcp_assets import main
        main()  # should not raise

    def test_validate_all_assets(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_box_mesh, mesh_path)
        spoof_path = tmp_output / "spoof.pkl"
        dump_divide_pickle(generate_divide_indices(sample_box_mesh.vertices, "spoof"), spoof_path)
        remove_path = tmp_output / "remove.pkl"
        dump_divide_pickle(generate_divide_indices(sample_box_mesh.vertices, "remove"), remove_path)
        perturb_path = tmp_output / "perturb.npy"
        save_perturbation(perturb_path, np.zeros((sample_box_mesh.vertices.shape[0], 3), dtype=np.float32))
        sys.argv = [
            "validate_advcp_assets",
            "--car-mesh", str(mesh_path),
            "--spoof-divide", str(spoof_path),
            "--remove-divide", str(remove_path),
            "--remove-perturb", str(perturb_path),
        ]
        from opencda.core.attack.advcp.utils.validate_advcp_assets import main
        main()

    def test_validate_fails_on_missing_file(self, tmp_output: Path) -> None:
        sys.argv = [
            "validate_advcp_assets",
            "--car-mesh", str(tmp_output / "nonexistent.ply"),
        ]
        from opencda.core.attack.advcp.utils.validate_advcp_assets import main
        with pytest.raises(SystemExit):
            main()

    def test_validate_fails_on_wrong_vertex_count(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_box_mesh, mesh_path)
        sys.argv = [
            "validate_advcp_assets",
            "--car-mesh", str(mesh_path),
            "--expected-vertices", "9999",
        ]
        from opencda.core.attack.advcp.utils.validate_advcp_assets import main
        with pytest.raises(SystemExit):
            main()


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