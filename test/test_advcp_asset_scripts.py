"""
CLI smoke tests for the AdvCP asset scripts in ``scripts/advcp``.

Covers ``generate_car_mesh``, ``generate_mesh_divide``,
``generate_remove_advshape_assets`` and ``validate_advcp_assets``. The
underlying library (``opencda.core.attack.advcp.utils.asset_utils``) is
tested in ``opencda/core/attack/advcp/test/test_advcp_asset_utils.py``.

All tests use temporary directories and do not modify the repository.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from opencda.core.attack.advcp.utils.asset_utils import (
    MeshData,
    box_mesh,
    copy_or_generate_mesh,
    dump_divide_pickle,
    generate_divide_indices,
    load_divide_pickle,
    read_mesh,
    save_perturbation,
    write_ascii_ply,
)


@pytest.fixture(autouse=True)
def restore_argv() -> Iterator[None]:
    """Restore ``sys.argv`` after each CLI invocation."""
    original = list(sys.argv)
    yield
    sys.argv = original


@pytest.fixture
def tmp_output() -> Iterator[Path]:
    """Yield a temporary directory path for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_box_mesh() -> MeshData:
    """A bare 8-vertex 4.3 x 1.91 x 1.26 m box mesh (Tesla Model 3)."""
    return box_mesh(4.3, 1.91, 1.26)


@pytest.fixture
def sample_car_mesh() -> MeshData:
    """The default generated (subdivided) car mesh for Tesla Model 3 dimensions."""
    return copy_or_generate_mesh(None, dimensions=(4.3, 1.91, 1.26), preserve_aspect=False)


class TestGenerateCarMeshCLI:
    """Smoke tests for generate_car_mesh CLI."""

    def test_generates_ply(self, tmp_output: Path) -> None:
        out = tmp_output / "car.ply"
        from scripts.advcp.generate_car_mesh import main

        sys.argv = ["generate_car_mesh", "--vehicle-blueprint", "vehicle.tesla.model3", "--output", str(out)]
        main()
        assert out.exists()
        mesh = read_mesh(out)
        assert mesh.vertices.shape[0] > 0

    def test_with_custom_dimensions(self, tmp_output: Path) -> None:
        out = tmp_output / "car.ply"
        sys.argv = [
            "generate_car_mesh",
            "--dimensions",
            "5.0",
            "2.0",
            "1.8",
            "--output",
            str(out),
        ]
        from scripts.advcp.generate_car_mesh import main

        main()
        mesh = read_mesh(out)
        size = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        np.testing.assert_allclose(size, [5.0, 2.0, 1.8], atol=1e-6)


class TestGenerateMeshDivideCLI:
    """Smoke tests for generate_mesh_divide CLI."""

    def test_generates_spoof_pkl(self, sample_car_mesh: MeshData, tmp_output: Path) -> None:
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_car_mesh, mesh_path)
        out = tmp_output / "divide.pkl"
        sys.argv = ["generate_mesh_divide", "--mesh", str(mesh_path), "--mode", "spoof", "--output", str(out)]
        from scripts.advcp.generate_mesh_divide import main

        main()
        assert out.exists()
        loaded = load_divide_pickle(out)
        assert len(loaded) == 8

    def test_generates_remove_pkl(self, sample_car_mesh: MeshData, tmp_output: Path) -> None:
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_car_mesh, mesh_path)
        out = tmp_output / "divide.pkl"
        sys.argv = ["generate_mesh_divide", "--mesh", str(mesh_path), "--mode", "remove", "--output", str(out)]
        from scripts.advcp.generate_mesh_divide import main

        main()
        loaded = load_divide_pickle(out)
        assert len(loaded) == 10

    def test_rejects_divide_with_empty_pieces(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        """A bare 8-vertex box cannot be split into 8 non-empty spoof pieces."""
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_box_mesh, mesh_path)
        out = tmp_output / "divide.pkl"
        sys.argv = ["generate_mesh_divide", "--mesh", str(mesh_path), "--mode", "spoof", "--output", str(out)]
        from scripts.advcp.generate_mesh_divide import main

        with pytest.raises(ValueError, match="no complete triangles"):
            main()
        assert not out.exists()


class TestGenerateRemoveAdvshapeAssetsCLI:
    """Smoke tests for generate_remove_advshape_assets CLI."""

    def test_generates_divide(self, tmp_output: Path) -> None:
        out = tmp_output / "divide.pkl"
        sys.argv = [
            "generate_remove_advshape_assets",
            "--mode",
            "divide",
            "--output",
            str(out),
        ]
        from scripts.advcp.generate_remove_advshape_assets import main

        main()
        assert out.exists()
        loaded = load_divide_pickle(out)
        assert len(loaded) == 10

    def test_generates_perturb(self, tmp_output: Path) -> None:
        out = tmp_output / "perturb.npy"
        sys.argv = [
            "generate_remove_advshape_assets",
            "--mode",
            "perturb",
            "--output",
            str(out),
        ]
        from scripts.advcp.generate_remove_advshape_assets import main

        main()
        assert out.exists()
        loaded = np.load(out)
        assert loaded.shape[1] == 3

    def test_generates_random_perturb(self, tmp_output: Path) -> None:
        out = tmp_output / "perturb.npy"
        sys.argv = [
            "generate_remove_advshape_assets",
            "--mode",
            "perturb",
            "--random",
            "--seed",
            "42",
            "--perturb-scale",
            "0.3",
            "--output",
            str(out),
        ]
        from scripts.advcp.generate_remove_advshape_assets import main

        main()
        loaded = np.load(out)
        assert loaded.min() >= -0.3
        assert loaded.max() <= 0.3

    def test_generates_both(self, tmp_output: Path) -> None:
        div_out = tmp_output / "divide.pkl"
        pert_out = tmp_output / "perturb.npy"
        sys.argv = [
            "generate_remove_advshape_assets",
            "--mode",
            "both",
            "--divide-output",
            str(div_out),
            "--perturb-output",
            str(pert_out),
        ]
        from scripts.advcp.generate_remove_advshape_assets import main

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
            "--car-mesh",
            str(mesh_path),
        ]
        from scripts.advcp.validate_advcp_assets import main

        main()  # should not raise

    def test_validate_all_assets(self, sample_car_mesh: MeshData, tmp_output: Path) -> None:
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_car_mesh, mesh_path)
        spoof_path = tmp_output / "spoof.pkl"
        dump_divide_pickle(generate_divide_indices(sample_car_mesh.vertices, "spoof"), spoof_path)
        remove_path = tmp_output / "remove.pkl"
        dump_divide_pickle(generate_divide_indices(sample_car_mesh.vertices, "remove"), remove_path)
        perturb_path = tmp_output / "perturb.npy"
        save_perturbation(perturb_path, np.zeros((sample_car_mesh.vertices.shape[0], 3), dtype=np.float32))
        sys.argv = [
            "validate_advcp_assets",
            "--car-mesh",
            str(mesh_path),
            "--spoof-divide",
            str(spoof_path),
            "--remove-divide",
            str(remove_path),
            "--remove-perturb",
            str(perturb_path),
        ]
        from scripts.advcp.validate_advcp_assets import main

        main()

    def test_validate_fails_on_empty_spoof_pieces(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_box_mesh, mesh_path)
        spoof_path = tmp_output / "spoof.pkl"
        dump_divide_pickle(generate_divide_indices(sample_box_mesh.vertices, "spoof"), spoof_path)
        sys.argv = [
            "validate_advcp_assets",
            "--car-mesh",
            str(mesh_path),
            "--spoof-divide",
            str(spoof_path),
        ]
        from scripts.advcp.validate_advcp_assets import main

        with pytest.raises(SystemExit):
            main()

    def test_validate_fails_on_missing_file(self, tmp_output: Path) -> None:
        sys.argv = [
            "validate_advcp_assets",
            "--car-mesh",
            str(tmp_output / "nonexistent.ply"),
        ]
        from scripts.advcp.validate_advcp_assets import main

        with pytest.raises(SystemExit):
            main()

    def test_validate_fails_on_wrong_vertex_count(self, sample_box_mesh: MeshData, tmp_output: Path) -> None:
        mesh_path = tmp_output / "mesh.ply"
        write_ascii_ply(sample_box_mesh, mesh_path)
        sys.argv = [
            "validate_advcp_assets",
            "--car-mesh",
            str(mesh_path),
            "--expected-vertices",
            "9999",
        ]
        from scripts.advcp.validate_advcp_assets import main

        with pytest.raises(SystemExit):
            main()
