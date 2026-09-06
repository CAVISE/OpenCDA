from __future__ import annotations

import hashlib
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from opencda.core.plan.model_resolver import ModelResolutionError, resolve_bundle, resolve_runtime_models


def create_bundle(root: Path, category: str, bundle_id: str, contents: bytes = b"model") -> Path:
    bundle_root = root / category / bundle_id
    bundle_root.mkdir(parents=True)
    artifact = bundle_root / "artifact.bin"
    artifact.write_bytes(contents)
    kind = "coperception-checkpoint" if category == "coperception" else "advcp-assets"
    bundle_root.joinpath("meta.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"id: {bundle_id}",
                f"kind: {kind}",
                "artifacts:",
                "  - path: artifact.bin",
                f"    size: {len(contents)}",
                f"    sha256: {hashlib.sha256(contents).hexdigest()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return bundle_root


def test_resolve_local_bundle(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    expected = create_bundle(models_root, "coperception", "test-model")

    assert resolve_bundle("coperception", "test-model", models_root=models_root, auto_fetch=False) == expected
    assert not tmp_path.joinpath(".models.lock").exists()


def test_rejects_invalid_bundle_id(tmp_path: Path) -> None:
    with pytest.raises(ModelResolutionError, match="Invalid model bundle ID"):
        resolve_bundle("coperception", "../escape", models_root=tmp_path)


def test_rejects_corrupt_artifact(tmp_path: Path) -> None:
    bundle = create_bundle(tmp_path, "advcp", "base-car")
    bundle.joinpath("artifact.bin").write_bytes(b"corrupt")
    with pytest.raises(ModelResolutionError, match="size does not match"):
        resolve_bundle("advcp", "base-car", models_root=tmp_path, auto_fetch=False)


def test_sparse_clone_fetches_only_requested_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source"
    create_bundle(source, "coperception", "selected")
    create_bundle(source, "coperception", "not-selected")
    subprocess.run(["git", "init", "--initial-branch=main", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=OpenCDA tests", "-c", "user.email=tests@example.invalid", "commit", "-m", "test"],
        check=True,
        capture_output=True,
    )

    checkout = tmp_path / "models"
    checkout.mkdir()
    resolved = resolve_bundle(
        "coperception",
        "selected",
        models_root=checkout,
        repository_url=source.as_uri(),
        ref="main",
    )

    assert resolved.is_dir()
    assert not checkout.joinpath("coperception", "not-selected").exists()
    assert tmp_path.joinpath(".models.lock").read_text(encoding="utf-8") == ""


def test_missing_bundle_after_clone_raises_runtime_error(tmp_path: Path) -> None:
    source = tmp_path / "source"
    create_bundle(source, "coperception", "available")
    subprocess.run(["git", "init", "--initial-branch=main", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=OpenCDA tests", "-c", "user.email=tests@example.invalid", "commit", "-m", "test"],
        check=True,
        capture_output=True,
    )

    with pytest.raises(ModelResolutionError, match="was not found after fetching"):
        resolve_bundle(
            "coperception",
            "missing",
            models_root=tmp_path / "checkout",
            repository_url=source.as_uri(),
            ref="main",
        )


def test_clone_failure_raises_runtime_error(tmp_path: Path) -> None:
    with pytest.raises(ModelResolutionError, match="Git failed while fetching"):
        resolve_bundle(
            "coperception",
            "missing",
            models_root=tmp_path / "checkout",
            repository_url=(tmp_path / "does-not-exist").as_uri(),
            ref="main",
        )


def test_existing_sparse_checkout_fetches_bundle_added_to_ref(tmp_path: Path) -> None:
    source = tmp_path / "source"
    create_bundle(source, "coperception", "initial")
    subprocess.run(["git", "init", "--initial-branch=main", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=OpenCDA tests", "-c", "user.email=tests@example.invalid", "commit", "-m", "initial"],
        check=True,
        capture_output=True,
    )

    checkout = tmp_path / "checkout"
    resolve_bundle(
        "coperception",
        "initial",
        models_root=checkout,
        repository_url=source.as_uri(),
        ref="main",
    )

    create_bundle(source, "coperception", "added-later")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=OpenCDA tests", "-c", "user.email=tests@example.invalid", "commit", "-m", "add bundle"],
        check=True,
        capture_output=True,
    )

    resolved = resolve_bundle(
        "coperception",
        "added-later",
        models_root=checkout,
        repository_url=source.as_uri(),
        ref="main",
    )
    assert resolved.is_dir()


def test_runtime_resolution_preserves_explicit_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "custom-model"
    model_dir.mkdir()
    options = Namespace(
        with_coperception=True,
        with_advcp=False,
        model_dir=str(model_dir),
        model_id=None,
        models_root=None,
        models_repository=None,
        models_ref=None,
        no_auto_fetch_models=False,
    )
    resolve_runtime_models(options)
    assert options.model_dir == str(model_dir.resolve())


def test_runtime_resolution_sets_model_and_advcp_asset_directories(tmp_path: Path) -> None:
    model_dir = create_bundle(tmp_path, "coperception", "test-model")
    assets_dir = create_bundle(tmp_path, "advcp", "base-car")
    options = Namespace(
        with_coperception=True,
        with_advcp=True,
        model_dir=None,
        model_id="test-model",
        advcp_assets_id="base-car",
        models_root=str(tmp_path),
        models_repository=None,
        models_ref=None,
        no_auto_fetch_models=True,
    )
    resolve_runtime_models(options)
    assert options.model_dir == str(model_dir)
    assert options.advcp_assets_dir == str(assets_dir)
