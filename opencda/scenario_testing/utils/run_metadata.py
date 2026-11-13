"""Save the configuration, component commits and model identity of a run."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from omegaconf import OmegaConf

logger = logging.getLogger("cavise.opencda.opencda.scenario_testing.utils.run_metadata")
REPO_ROOT = Path(__file__).resolve().parents[3]
REPRODUCIBILITY_DIRNAME = "reproducibility"
MANIFEST_FILENAME = "run_manifest.json"
CONFIG_FILENAME = "config.yaml"
ADVCP_PATH_KEYS = (
    "model_path",
    "mesh_divide_path",
    "car_mesh_path",
    "car_mesh_divide_path",
    "remove_adv_shape_perturb_path",
    "remove_adv_shape_divide_path",
)


def collect_code_versions() -> dict[str, Any]:
    """Read local component commits and worktree status without network access.

    Returns
    -------
    dict
        CAVISE, OpenCDA, OpenCOOD, SUMO and Artery commit/dirty pairs. Missing
        checkouts use the corresponding COMPONENT_COMMIT environment variable;
        an unknown commit or worktree status is recorded as None.
    """
    paths = {name: REPO_ROOT.parent / name for name in ("opencood", "sumo", "artery")}
    paths.update(cavise=REPO_ROOT.parent, opencda=REPO_ROOT)
    # GitPython checks for the Git executable on import. Images without it still
    # need to record build revisions supplied through the environment.
    versions: dict[str, dict[str, str | bool | None]] = {name: {"commit": os.environ.get(f"{name.upper()}_COMMIT"), "dirty": None} for name in paths}
    try:
        from git import Repo
        from git.exc import GitError
    except ImportError:
        return versions
    for name, default_path in paths.items():
        path = Path(os.environ.get(f"{name.upper()}_WORKSPACE", str(default_path)))
        try:
            with Repo(path, search_parent_directories=False) as repo:
                if repo.working_tree_dir is not None and Path(repo.working_tree_dir).resolve() == path.resolve():
                    commit = repo.head.commit.hexsha
                    # Preserve porcelain status semantics and the command timeout.
                    dirty = bool(repo.git.status(porcelain=True, kill_after_timeout=10))
                    versions[name] = {"commit": commit, "dirty": dirty}
        except (GitError, OSError, ValueError):
            pass
    return versions


def reproducibility_dir(run_dir: Path) -> Path:
    """Locate the saved configuration and manifest of a run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Evaluation output directory or its reproducibility directory.

    Returns
    -------
    pathlib.Path
        Directory containing the reproducibility artifacts.
    """
    run_dir = Path(run_dir)
    return run_dir if (run_dir / MANIFEST_FILENAME).is_file() else run_dir / REPRODUCIBILITY_DIRNAME


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a JSON artifact after serialization completes.

    Parameters
    ----------
    path : pathlib.Path
        Destination in an existing directory.
    payload : mapping
        JSON-compatible manifest data.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_run_artifacts(
    output_dir: Path,
    *,
    opt: Any,
    argv: Iterable[str],
    merged_config: Any,
    input_files: Mapping[str, Path | None] | None = None,
    replayed_from: str | None = None,
) -> dict[str, Any]:
    """Save the resolved scenario, launch arguments and selected model inputs.

    Parameters
    ----------
    output_dir : pathlib.Path
        Evaluation output directory.
    opt : argparse.Namespace
        Runtime options after model resolution.
    argv : iterable of str
        Effective command arguments, excluding the executable and replay path.
    merged_config : omegaconf.DictConfig
        Final scenario configuration, including resolved interpolations.
    input_files : mapping of str to pathlib.Path or None, optional
        Optional AdvCP configuration to retain for replay.
    replayed_from : str, optional
        Original run directory, or None for a new experiment.

    Returns
    -------
    dict
        Manifest containing the scenario, launch arguments, commits and model.
    """
    destination = Path(output_dir) / REPRODUCIBILITY_DIRNAME
    destination.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(merged_config, destination / CONFIG_FILENAME, resolve=True)
    inputs = dict(input_files or {})
    model = None
    if getattr(opt, "with_coperception", False):
        directory = Path(opt.model_dir).resolve()
        model = {"id": getattr(opt, "_pinned_model_id", None) or getattr(opt, "model_id", None), "directory": str(directory)}
        if getattr(opt, "with_advcp", False) and getattr(opt, "advcp_assets_dir", None):
            model["advcp_assets_dir"] = str(Path(opt.advcp_assets_dir).resolve())
        inputs.update(model_config=directory / "config.yaml", model_manifest=directory / "meta.yaml")
    copies = {}
    for key, source in inputs.items():
        if source is None or not Path(source).is_file():
            continue
        source = Path(source).expanduser().resolve()
        target = destination / "inputs" / f"{key}{source.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if key == "advcp_config":
            config = yaml.safe_load(source.read_text(encoding="utf-8"))
            if isinstance(config, dict):
                for name in ADVCP_PATH_KEYS:
                    if config.get(name) is not None:
                        asset = Path(str(config[name])).expanduser()
                        config[name] = str(asset if asset.is_absolute() else (source.parent / asset).resolve())
                target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            else:
                shutil.copy2(source, target)
        else:
            # Model YAML can contain NumPy tags; copy it without deserialization.
            shutil.copy2(source, target)
        copies[key] = {"source": str(source), "copy": target.relative_to(destination).as_posix()}
    code = collect_code_versions()
    for name, revision in code.items():
        if not revision["commit"]:
            logger.warning("%s commit is unavailable; retain its revision separately before sharing this experiment.", name)
        if revision["dirty"]:
            logger.warning("%s has uncommitted changes. Commit them before recording a reproducible experiment; patches are not saved.", name)
    manifest = {
        "schema_version": 2,
        "scenario": opt.test_scenario,
        "launch": {"argv": list(argv), "replayed_from": replayed_from},
        "code": code,
        "model": model,
        "inputs": copies,
        "status": "running",
    }
    write_json(destination / MANIFEST_FILENAME, manifest)
    return manifest


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    """Read a previously saved run manifest.

    Parameters
    ----------
    run_dir : pathlib.Path
        Evaluation output or reproducibility directory.

    Returns
    -------
    dict
        Manifest containing the saved launch arguments.

    Raises
    ------
    FileNotFoundError
        If the run manifest is missing.
    """
    return json.loads((reproducibility_dir(run_dir) / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def finalize_run_manifest(output_dir: Path, *, status: str, error: str | None = None) -> None:
    """Record completion, interruption or failure without replacing run inputs.

    Parameters
    ----------
    output_dir : pathlib.Path
        Evaluation output or reproducibility directory.
    status : str
        Final execution status.
    error : str, optional
        Failure description, or None when no error occurred.
    """
    manifest = load_run_manifest(output_dir)
    manifest.update(status=status, error=error)
    write_json(reproducibility_dir(output_dir) / MANIFEST_FILENAME, manifest)


def compare_with_original(original: Mapping[str, Any]) -> list[str]:
    """Compare component commits and flag uncommitted experiment code.

    Parameters
    ----------
    original : mapping
        Saved run manifest.

    Returns
    -------
    list of str
        Commit mismatch, unavailable revision and dirty-worktree warnings.
        Legacy manifests with nested Git metadata are also accepted.
    """
    warnings = []
    current = collect_code_versions()
    for name, saved in (original.get("code") or {}).items():
        if name not in current:
            continue
        saved = saved.get("git") or saved
        revision = current[name]
        if not saved.get("commit") or not revision["commit"]:
            warnings.append(f"{name} commit is unavailable; the revision cannot be compared.")
        elif saved["commit"] != revision["commit"]:
            warnings.append(f"{name} commit differs: original {saved['commit']}, current {revision['commit']}.")
        if saved.get("dirty") or revision["dirty"]:
            warnings.append(f"{name} has uncommitted changes in the original or current run; commit them for reproducibility.")
    return warnings
