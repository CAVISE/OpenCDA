"""Save the configuration and model identity of a run."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from omegaconf import OmegaConf

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
        Manifest containing the scenario, launch arguments and model.
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
    manifest = {
        "schema_version": 2,
        "scenario": opt.test_scenario,
        "launch": {"argv": list(argv), "replayed_from": replayed_from},
        "model": model,
        "inputs": copies,
        "status": "running",
    }
    write_json(destination / MANIFEST_FILENAME, manifest)
    return manifest


def finalize_run_manifest(output_dir: Path, *, status: str, error: str | None = None) -> None:
    """Record completion, interruption or failure without replacing run inputs.

    Parameters
    ----------
    output_dir : pathlib.Path
        Evaluation output directory.
    status : str
        Final execution status.
    error : str, optional
        Failure description, or None when no error occurred.
    """
    manifest_path = output_dir / REPRODUCIBILITY_DIRNAME / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(status=status, error=error)
    write_json(manifest_path, manifest)
