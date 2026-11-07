"""Persist run configuration, metadata and prediction records.

Artifacts are stored alongside the evaluation report.
"""

from __future__ import annotations

import enum
import hashlib
import importlib.util
import json
import logging
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger("cavise.opencda.opencda.scenario_testing.utils.run_metadata")

SCHEMA_VERSION = 1
REPRODUCIBILITY_DIRNAME = "reproducibility"
MANIFEST_FILENAME = "run_manifest.json"
REPRODUCE_SCRIPT_FILENAME = "reproduce.sh"
CONFIG_FILENAME = "config.yaml"
INPUTS_DIRNAME = "inputs"
PREDICTIONS_DIRNAME = "predictions"
CODE_DIFF_FILENAME = "code_diff.patch"

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG_FILENAME = "config.yaml"
# Paths resolved relative to the AdvCP configuration directory.
ADVCP_PATH_KEYS = ("car_mesh_path", "car_mesh_divide_path", "remove_adv_shape_perturb_path", "remove_adv_shape_divide_path")
CHECKPOINT_SUFFIXES = (".pth", ".pt", ".ckpt", ".bin", ".safetensors")
TRACKED_PACKAGES = ("numpy", "torch", "torchvision", "carla", "omegaconf", "open3d", "opencood", "spconv", "scipy", "PyYAML")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _to_jsonable(value: Any) -> Any:
    """Convert option values to JSON-compatible types."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


# Credentials must never be stored in shared simulation artifacts.
REDACTED = "[REDACTED]"
SECRET_KEY_PATTERN = re.compile(
    r"(?:password|passwd|token|secret|api[_-]?key|access[_-]?key|authorization|credential|private[_-]?key|(?:^|[_-])auth(?:$|[_-]))",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
SECRET_QUERY_KEYS = frozenset(
    {
        "token",
        "access_token",
        "auth",
        "authorization",
        "password",
        "passwd",
        "secret",
        "api_key",
        "apikey",
        "client_secret",
        "signature",
        "sig",
    }
)


def _redact_url(url: str) -> str:
    """Remove HTTPS basic-auth credentials and known sensitive query values."""
    from urllib.parse import parse_qsl, urlencode

    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return url
        host = parsed.netloc.rsplit("@", 1)[-1] if "@" in parsed.netloc else parsed.netloc
        # Removing userinfo instead of inserting a placeholder avoids persisting
        # something that looks like a reusable credential.
        query = urlencode(
            [
                (key, REDACTED if SECRET_KEY_PATTERN.search(key) or key.lower() in SECRET_QUERY_KEYS else value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            ]
        )
        return urlunsplit((parsed.scheme, host, parsed.path, query, parsed.fragment))
    except ValueError:
        return REDACTED


def _redact_string(value: str) -> str:
    return URL_PATTERN.sub(lambda match: _redact_url(match.group(0)), value)


def _redact_argv(argv: Iterable[str]) -> list[str]:
    """Redact secrets in --flag=value and --flag value forms."""
    original = list(argv)
    safe: list[str] = []
    redact_next = False
    for item in original:
        if redact_next:
            safe.append(REDACTED)
            redact_next = False
        elif item.startswith("--") and "=" in item:
            key, value = item.split("=", 1)
            safe.append(f"{key}={REDACTED}" if SECRET_KEY_PATTERN.search(key) else f"{key}={_redact_string(value)}")
        elif item.startswith("--") and SECRET_KEY_PATTERN.search(item):
            safe.append(item)
            redact_next = True
        else:
            safe.append(_redact_string(item))
    return safe


def _redact_metadata(value: Any, key: str = "") -> Any:
    """Recursively scrub URLs and explicitly sensitive fields in a manifest."""
    if key in ("argv", "raw_argv") and isinstance(value, (list, tuple)):
        return _redact_argv(value)
    if SECRET_KEY_PATTERN.search(key):
        return REDACTED if value is not None else None
    if isinstance(value, Mapping):
        return {str(k): _redact_metadata(v, str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_redact_metadata(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo: Path, *arguments: str, preserve_whitespace: bool = False) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run([git, "-C", str(repo), *arguments], check=True, capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout if preserve_whitespace else result.stdout.strip()


def collect_git_info(repo: Path) -> dict[str, Any] | None:
    """Collect commit, branch, remote and worktree status."""
    commit = _run_git(repo, "rev-parse", "HEAD")
    if not commit:
        return None
    status = _run_git(repo, "status", "--porcelain", "--untracked-files=no")
    untracked = _run_git(repo, "ls-files", "--others", "--exclude-standard")
    return {
        "path": str(repo),
        "commit": commit,
        "branch": _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "remote": _redact_string(_run_git(repo, "config", "--get", "remote.origin.url") or ""),
        "dirty": bool(status),
        # Untracked files are not included in the diff; record their names.
        "untracked_files": untracked.splitlines() if untracked else [],
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _module_location(module_name: str) -> Path | None:
    """Find where a module lives without importing it."""
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file:
        return Path(module_file).resolve().parent
    # OpenCOOD might not be imported yet when the manifest is written.
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        return Path(spec.origin).resolve().parent
    return None


def _module_repo(module_name: str) -> Path | None:
    """Find the Git checkout backing a module."""
    location = _module_location(module_name)
    if location is None:
        return None
    toplevel = _run_git(location, "rev-parse", "--show-toplevel")
    return Path(toplevel) if toplevel else None


def collect_source_digest() -> str:
    """Hash OpenCDA source files independently of Git metadata."""
    digest = hashlib.sha256()
    extensions = {
        ".py",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".xml",
        ".proto",
        ".cfg",
        ".ini",
        ".txt",
        ".sh",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".cu",
        ".cuh",
        ".so",
    }

    paths = [REPO_ROOT / "opencda.py"]

    for directory in (REPO_ROOT / "opencda", REPO_ROOT / "AIM"):
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file() and path.suffix in extensions and "__pycache__" not in path.parts)

    for source in sorted(paths):
        if not source.is_file():
            continue
        relative_path = source.relative_to(REPO_ROOT).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\\0")
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)

    return digest.hexdigest()


def _digest_source_directory(directory: Path) -> str | None:
    """Fingerprint installed package sources even when Git is unavailable."""
    source_files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix
        in {
            ".py",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".cfg",
            ".ini",
            ".proto",
            ".sh",
            ".txt",
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hpp",
            ".cu",
            ".cuh",
            ".so",
        }
        and "__pycache__" not in path.parts
    )
    if not source_files:
        return None
    digest = hashlib.sha256()
    for source in source_files:
        digest.update(source.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def collect_code_versions() -> dict[str, Any]:
    from opencda.version import __version__

    versions: dict[str, Any] = {
        "opencda": {
            "version": __version__,
            "git": collect_git_info(REPO_ROOT),
            "source_digest": collect_source_digest(),
        },
        "python": sys.version.split()[0],
        "packages": {name: _package_version(name) for name in TRACKED_PACKAGES},
    }

    opencood_location = _module_location("opencood")
    if opencood_location is not None:
        opencood_repo = _module_repo("opencood")
        if opencood_repo is None or opencood_repo.resolve() != REPO_ROOT:
            versions["opencood"] = {
                "git": collect_git_info(opencood_repo) if opencood_repo else None,
                "source_digest": _digest_source_directory(opencood_location),
            }
    return versions


def collect_device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info

    try:
        cuda_available = bool(torch.cuda.is_available())
        torch_info: dict[str, Any] = {
            "device": "cuda" if cuda_available else "cpu",
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
        if cuda_available:
            torch_info["gpus"] = [
                {"index": index, "name": torch.cuda.get_device_name(index), "capability": list(torch.cuda.get_device_capability(index))}
                for index in range(torch.cuda.device_count())
            ]
    except Exception as error:  # Device detection failure must not interrupt a run.
        torch_info = {"error": repr(error)}
    info["torch"] = torch_info
    return info


def repository_web_url(remote: str | None) -> str | None:
    """Convert a Git remote into a browser URL."""
    if not remote:
        return None
    url = remote.strip()
    if url.startswith("git@") and ":" in url:
        host, path = url[len("git@") :].split(":", 1)
        url = f"https://{host}/{path}"
    elif url.startswith("ssh://git@"):
        url = "https://" + url[len("ssh://git@") :]
    if not url.startswith(("https://", "http://")):
        return None
    return url.removesuffix("/").removesuffix(".git")


def _bundle_commit(repo_root: Path, bundle_path: str) -> str | None:
    """Find a commit matching the local model bundle.

    Checks HEAD and FETCH_HEAD; returns None if neither matches.
    """
    for ref in ("HEAD", "FETCH_HEAD"):
        commit = _run_git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        if not commit:
            continue
        tracked = _run_git(repo_root, "ls-tree", "-r", "--name-only", commit, "--", bundle_path)
        if not tracked:
            continue
        unchanged = _run_git(repo_root, "diff", "--quiet", commit, "--", bundle_path) is not None
        untracked = _run_git(repo_root, "ls-files", "--others", "--exclude-standard", "--", bundle_path)
        if unchanged and not untracked:
            return commit
    return None


def _meta_checksums(bundle_dir: Path) -> dict[str, str | None]:
    """Read artifact checksums from model meta.yaml."""
    meta_path = bundle_dir / "meta.yaml"
    if not meta_path.is_file():
        return {}
    import yaml

    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    artifacts = meta.get("artifacts") if isinstance(meta, dict) else None
    if not isinstance(artifacts, list):
        return {}
    return {
        str(item["path"]): (str(item["sha256"]) if item.get("sha256") else None) for item in artifacts if isinstance(item, dict) and item.get("path")
    }


def _bundle_artifacts(bundle_dir: Path) -> list[dict[str, Any]]:
    """Collect model artifacts and their actual SHA-256 hashes.

    Compare them against checksums from meta.yaml when available.
    """
    expected = _meta_checksums(bundle_dir)
    paths = set(expected)
    paths.update(path.relative_to(bundle_dir).as_posix() for path in bundle_dir.rglob("*") if path.is_file() and path.suffix in CHECKPOINT_SUFFIXES)
    # Model config.yaml affects inference even if it is absent from meta.yaml.
    if (bundle_dir / MODEL_CONFIG_FILENAME).is_file():
        paths.add(MODEL_CONFIG_FILENAME)

    artifacts: list[dict[str, Any]] = []
    for relative in sorted(paths):
        path = bundle_dir / relative
        exists = path.is_file()
        entry: dict[str, Any] = {
            "path": relative,
            "size": path.stat().st_size if exists else None,
            "sha256": sha256_file(path) if exists else None,
        }
        if relative in expected:
            entry["expected_sha256"] = expected[relative]
            entry["matches_meta"] = exists and entry["sha256"] == expected[relative]
            if not entry["matches_meta"]:
                logger.warning('Model artifact "%s" does not match the checksum in %s/meta.yaml.', relative, bundle_dir)
        artifacts.append(entry)
    return artifacts


def describe_bundle(bundle_dir_value: str | None, *, bundle_id: str | None, category: str, opt: Any) -> dict[str, Any]:
    """Describe the model bundle and its repository revision."""
    info: dict[str, Any] = {
        "id": bundle_id,
        "local_dir": bundle_dir_value,
        "models_ref": getattr(opt, "models_ref", None),
        "repository": None,
        "path_in_repository": None,
        "commit": None,
        "url": None,
        "artifacts": [],
    }
    if not bundle_dir_value or not Path(bundle_dir_value).is_dir():
        return info

    bundle_dir = Path(bundle_dir_value).resolve()
    info["artifacts"] = _bundle_artifacts(bundle_dir)

    toplevel = _run_git(bundle_dir, "rev-parse", "--show-toplevel")
    if not toplevel:
        return info  # Outside Git, the model is identified by artifact hashes only.
    repo_root = Path(toplevel)
    bundle_path = bundle_dir.relative_to(repo_root).as_posix()
    remote = _run_git(repo_root, "config", "--get", "remote.origin.url") or getattr(opt, "models_repository", None)
    web_url = repository_web_url(remote)
    commit = _bundle_commit(repo_root, bundle_path)

    info.update(
        {
            "repository": _redact_string(web_url or remote) if (web_url or remote) else None,
            "path_in_repository": bundle_path,
            "commit": commit,
            "url": _redact_string(f"{web_url}/tree/{commit}/{bundle_path}") if web_url and commit else None,
        }
    )
    if commit is None:
        logger.warning(
            'Model bundle "%s" differs from any commit of %s; it is identified only by artifact hashes.', bundle_path, _redact_string(remote or "")
        )
    return info


def collect_checkpoint_info(opt: Any) -> dict[str, Any] | None:
    """Collect checkpoint and optional AdvCP asset metadata."""
    if not getattr(opt, "with_coperception", False):
        return None

    checkpoint = describe_bundle(getattr(opt, "model_dir", None), bundle_id=getattr(opt, "model_id", None), category="coperception", opt=opt)
    if getattr(opt, "with_advcp", False):
        checkpoint["advcp_assets"] = describe_bundle(
            getattr(opt, "advcp_assets_dir", None),
            bundle_id=getattr(opt, "advcp_assets_id", None),
            category="advcp",
            opt=opt,
        )
    return checkpoint


def collect_seed(merged_config: Mapping[str, Any]) -> int | None:
    world = merged_config.get("world") or {}
    seed = world.get("seed") if isinstance(world, Mapping) else None
    return int(seed) if seed is not None else None


def reproducibility_dir(run_dir: Path) -> Path:
    """Locate the reproducibility directory for a run."""
    run_dir = Path(run_dir)
    if (run_dir / MANIFEST_FILENAME).is_file():
        return run_dir
    return run_dir / REPRODUCIBILITY_DIRNAME


def _write_reproduce_script(output_dir: Path, command: list[str]) -> str:
    script = output_dir / REPRODUCE_SCRIPT_FILENAME
    script.write_text(
        "#!/usr/bin/env bash\n"
        "# Run from the OpenCDA repository root. Additional arguments override saved ones.\n"
        "set -euo pipefail\n\n"
        "# Original command (loads the current YAML files from the repository):\n"
        f"# {shlex.join(command)}\n\n"
        "# Replay with the saved configuration, arguments, and AdvCP config.\n"
        "# The run directory may be moved or copied. Supply credentials separately.\n"
        'RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
        'python3 opencda.py --replay-from "$RUN_DIR" "$@"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return REPRODUCE_SCRIPT_FILENAME


def _redact_yaml_file(path: Path) -> bool:
    """Remove known credentials from a saved YAML copy without changing normal files."""
    import yaml

    original = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(original)
    safe = _redact_metadata(parsed)
    if safe != parsed:
        path.write_text(yaml.safe_dump(safe, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return True
    cleaned_text = _redact_string(original)
    if cleaned_text != original:
        path.write_text(cleaned_text, encoding="utf-8")
        return True
    return False


def _copy_advcp_config(source: Path, target: Path) -> None:
    """Copy an AdvCP config, making relative asset paths absolute.

    AdvCP resolves these paths against the folder of the YAML file, so a copy
    in inputs/ would point to the wrong place.
    """
    import yaml

    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        config = None
    relative_keys = [
        key
        for key in ADVCP_PATH_KEYS
        if isinstance(config, dict) and config.get(key) is not None and not Path(str(config[key])).expanduser().is_absolute()
    ]
    if not relative_keys:
        shutil.copy2(source, target)
        return
    for key in relative_keys:
        config[key] = str((source.parent / Path(str(config[key])).expanduser()).resolve())
    header = f"# Copy of {source}; relative paths ({', '.join(relative_keys)}) are resolved against its folder.\n"
    target.write_text(header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _copy_inputs(output_dir: Path, input_files: Mapping[str, Path | None], opt: Any) -> dict[str, Any]:
    inputs_dir = output_dir / INPUTS_DIRNAME
    inputs_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Any] = {}

    files = dict(input_files)
    model_dir = getattr(opt, "model_dir", None) if getattr(opt, "with_coperception", False) else None
    if model_dir:
        for name in ("config.yaml", "meta.yaml"):
            candidate = Path(model_dir) / name
            files[f"model_{Path(name).stem}"] = candidate if candidate.is_file() else None

    for key, source in files.items():
        if source is None or not Path(source).is_file():
            continue
        source = Path(source)
        target = inputs_dir / f"{key}{source.suffix}"
        if key == "advcp_config":
            _copy_advcp_config(source, target)
        else:
            shutil.copy2(source, target)
        credentials_redacted = _redact_yaml_file(target) if target.suffix.lower() in {".yaml", ".yml"} else False
        if credentials_redacted:
            logger.warning("Credentials were redacted from saved input %s; external credentials may be needed for replay.", key)
        copied[key] = {
            "source": str(source),
            "copy": target.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(target),
            "redacted": credentials_redacted,
        }
    return copied


def _save_code_diff(output_dir: Path, git_info: Mapping[str, Any] | None) -> str | None:
    if not git_info:
        return None
    if git_info.get("untracked_files"):
        # Record untracked filenames only; their contents are not captured.
        logger.warning(
            "OpenCDA has untracked files (%d): only their names are saved, not their contents. Commit them for full reproducibility.",
            len(git_info["untracked_files"]),
        )
    if not git_info.get("dirty"):
        return None
    logger.warning("OpenCDA has uncommitted changes: they are saved to %s.", CODE_DIFF_FILENAME)
    diff = _run_git(REPO_ROOT, "diff", "HEAD", "--binary", preserve_whitespace=True)
    if not diff:
        return None
    if _redact_string(diff) != diff:
        logger.warning("Git patch contains credential-bearing URLs; refusing to persist it. Commit or sanitize the changes before sharing this run.")
        return None
    (output_dir / CODE_DIFF_FILENAME).write_text(diff, encoding="utf-8")
    return CODE_DIFF_FILENAME


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def write_run_artifacts(
    output_dir: Path,
    *,
    opt: Any,
    argv: Iterable[str],
    raw_argv: Iterable[str] | None = None,
    merged_config: Any,
    current_time: str,
    input_files: Mapping[str, Path | None] | None = None,
    replayed_from: str | None = None,
) -> dict[str, Any]:
    """Save the resolved configuration and run manifest before execution."""
    from omegaconf import OmegaConf

    run_dir = Path(output_dir).resolve()
    output_dir = run_dir / REPRODUCIBILITY_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = list(argv)
    safe_argv = _redact_argv(argv)
    command = ["python3", "opencda.py", *safe_argv]

    OmegaConf.save(merged_config, output_dir / CONFIG_FILENAME, resolve=True)
    config_credentials_redacted = _redact_yaml_file(output_dir / CONFIG_FILENAME)
    if config_credentials_redacted:
        logger.warning("Credentials were removed from the saved replay config; a replay requires secure replacements.")
    resolved_config = OmegaConf.to_container(merged_config, resolve=True)

    code_versions = collect_code_versions()
    seed = collect_seed(resolved_config)
    if seed is None:
        logger.warning("world.seed is not set: numpy/random/torch are not seeded, the run is not reproducible.")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scenario": getattr(opt, "test_scenario", None),
        "current_time": current_time,
        "started_at": _now_iso(),
        "finished_at": None,
        "status": "running",
        "seed": seed,
        "config_file": CONFIG_FILENAME,
        "config_sha256": sha256_file(output_dir / CONFIG_FILENAME),
        "config_redacted": config_credentials_redacted,
        "launch": {
            "argv": safe_argv,
            "command": shlex.join(command),
            "raw_argv": _redact_argv(raw_argv) if raw_argv is not None else None,
            "reproduce_script": _write_reproduce_script(output_dir, command),
            "cwd": os.getcwd(),
            "options": _redact_metadata(vars(opt)),
            "replayed_from": replayed_from,
        },
        "code": code_versions,
        "device": collect_device_info(),
        "checkpoint": collect_checkpoint_info(opt),
        "inputs": _copy_inputs(output_dir, input_files or {}, opt),
        "code_diff": _save_code_diff(output_dir, code_versions["opencda"]["git"]),
        "outputs": {
            "report": "../report.json",
            "predictions": f"../{PREDICTIONS_DIRNAME}"
            if getattr(opt, "with_coperception", False) and getattr(opt, "save_predictions", False)
            else None,
        },
    }

    write_json(output_dir / MANIFEST_FILENAME, _redact_metadata(manifest))
    logger.info("Run manifest saved to: %s", output_dir / MANIFEST_FILENAME)
    return manifest


def finalize_run_manifest(output_dir: Path, *, status: str, duration_seconds: float | None = None, error: str | None = None) -> None:
    """Update run status if a manifest exists."""
    manifest_path = reproducibility_dir(output_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not update run manifest %s", manifest_path)
        return

    manifest.update({"status": status, "finished_at": _now_iso(), "duration_seconds": duration_seconds, "error": error})
    write_json(manifest_path, manifest)


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = reproducibility_dir(run_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found: this folder is not a reproducible OpenCDA run.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_prediction_records(run_dir: Path) -> list[dict[str, Any]]:  # noqa: DC02
    """Load per-tick boxes and scores, using None for missing tensors."""
    import numpy as np

    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / PREDICTIONS_DIRNAME).glob("tick_*.npz")):
        with np.load(path) as data:
            record: dict[str, Any] = {"tick": int(data["tick"])}
            for key in data.files:
                if key == "tick" or key.startswith("has_"):
                    continue
                record[key] = data[key] if bool(data.get(f"has_{key}", True)) else None
        records.append(record)
    return records


def _model_input_hashes(checkpoint: Mapping[str, Any] | None) -> dict[str, str | None]:
    """Collect hashes of every model file used at runtime, including AdvCP assets."""
    if not checkpoint:
        return {}
    hashes = {f"coperception/{item['path']}": item.get("sha256") for item in checkpoint.get("artifacts", [])}
    advcp_assets = checkpoint.get("advcp_assets") or {}
    hashes.update({f"advcp/{item['path']}": item.get("sha256") for item in advcp_assets.get("artifacts", [])})
    return hashes


def _device_signature(device: Mapping[str, Any] | None) -> dict[str, Any]:
    """Device details that may change inference results."""
    torch_info = (device or {}).get("torch") or {}
    return {
        "device": torch_info.get("device"),
        "gpus": [gpu.get("name") for gpu in torch_info.get("gpus") or []],
        "cuda": torch_info.get("cuda_version"),
        "cudnn": torch_info.get("cudnn_version"),
    }


def compare_with_original(original: Mapping[str, Any], opt: Any) -> list[str]:
    """Check for differences from the original run."""
    warnings: list[str] = []

    original_code = original.get("code") or {}
    current_code = collect_code_versions()

    original_python = original_code.get("python")
    current_python = current_code.get("python")
    if original_python and original_python != current_python:
        warnings.append(f"Python version differs: original {original_python}, current {current_python}.")

    original_opencda = original_code.get("opencda") or {}
    current_opencda = current_code.get("opencda") or {}

    original_digest = original_opencda.get("source_digest")
    current_digest = current_opencda.get("source_digest")

    if original_digest and original_digest != current_digest:
        warnings.append("OpenCDA source digest differs from the original run.")

    original_opencood = original_code.get("opencood") or {}
    current_opencood = current_code.get("opencood") or {}
    original_opencood_digest = original_opencood.get("source_digest")
    if original_opencood_digest and original_opencood_digest != current_opencood.get("source_digest"):
        warnings.append("OpenCOOD source digest differs from the original run (or its source is missing).")

    for name in ("opencda", "opencood"):
        original_git = (original_code.get(name) or {}).get("git") or {}
        current_git = (current_code.get(name) or {}).get("git") or {}
        if original_git.get("commit") and original_git.get("commit") != current_git.get("commit"):
            warnings.append(f"{name} commit differs: original {original_git.get('commit')}, current {current_git.get('commit')}.")
        if current_git.get("dirty"):
            warnings.append(f"{name} replay worktree has uncommitted changes; results may differ.")
        if name == "opencood" and original_git.get("dirty"):
            # OpenCOOD diffs are not saved; exact changes cannot be compared.
            warnings.append("Original opencood worktree had uncommitted changes that cannot be verified.")
    original_packages = original_code.get("packages") or {}
    current_packages = current_code.get("packages") or {}
    for package in sorted(original_packages):
        if original_packages[package] != current_packages.get(package):
            warnings.append(f"{package} version differs: original {original_packages[package]}, current {current_packages.get(package)}.")
    if original.get("code_diff"):
        warnings.append(f"Original run had uncommitted changes; apply {original['code_diff']} from the run folder to match it.")
    elif ((original_code.get("opencda") or {}).get("git") or {}).get("dirty"):
        warnings.append("Original OpenCDA worktree was dirty but its code diff is not available; exact replay cannot be verified.")

    original_hashes = _model_input_hashes(original.get("checkpoint"))
    current_hashes = _model_input_hashes(collect_checkpoint_info(opt))
    changed = sorted(path for path in original_hashes.keys() | current_hashes.keys() if original_hashes.get(path) != current_hashes.get(path))
    if changed:
        warnings.append(f"Model files differ from the original run (sha256 mismatch or missing): {', '.join(changed)}.")

    # GPU model and CUDA/cuDNN versions can also influence inference results.
    original_device = _device_signature(original.get("device"))
    current_device = _device_signature(collect_device_info())
    if original_device["device"]:
        for key, label in (("device", "Device"), ("gpus", "GPU"), ("cuda", "CUDA version"), ("cudnn", "cuDNN version")):
            if original_device[key] != current_device[key]:
                warnings.append(f"{label} differs: original {original_device[key]}, current {current_device[key]}.")
    return warnings
