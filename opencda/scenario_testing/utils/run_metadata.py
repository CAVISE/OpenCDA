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
from typing import Any, Iterable, Mapping, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml
from omegaconf import OmegaConf

from opencda.core.plan.model_resolver import sha256_file
from opencda.version import __version__

logger = logging.getLogger("cavise.opencda.opencda.scenario_testing.utils.run_metadata")

SCHEMA_VERSION = 1
REPRODUCIBILITY_DIRNAME = "reproducibility"
MANIFEST_FILENAME = "run_manifest.json"
CONFIG_FILENAME = "config.yaml"
INPUTS_DIRNAME = "inputs"
PREDICTIONS_DIRNAME = "predictions"
CODE_DIFF_FILENAME = "code_diff.patch"

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG_FILENAME = "config.yaml"
# Paths resolved relative to the AdvCP configuration directory.
ADVCP_PATH_KEYS = (
    "model_path",
    "mesh_divide_path",
    "car_mesh_path",
    "car_mesh_divide_path",
    "remove_adv_shape_perturb_path",
    "remove_adv_shape_divide_path",
)
CHECKPOINT_SUFFIXES = (".pth", ".pt", ".ckpt", ".bin", ".safetensors")
TRACKED_PACKAGES = ("numpy", "torch", "torchvision", "carla", "omegaconf", "open3d", "opencood", "spconv", "scipy", "PyYAML")


def _to_jsonable(value: Any, key: str = "") -> Any:
    """Convert metadata to JSON-compatible values and redact credentials.

    Parameters
    ----------
    value : Any
        Scalar, enum, path, mapping or collection to serialize.
    key : str, optional
        Containing field name; an empty string denotes the root value.

    Returns
    -------
    Any
        Sanitized JSON-compatible value; unsupported objects become strings.
    """
    if key in ("argv", "raw_argv") and isinstance(value, (list, tuple)):
        return _redact_argv(value)
    if SECRET_KEY_PATTERN.search(key):
        return REDACTED if value is not None else None
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(item, str(k)) for k, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _redact_string(repr(value))


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
    """Remove HTTP credentials and sensitive query values.

    Parameters
    ----------
    url : str
        URL to inspect; other schemes are returned unchanged.

    Returns
    -------
    str
        URL without userinfo and with sensitive query values redacted.
    """
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
    """Redact credential-bearing HTTP URLs embedded in text.

    Parameters
    ----------
    value : str
        Text that may contain URLs.

    Returns
    -------
    str
        Text with URL credentials removed.
    """
    return URL_PATTERN.sub(lambda match: _redact_url(match.group(0)), value)


def _redact_argv(argv: Iterable[str]) -> list[str]:
    """Redact secrets in option values and URLs.

    Parameters
    ----------
    argv : iterable of str
        Command arguments in ``--flag=value`` or ``--flag value`` form.

    Returns
    -------
    list of str
        Arguments safe to store in shared run artifacts.
    """
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


def _run_git(repo: Path, *arguments: str, preserve_whitespace: bool = False) -> str | None:
    """Run a read-only Git command and capture its output.

    Parameters
    ----------
    repo : pathlib.Path
        Git working directory.
    *arguments : str
        Git subcommand and arguments.
    preserve_whitespace : bool, optional
        Retain all whitespace, including CRLF bytes, for patch capture.

    Returns
    -------
    str or None
        UTF-8 output decoded with surrogateescape, or None if Git fails.
        Re-encoding preserved output with surrogateescape recovers its bytes.
    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run([git, "-C", str(repo), *arguments], check=True, capture_output=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return None
    output = result.stdout.decode("utf-8", errors="surrogateescape")
    return output if preserve_whitespace else output.strip()


def collect_git_info(repo: Path) -> dict[str, Any] | None:
    """Collect a checkout revision and worktree status.

    Parameters
    ----------
    repo : pathlib.Path
        Directory inside the Git checkout.

    Returns
    -------
    dict or None
        Commit, branch, redacted remote and dirty status, or None without Git.
    """
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


def _module_location(module_name: str) -> Path | None:
    """Locate an installed module without importing its dependencies.

    Parameters
    ----------
    module_name : str
        Top-level module name to locate.

    Returns
    -------
    pathlib.Path or None
        Module directory, or None when the module cannot be located.
    """
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


def collect_source_digest(directory: Path | None = None) -> str:
    """Fingerprint runtime sources, configurations and checkpoint bytes.

    Parameters
    ----------
    directory : pathlib.Path, optional
        Package directory to hash. None selects OpenCDA's entrypoint,
        ``opencda`` and ``AIM`` directories, including AIM weights.

    Returns
    -------
    str
        SHA-256 digest of sorted relative filenames and file contents.
    """
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
    } | set(CHECKPOINT_SUFFIXES)
    root = directory if directory is not None else REPO_ROOT
    paths = [] if directory is not None else [root / "opencda.py"]
    directories = (directory,) if directory is not None else (root / "opencda", root / "AIM")
    for source_dir in directories:
        if source_dir.is_dir():
            paths.extend(path for path in source_dir.rglob("*") if path.is_file() and path.suffix in extensions and "__pycache__" not in path.parts)

    for source in sorted(paths):
        if not source.is_file():
            continue
        relative_path = source.relative_to(root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)

    return digest.hexdigest()


def collect_code_versions() -> dict[str, Any]:
    """Collect code and dependency identities without heavy imports.

    Returns
    -------
    dict
        OpenCDA/OpenCOOD revisions and source digests, Python version and
        installed package versions; unavailable package versions are None.
    """
    versions: dict[str, Any] = {
        "opencda": {
            "version": __version__,
            "git": collect_git_info(REPO_ROOT),
            "source_digest": collect_source_digest(),
        },
        "python": sys.version.split()[0],
        "packages": {},
    }
    for name in TRACKED_PACKAGES:
        try:
            versions["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions["packages"][name] = None

    opencood_location = _module_location("opencood")
    if opencood_location is not None:
        toplevel = _run_git(opencood_location, "rev-parse", "--show-toplevel")
        opencood_repo = Path(toplevel) if toplevel else None
        if opencood_repo is None or opencood_repo.resolve() != REPO_ROOT:
            versions["opencood"] = {
                "git": collect_git_info(opencood_repo) if opencood_repo else None,
                "source_digest": collect_source_digest(opencood_location),
            }
    return versions


def collect_device_info() -> dict[str, Any]:
    """Collect host and optional Torch device information.

    Returns
    -------
    dict
        Platform, architecture and CPU/GPU metadata. Torch is imported lazily;
        unavailable Torch or device-query failures are recorded in the result.
    """
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    try:
        # Torch is optional and expensive; import it only for device inspection.
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
    """Convert a Git remote into an HTTP browser URL.

    Parameters
    ----------
    remote : str or None
        HTTP, SSH or local Git remote.

    Returns
    -------
    str or None
        Browser URL without the .git suffix, or None for unsupported remotes.
    """
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


def _bundle_commit(repo_root: Path, bundle_path: str, artifacts: list[dict[str, Any]]) -> str | None:
    """Find a commit whose tree contains the runtime bundle bytes.

    Parameters
    ----------
    repo_root : pathlib.Path
        Models checkout root.
    bundle_path : str
        Bundle directory relative to the checkout root.
    artifacts : list of dict
        Runtime artifact paths to verify, including ignored downloads.

    Returns
    -------
    str or None
        Matching HEAD or FETCH_HEAD commit, or None if any artifact is
        missing, differs from its Git blob, or the bundle has local changes.
    """
    # Git status excludes ignored downloads; verify every runtime file's bytes.
    blobs = {
        f"{bundle_path}/{item['path']}": _run_git(repo_root, "hash-object", "--no-filters", "--", f"{bundle_path}/{item['path']}")
        for item in artifacts
    }
    if not blobs or not all(blobs.values()):
        return None
    for ref in ("HEAD", "FETCH_HEAD"):
        commit = _run_git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        if not commit:
            continue
        tracked = _run_git(repo_root, "ls-tree", "-r", "--name-only", commit, "--", bundle_path)
        if not tracked:
            continue
        unchanged = _run_git(repo_root, "diff", "--quiet", commit, "--", bundle_path) is not None
        untracked = _run_git(repo_root, "ls-files", "--others", "--exclude-standard", "--", bundle_path)
        if (
            unchanged
            and not untracked
            and all(_run_git(repo_root, "rev-parse", "--verify", f"{commit}:{path}") == blob for path, blob in blobs.items())
        ):
            return commit
    return None


def _bundle_artifacts(bundle_dir: Path) -> list[dict[str, Any]]:
    """Hash runtime model inputs and validate declared checksums.

    Parameters
    ----------
    bundle_dir : pathlib.Path
        Bundle containing checkpoints, config.yaml and optional meta.yaml.

    Returns
    -------
    list of dict
        Relative paths, sizes and actual hashes, with expected hashes and
        match status where declared. Missing files have None sizes and hashes.
    """
    expected: dict[str, str | None] = {}
    meta_path = bundle_dir / "meta.yaml"
    if meta_path.is_file():
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            meta = None
        if isinstance(meta, dict) and isinstance(meta.get("artifacts"), list):
            expected = {
                str(item["path"]): str(item["sha256"]) if item.get("sha256") else None
                for item in meta["artifacts"]
                if isinstance(item, dict) and item.get("path")
            }
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


def describe_bundle(bundle_dir_value: str | None, *, bundle_id: str | None, opt: Any) -> dict[str, Any]:
    """Describe a bundle using hashes and a verified repository revision.

    Parameters
    ----------
    bundle_dir_value : str or None
        Local bundle directory; None records an unresolved bundle.
    bundle_id : str or None
        Logical model or asset bundle identifier.
    opt : argparse.Namespace
        Launch options containing the models reference and repository URL.

    Returns
    -------
    dict
        Bundle identity and artifact hashes. A commit URL is included only
        when all runtime artifacts match that commit's tree.
    """
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
    commit = _bundle_commit(repo_root, bundle_path, info["artifacts"])

    info.update(
        {
            "repository": _redact_string(web_url or remote or "") if (web_url or remote) else None,
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


def _advcp_override_artifacts(config_path: str | None) -> list[dict[str, Any]]:
    """Hash assets selected explicitly by an AdvCP configuration.

    Parameters
    ----------
    config_path : str or None
        AdvCP YAML path; None means there are no explicit overrides.

    Returns
    -------
    list of dict
        Asset keys, resolved paths and hashes; missing files have None hashes.
    """
    if not config_path:
        return []
    source = Path(config_path).expanduser().resolve()
    if not source.is_file():
        return []
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        logger.warning("Unable to read AdvCP config %s for artifact hashing.", source)
        return []
    if not isinstance(config, Mapping):
        return []
    artifacts: list[dict[str, Any]] = []
    for key in ADVCP_PATH_KEYS:
        value = config.get(key)
        if not isinstance(value, (str, Path)) or not value:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        candidate = candidate.resolve()
        artifacts.append({"path": key, "source": str(candidate), "sha256": sha256_file(candidate) if candidate.is_file() else None})
    return artifacts


def collect_checkpoint_info(opt: Any) -> dict[str, Any] | None:
    """Collect cooperative-perception and optional AdvCP model metadata.

    Parameters
    ----------
    opt : argparse.Namespace
        Runtime model selectors and resolved model/asset directories.

    Returns
    -------
    dict or None
        Bundle metadata and explicit AdvCP asset hashes, or None when
        cooperative perception is disabled.
    """
    if not getattr(opt, "with_coperception", False):
        return None

    checkpoint = describe_bundle(getattr(opt, "model_dir", None), bundle_id=getattr(opt, "model_id", None), opt=opt)
    if getattr(opt, "with_advcp", False):
        checkpoint["advcp_assets"] = describe_bundle(
            getattr(opt, "advcp_assets_dir", None),
            bundle_id=getattr(opt, "advcp_assets_id", None),
            opt=opt,
        )
        checkpoint["advcp_overrides"] = {"artifacts": _advcp_override_artifacts(getattr(opt, "advcp_config", None))}
    return checkpoint


def reproducibility_dir(run_dir: Path) -> Path:
    """Locate a run's reproducibility directory.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run output directory or reproducibility directory itself.

    Returns
    -------
    pathlib.Path
        Directory containing the run manifest and saved inputs.
    """
    run_dir = Path(run_dir)
    if (run_dir / MANIFEST_FILENAME).is_file():
        return run_dir
    return run_dir / REPRODUCIBILITY_DIRNAME


def _redact_yaml_file(path: Path) -> bool:
    """Redact a saved YAML copy without constructing tagged objects.

    Parameters
    ----------
    path : pathlib.Path
        YAML copy to sanitize in place.

    Returns
    -------
    bool
        True if credentials were removed. Files without credentials remain
        unchanged; custom tags and aliases are inspected as YAML nodes only.
    """
    original = path.read_text(encoding="utf-8")
    # Compose nodes without constructing Python/NumPy objects from custom tags.
    root = yaml.compose(original, Loader=yaml.SafeLoader)
    pending = [root] if root is not None else []
    visited: set[int] = set()
    redacted = False
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, yaml.MappingNode):
            for index, (key, value) in enumerate(node.value):
                if isinstance(key, yaml.ScalarNode) and SECRET_KEY_PATTERN.search(key.value) and value.tag != "tag:yaml.org,2002:null":
                    node.value[index] = (key, yaml.ScalarNode("tag:yaml.org,2002:str", REDACTED))
                    redacted = True
                else:
                    pending.extend((key, value))
        elif isinstance(node, yaml.SequenceNode):
            pending.extend(node.value)
        elif isinstance(node, yaml.ScalarNode):
            cleaned = _redact_string(node.value)
            if cleaned != node.value:
                node.value = cleaned
                redacted = True
    if redacted:
        path.write_text(yaml.serialize(root, allow_unicode=True), encoding="utf-8")
        return True
    cleaned_text = _redact_string(original)
    if cleaned_text != original:
        path.write_text(cleaned_text, encoding="utf-8")
        return True
    return False


def _copy_advcp_config(source: Path, target: Path) -> None:
    """Copy an AdvCP configuration while preserving asset locations.

    Parameters
    ----------
    source : pathlib.Path
        Original YAML file; symlinks are resolved before anchoring asset paths.
    target : pathlib.Path
        Destination YAML file; relative canonical and legacy asset paths
        are rewritten to absolute paths before copying.
    """
    source = source.expanduser().resolve()
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
    """Snapshot runtime YAML inputs and redact their credentials.

    Parameters
    ----------
    output_dir : pathlib.Path
        Reproducibility directory in which to create inputs/.
    input_files : mapping of str to pathlib.Path or None
        Named source inputs; missing and None paths are skipped.
    opt : argparse.Namespace
        Runtime options identifying an optional model bundle.

    Returns
    -------
    dict
        Source and copy paths, hashes and redaction status keyed by input name.
    """
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
    """Save a byte-exact tracked patch unless it may contain credentials.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory in which to write code_diff.patch.
    git_info : mapping or None
        Recorded OpenCDA dirty status and untracked filenames.

    Returns
    -------
    str or None
        Patch filename, or None if there is no diff or it contains potential
        credentials. Untracked file contents are never copied.
    """
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
    if _redact_string(diff) != diff or SECRET_KEY_PATTERN.search(diff):
        logger.warning("Git patch contains potential credentials; refusing to persist it. Commit or sanitize the changes before sharing this run.")
        return None
    (output_dir / CODE_DIFF_FILENAME).write_bytes(diff.encode("utf-8", errors="surrogateescape"))
    return CODE_DIFF_FILENAME


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON artifact.

    Parameters
    ----------
    path : pathlib.Path
        Destination file in an existing directory.
    payload : mapping
        Metadata converted to JSON-compatible values before serialization.
    """
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
    """Save resolved configuration, inputs and metadata before execution.

    Parameters
    ----------
    output_dir : pathlib.Path
        Evaluation output directory for this run.
    opt : argparse.Namespace
        Resolved runtime options.
    argv : iterable of str
        Effective launch arguments after replay overrides.
    raw_argv : iterable of str, optional
        Original user arguments; None omits the raw command record.
    merged_config : omegaconf.DictConfig
        Resolved default and scenario configuration.
    current_time : str
        Timestamp used to identify the run output directory.
    input_files : mapping of str to pathlib.Path or None, optional
        Named source files to snapshot; None means no additional inputs.
    replayed_from : str, optional
        Original run directory, or None for a new run.

    Returns
    -------
    dict
        Run manifest initially marked running, with credentials redacted on disk.
    """
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
    resolved_config = cast(Mapping[str, Any], OmegaConf.to_container(merged_config, resolve=True))

    code_versions = collect_code_versions()
    world = resolved_config.get("world") or {}
    seed = world.get("seed") if isinstance(world, Mapping) else None
    seed = int(seed) if seed is not None else None
    if seed is None:
        logger.warning("world.seed is not set: numpy/random/torch are not seeded, the run is not reproducible.")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scenario": getattr(opt, "test_scenario", None),
        "current_time": current_time,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
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
            "cwd": os.getcwd(),
            "options": _to_jsonable(vars(opt)),
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

    write_json(output_dir / MANIFEST_FILENAME, manifest)
    logger.info("Run manifest saved to: %s", output_dir / MANIFEST_FILENAME)
    return manifest


def finalize_run_manifest(output_dir: Path, *, status: str, duration_seconds: float | None = None, error: str | None = None) -> None:
    """Record the final status of a run whose manifest exists.

    Parameters
    ----------
    output_dir : pathlib.Path
        Evaluation output or reproducibility directory.
    status : str
        Final run status, such as completed, failed or interrupted.
    duration_seconds : float, optional
        Total elapsed run time; None means unavailable.
    error : str, optional
        Failure description, or None for a run without an error.
    """
    manifest_path = reproducibility_dir(output_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not update run manifest %s", manifest_path)
        return

    manifest.update(
        {
            "status": status,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_seconds": duration_seconds,
            "error": error,
        }
    )
    write_json(manifest_path, manifest)


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    """Read a previously saved run manifest.

    Parameters
    ----------
    run_dir : pathlib.Path
        Evaluation output or reproducibility directory.

    Returns
    -------
    dict
        Decoded manifest, including manifests from older schema revisions.

    Raises
    ------
    FileNotFoundError
        If the directory does not contain a run manifest.
    """
    manifest_path = reproducibility_dir(run_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found: this folder is not a reproducible OpenCDA run.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_prediction_records(run_dir: Path) -> list[dict[str, Any]]:  # noqa: DC02
    """Load saved per-tick boxes, scores and post-processing thresholds.

    Parameters
    ----------
    run_dir : pathlib.Path
        Evaluation output directory containing predictions/.

    Returns
    -------
    list of dict
        Records in filename order, with None for missing tensors.
    """
    # Keep NumPy optional for metadata-only commands.
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
    """Flatten runtime model and AdvCP input hashes for comparison.

    Parameters
    ----------
    checkpoint : mapping or None
        Recorded model metadata, including optional AdvCP assets/overrides.

    Returns
    -------
    dict of str to str or None
        Category-prefixed artifact paths mapped to hashes or missing markers.
    """
    if not checkpoint:
        return {}
    bundles = (("coperception", checkpoint), ("advcp", checkpoint.get("advcp_assets")), ("advcp_override", checkpoint.get("advcp_overrides")))
    return {f"{category}/{item['path']}": item.get("sha256") for category, bundle in bundles for item in (bundle or {}).get("artifacts", [])}


def _device_signature(device: Mapping[str, Any] | None) -> dict[str, Any]:
    """Select device properties that may affect numerical results.

    Parameters
    ----------
    device : mapping or None
        Host and Torch device metadata from a run manifest.

    Returns
    -------
    dict
        Device, GPU/CUDA details and CPU platform/architecture where applicable.
    """
    torch_info = (device or {}).get("torch") or {}
    return {
        "platform": (device or {}).get("platform") if torch_info.get("device") == "cpu" else None,
        "machine": (device or {}).get("machine") if torch_info.get("device") == "cpu" else None,
        "device": torch_info.get("device"),
        "gpus": [gpu.get("name") for gpu in torch_info.get("gpus") or []],
        "cuda": torch_info.get("cuda_version"),
        "cudnn": torch_info.get("cudnn_version"),
    }


def compare_with_original(original: Mapping[str, Any], opt: Any) -> list[str]:
    """Compare current code, model inputs and hardware with a saved run.

    Parameters
    ----------
    original : mapping
        Manifest from the original run.
    opt : argparse.Namespace
        Resolved options describing the current runtime model inputs.

    Returns
    -------
    list of str
        Warnings for differing or unverifiable provenance, versions and devices.
    """
    warnings: list[str] = []

    original_code = original.get("code") or {}
    current_code = collect_code_versions()

    original_python = original_code.get("python")
    current_python = current_code.get("python")
    if original_python and original_python != current_python:
        warnings.append(f"Python version differs: original {original_python}, current {current_python}.")

    for name, label in (("opencda", "OpenCDA"), ("opencood", "OpenCOOD")):
        original_repo = original_code.get(name) or {}
        current_repo = current_code.get(name) or {}
        if original_repo.get("source_digest") and original_repo["source_digest"] != current_repo.get("source_digest"):
            warnings.append(f"{label} source digest differs from the original run (or its source is missing).")
        original_git = original_repo.get("git") or {}
        current_git = current_repo.get("git") or {}
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
        for key, label in (
            ("device", "Device"),
            ("gpus", "GPU"),
            ("cuda", "CUDA version"),
            ("cudnn", "cuDNN version"),
            ("platform", "CPU platform"),
            ("machine", "CPU architecture"),
        ):
            if original_device[key] != current_device[key]:
                warnings.append(f"{label} differs: original {original_device[key]}, current {current_device[key]}.")
    return warnings
