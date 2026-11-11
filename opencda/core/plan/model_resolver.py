"""Resolve and validate model and runtime-asset bundles."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Iterator

import yaml


logger = logging.getLogger("cavise.opencda.models")

DEFAULT_MODELS_REPOSITORY = "https://github.com/CAVISE/models.git"
DEFAULT_MODELS_REF = "main"
VALID_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CATEGORY_KINDS = {
    "coperception": "coperception-checkpoint",
    "advcp": "advcp-assets",
}


class ModelResolutionError(RuntimeError):
    """Raised when a requested model or asset bundle cannot be resolved."""


def default_models_root() -> Path:
    """Return the default models repository location.

    Returns
    -------
    pathlib.Path
        Configured models root or the ``models`` directory next to OpenCDA.
    """
    configured_root = os.environ.get("CAVISE_MODELS_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "models"


def _environment_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : bool
        Value returned when the variable is not set.

    Returns
    -------
    bool
        ``False`` for common false-like strings; otherwise ``True``.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@contextmanager
def _repository_lock(models_root: Path) -> Iterator[None]:
    """Serialize modifications to a models checkout.

    Parameters
    ----------
    models_root : pathlib.Path
        Models checkout protected by the advisory lock.

    Yields
    ------
    None
        Control while the exclusive repository lock is held.
    """
    models_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = models_root.parent / f".{models_root.name}.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _run_git(arguments: list[str], *, timeout: int = 300) -> None:
    """Run Git and translate command failures into resolution errors.

    Parameters
    ----------
    arguments : list[str]
        Arguments passed to the Git executable.
    timeout : int, default=300
        Maximum command duration in seconds.

    Raises
    ------
    ModelResolutionError
        If Git is unavailable, times out, or exits unsuccessfully.
    """
    git_executable = shutil.which("git")
    if git_executable is None:
        raise ModelResolutionError("Git is required to fetch model bundles, but it is not installed in the runtime environment.")

    command = [git_executable, *arguments]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise ModelResolutionError(f"Git timed out after {timeout} seconds while fetching the requested model bundle.") from error
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or str(error)).strip()
        raise ModelResolutionError(f"Git failed while fetching the requested model bundle: {details}") from error


def _clone_sparse_repository(models_root: Path, repository_url: str, ref: str, bundle_path: str) -> None:
    """Create a sparse models checkout containing one bundle.

    Parameters
    ----------
    models_root : pathlib.Path
        Destination for the checkout.
    repository_url : str
        Models Git repository URL.
    ref : str
        Branch or tag to clone.
    bundle_path : str
        Repository-relative bundle path selected for checkout.

    Raises
    ------
    ModelResolutionError
        If the destination is non-empty or Git cannot create the checkout.
    """
    if models_root.exists() and any(models_root.iterdir()):
        raise ModelResolutionError(
            f'Model repository root "{models_root}" exists, is not a Git checkout, and is not empty; refusing to overwrite it.'
        )

    _run_git(
        [
            "clone",
            "--filter=blob:none",
            "--depth=1",
            "--sparse",
            "--branch",
            ref,
            "--",
            repository_url,
            str(models_root),
        ]
    )
    _run_git(["-C", str(models_root), "sparse-checkout", "set", bundle_path])


def _expand_sparse_checkout(models_root: Path, bundle_path: str) -> None:
    """Add a bundle path to an existing sparse checkout.

    Parameters
    ----------
    models_root : pathlib.Path
        Existing models Git checkout.
    bundle_path : str
        Repository-relative bundle path to add.

    Raises
    ------
    ModelResolutionError
        If Git cannot update the sparse-checkout definition.
    """
    _run_git(["-C", str(models_root), "sparse-checkout", "add", bundle_path])


def _fetch_bundle_from_ref(models_root: Path, bundle_path: str, ref: str) -> None:
    """Fetch a ref and restore one bundle from it.

    Parameters
    ----------
    models_root : pathlib.Path
        Existing models Git checkout.
    bundle_path : str
        Repository-relative bundle path to restore.
    ref : str
        Remote ref containing the requested bundle.

    Raises
    ------
    ModelResolutionError
        If Git cannot fetch the ref or restore the bundle.
    """
    _run_git(["-C", str(models_root), "fetch", "--depth=1", "--end-of-options", "origin", ref])
    _run_git(["-C", str(models_root), "checkout", "FETCH_HEAD", "--", bundle_path])


def _safe_artifact_path(bundle_root: Path, relative_path: object) -> Path:
    """Resolve an artifact path without allowing directory traversal.

    Parameters
    ----------
    bundle_root : pathlib.Path
        Root directory of the selected bundle.
    relative_path : object
        Untrusted artifact path read from metadata.

    Returns
    -------
    pathlib.Path
        Artifact path rooted inside the bundle.

    Raises
    ------
    ModelResolutionError
        If the metadata value is not a safe relative path.
    """
    if not isinstance(relative_path, str):
        raise ModelResolutionError(f'Invalid artifact path in "{bundle_root / "meta.yaml"}": expected a string.')
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts:
        raise ModelResolutionError(f'Unsafe artifact path "{relative_path}" in "{bundle_root / "meta.yaml"}".')
    return bundle_root.joinpath(*pure_path.parts)


def _sha256(path: Path) -> str:
    """Calculate a file's SHA-256 digest.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(bundle_root: Path, *, bundle_id: str, category: str) -> None:
    """Validate bundle identity and artifact integrity.

    Parameters
    ----------
    bundle_root : pathlib.Path
        Bundle directory containing ``meta.yaml`` and artifacts.
    bundle_id : str
        Expected logical bundle ID.
    category : str
        Expected bundle category from :data:`CATEGORY_KINDS`.

    Raises
    ------
    ModelResolutionError
        If metadata is missing or invalid, or an artifact fails validation.
    """
    metadata_path = bundle_root / "meta.yaml"
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ModelResolutionError(f'Model metadata is missing or unreadable: "{metadata_path}".') from error
    except yaml.YAMLError as error:
        raise ModelResolutionError(f'Model metadata is invalid YAML: "{metadata_path}": {error}') from error

    if not isinstance(metadata, dict):
        raise ModelResolutionError(f'Model metadata must be a mapping: "{metadata_path}".')
    if metadata.get("schema_version") != 1:
        raise ModelResolutionError(f'Unsupported metadata schema in "{metadata_path}".')
    if metadata.get("id") != bundle_id:
        raise ModelResolutionError(f'Model metadata ID does not match bundle "{bundle_id}".')
    if metadata.get("kind") != CATEGORY_KINDS[category]:
        raise ModelResolutionError(f'Model metadata kind does not match category "{category}".')

    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ModelResolutionError(f'Model metadata contains no artifacts: "{metadata_path}".')

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ModelResolutionError(f'Invalid artifact entry in "{metadata_path}".')
        artifact_path = _safe_artifact_path(bundle_root, artifact.get("path"))
        if not artifact_path.is_file():
            raise ModelResolutionError(f'Model artifact is missing: "{artifact_path}".')
        if artifact_path.stat().st_size != artifact.get("size"):
            raise ModelResolutionError(f'Model artifact size does not match metadata: "{artifact_path}".')
        if _sha256(artifact_path) != artifact.get("sha256"):
            raise ModelResolutionError(f'Model artifact checksum does not match metadata: "{artifact_path}".')


def resolve_bundle(
    category: str,
    bundle_id: str,
    *,
    models_root: Path | None = None,
    repository_url: str | None = None,
    ref: str | None = None,
    auto_fetch: bool | None = None,
) -> Path:
    """Resolve a local bundle and optionally fetch it on demand.

    Parameters
    ----------
    category : str
        Bundle category, such as ``coperception`` or ``advcp``.
    bundle_id : str
        Logical bundle ID.
    models_root : pathlib.Path, optional
        Models checkout location. Defaults to :func:`default_models_root`.
    repository_url : str, optional
        Git repository used when the bundle must be fetched.
    ref : str, optional
        Branch or tag selected for the models checkout.
    auto_fetch : bool, optional
        Whether a missing bundle may be fetched. The environment controls the
        value when omitted.

    Returns
    -------
    pathlib.Path
        Validated local bundle directory.

    Raises
    ------
    ModelResolutionError
        If input is invalid or the bundle cannot be fetched and validated.
    """
    if category not in CATEGORY_KINDS:
        raise ModelResolutionError(f'Unsupported model bundle category: "{category}".')
    if not VALID_BUNDLE_ID.fullmatch(bundle_id):
        raise ModelResolutionError(f'Invalid model bundle ID: "{bundle_id}".')

    resolved_root = (models_root or default_models_root()).expanduser().resolve()
    resolved_repository = repository_url or os.environ.get("CAVISE_MODELS_REPOSITORY", DEFAULT_MODELS_REPOSITORY)
    resolved_ref = ref or os.environ.get("CAVISE_MODELS_REF", DEFAULT_MODELS_REF)
    should_fetch = _environment_flag("CAVISE_MODELS_AUTO_FETCH", True) if auto_fetch is None else auto_fetch
    bundle_root = resolved_root / category / bundle_id

    if bundle_root.is_dir():
        validate_bundle(bundle_root, bundle_id=bundle_id, category=category)
        return bundle_root
    if not should_fetch:
        raise ModelResolutionError(f'Model bundle "{category}/{bundle_id}" is not available locally at "{bundle_root}".')

    logger.warning(
        'Model bundle "%s/%s" is not available locally at "%s"; attempting a sparse clone from %s at ref %s.',
        category,
        bundle_id,
        bundle_root,
        resolved_repository,
        resolved_ref,
    )
    with _repository_lock(resolved_root):
        if not bundle_root.is_dir():
            bundle_path = f"{category}/{bundle_id}"
            if (resolved_root / ".git").is_dir():
                _expand_sparse_checkout(resolved_root, bundle_path)
                if not bundle_root.is_dir():
                    logger.warning(
                        'Model bundle "%s" is not present in the local checkout; fetching ref %s from origin.',
                        bundle_path,
                        resolved_ref,
                    )
                    _fetch_bundle_from_ref(resolved_root, bundle_path, resolved_ref)
            else:
                _clone_sparse_repository(resolved_root, resolved_repository, resolved_ref, bundle_path)

    if not bundle_root.is_dir():
        raise ModelResolutionError(
            f'Model bundle "{category}/{bundle_id}" was not found after fetching "{resolved_repository}" at ref "{resolved_ref}".'
        )
    validate_bundle(bundle_root, bundle_id=bundle_id, category=category)
    return bundle_root


def resolve_runtime_models(options: object) -> None:
    """Resolve model and AdvCP asset paths on runtime options.

    Parameters
    ----------
    options : object
        Parsed OpenCDA options. Resolved ``model_dir`` and
        ``advcp_assets_dir`` attributes are written back to this object.

    Raises
    ------
    ModelResolutionError
        If option combinations are invalid or a required bundle is unavailable.
    """
    models_root_value = getattr(options, "models_root", None)
    models_root = Path(models_root_value) if models_root_value else default_models_root()
    repository_url = getattr(options, "models_repository", None)
    ref = getattr(options, "models_ref", None)
    auto_fetch = False if getattr(options, "no_auto_fetch_models", False) else None

    if getattr(options, "with_coperception", False):
        model_dir = getattr(options, "model_dir", None)
        model_id = getattr(options, "model_id", None)
        if model_dir and model_id:
            raise ModelResolutionError("--model-dir and --model-id cannot be used together.")
        if model_dir:
            resolved_model_dir = Path(model_dir).expanduser().resolve()
            if not resolved_model_dir.is_dir():
                raise ModelResolutionError(f'Explicit model directory does not exist: "{resolved_model_dir}".')
            setattr(options, "model_dir", str(resolved_model_dir))
        elif model_id:
            resolved_model_dir = resolve_bundle(
                "coperception",
                model_id,
                models_root=models_root,
                repository_url=repository_url,
                ref=ref,
                auto_fetch=auto_fetch,
            )
            setattr(options, "model_dir", str(resolved_model_dir))
        else:
            raise ModelResolutionError("Cooperative perception requires either --model-id or --model-dir.")
    elif getattr(options, "model_id", None) or getattr(options, "model_dir", None):
        raise ModelResolutionError("--model-id and --model-dir require --with-coperception.")

    if getattr(options, "with_advcp", False):
        asset_bundle_id = getattr(options, "advcp_assets_id", "base-car")
        assets_dir = resolve_bundle(
            "advcp",
            asset_bundle_id,
            models_root=models_root,
            repository_url=repository_url,
            ref=ref,
            auto_fetch=auto_fetch,
        )
        setattr(options, "advcp_assets_dir", str(assets_dir))
