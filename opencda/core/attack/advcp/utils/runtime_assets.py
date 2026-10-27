"""
Runtime asset generation helper for AdvCP mesh-derived assets.

Provides :class:`AdvCPRuntimeAssetHelper`, which the AdvCP model manager
uses to generate spoof and removal assets on demand while loading the
AdvCP config, so that ``.ply``, ``.pkl`` and ``.npy`` files do not have
to be committed or downloaded beforehand.

Runtime generation is opt-in through the AdvCP YAML::

    asset_runtime_generation: true
    vehicle_blueprint: vehicle.tesla.model3  # or car_mesh_dimensions: [4.3, 1.91, 1.26]
    # car_mesh_source_path: /path/to/vehicle.obj
    # car_mesh_preserve_aspect: true
    # asset_cache_dir: ~/.cache/cavise/advcp-assets

Generation and caching are independent:

- **Explicit paths** (``car_mesh_path`` / ``car_mesh_divide_path``,
  ``remove_adv_shape_*_path``): missing files are generated there.
- **Cache** (``asset_cache_dir``): assets are stored under a directory
  keyed by a hash of every generation input and reused across runs.
- **Neither**: spoof assets are generated into a per-process temporary
  directory. Removal assets need no generation in that case, because the
  runtime's default adv-shape partition is identical to the generated one.

Every generated asset set carries a ``metadata.json`` (or a
``<file>.advcp-meta.json`` sidecar for explicit paths) recording the
generation inputs and :data:`ASSET_FORMAT_VERSION`. Assets whose
metadata does not match the current inputs are regenerated. Files at
explicit paths without metadata are user-provided: they are never
overwritten, and only the missing files are derived from them (e.g. a
spoof divide for a user-supplied mesh).
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, MutableMapping

import numpy as np

from opencda.core.attack.advcp.utils.asset_utils import (
    ADVSHAPE_TEMPLATE_DIMENSIONS_M,
    ADVSHAPE_TEMPLATE_SUBDIVISION_LEVELS,
    DEFAULT_CAR_MESH_SUBDIVISION_LEVELS,
    advshape_template_mesh,
    blueprint_dimensions_m,
    copy_or_generate_mesh,
    dump_divide_pickle,
    generate_divide_indices,
    parse_dimensions_arg,
    read_mesh,
    save_perturbation,
    validate_divide_pieces,
    write_mesh,
)

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.advcp_manager")

# Bump whenever the generation algorithm changes so that cached assets
# produced by an older algorithm are regenerated.
ASSET_FORMAT_VERSION = 2

_CACHE_METADATA_NAME = "metadata.json"
_SIDECAR_SUFFIX = ".advcp-meta.json"


class AdvCPRuntimeAssetHelper:
    """Runtime generation helper for AdvCP mesh-derived assets.

    Generates (or reuses) the car mesh, spoof mesh-divide, and removal
    adv-shape assets described by the AdvCP config and points the config
    at them.

    Typical usage (performed by ``AdvCoperceptionModelManager.load_config``)::

        AdvCPRuntimeAssetHelper.prepare_assets(config, config_dir, spoof_paths_explicit=False)
    """

    _temporary_dir: Path | None = None

    @staticmethod
    def _resolve_path(path_value: Any, base_dir: Path | None) -> Path:
        """Resolve a config path value to an absolute :class:`Path`.

        Parameters
        ----------
        path_value : Any
            Path value from the AdvCP config (typically a ``str`` or
            ``Path``).
        base_dir : Path or None
            Directory relative paths are resolved against (the AdvCP YAML
            directory); the current working directory when ``None``.

        Returns
        -------
        Path
            Absolute, resolved path.
        """
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = ((base_dir or Path.cwd()) / path).resolve()
        return path

    @classmethod
    def _temporary_root(cls) -> Path:
        """Return the per-process temporary directory for uncached assets.

        Returns
        -------
        Path
            Directory created on first use and reused for the rest of the
            process.
        """
        if cls._temporary_dir is None or not cls._temporary_dir.exists():
            cls._temporary_dir = Path(tempfile.mkdtemp(prefix="advcp-assets-"))
        return cls._temporary_dir

    @staticmethod
    def _spec_key(spec: dict[str, Any]) -> str:
        """Return a short, deterministic hash of generation inputs.

        Parameters
        ----------
        spec : dict
            JSON-serialisable generation inputs.

        Returns
        -------
        str
            First 16 hex digits of the SHA-256 of the canonical JSON.
        """
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """Return the SHA-256 of a file's content.

        Parameters
        ----------
        path : Path
            File to hash.

        Returns
        -------
        str
            Hex digest.
        """
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any] | None:
        """Read a metadata JSON file.

        Parameters
        ----------
        path : Path
            Metadata file.

        Returns
        -------
        dict or None
            Parsed metadata, or ``None`` when missing or unreadable.
        """
        try:
            with path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError):
            return None
        return metadata if isinstance(metadata, dict) else None

    @staticmethod
    def _write_metadata(path: Path, spec: dict[str, Any]) -> None:
        """Write generation inputs as a metadata JSON file.

        Parameters
        ----------
        path : Path
            Destination file; parent directories are created.
        spec : dict
            JSON-serialisable generation inputs.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=2, sort_keys=True)

    @classmethod
    def _is_current(cls, outputs: list[Path], metadata_path: Path, spec: dict[str, Any]) -> bool:
        """Check whether a generated asset set exists and matches *spec*.

        Parameters
        ----------
        outputs : list of Path
            Asset files that make up the set.
        metadata_path : Path
            Metadata file recording the inputs the set was generated from.
        spec : dict
            Current generation inputs.

        Returns
        -------
        bool
            ``True`` when every output exists and the recorded inputs
            equal *spec*.
        """
        return all(path.exists() for path in outputs) and cls._read_metadata(metadata_path) == spec

    @classmethod
    def _spoof_spec(cls, config: MutableMapping[str, Any], config_dir: Path | None) -> tuple[dict[str, Any], Path | None]:
        """Collect every input that determines the generated spoof assets.

        Parameters
        ----------
        config : MutableMapping
            AdvCP configuration mapping.
        config_dir : Path or None
            AdvCP YAML directory used to resolve relative paths.

        Returns
        -------
        tuple
            ``(spec, source_path)`` where *spec* is the JSON-serialisable
            generation input set and *source_path* the optional external
            mesh.

        Raises
        ------
        ValueError
            If neither usable dimensions nor a known blueprint are given.
        FileNotFoundError
            If ``car_mesh_source_path`` does not exist.
        """
        blueprint = config.get("vehicle_blueprint")
        dimensions = parse_dimensions_arg(config.get("car_mesh_dimensions"))
        if dimensions is None:
            dimensions = blueprint_dimensions_m(blueprint)
        source_value = config.get("car_mesh_source_path")
        source_path = cls._resolve_path(source_value, config_dir) if source_value else None
        if source_path is not None and not source_path.is_file():
            raise FileNotFoundError(f"AdvCP car_mesh_source_path '{source_path}' does not exist.")
        spec = {
            "format_version": ASSET_FORMAT_VERSION,
            "kind": "spoof",
            "vehicle_blueprint": blueprint,
            "dimensions_m": [float(value) for value in dimensions],
            "source_path": str(source_path) if source_path is not None else None,
            "source_sha256": cls._file_sha256(source_path) if source_path is not None else None,
            "preserve_aspect": bool(config.get("car_mesh_preserve_aspect", True)),
            "subdivision_levels": DEFAULT_CAR_MESH_SUBDIVISION_LEVELS if source_path is None else 0,
        }
        return spec, source_path

    @classmethod
    def ensure_spoof_assets(
        cls,
        config: MutableMapping[str, Any],
        config_dir: Path | None = None,
        spoof_paths_explicit: bool = True,
    ) -> tuple[Path, Path]:
        """Ensure that spoof assets (car mesh and mesh-divide) exist.

        The output location is chosen as follows: ``asset_cache_dir``
        (hash-keyed subdirectory) when configured; otherwise the explicit
        ``car_mesh_path`` / ``car_mesh_divide_path``; otherwise a
        per-process temporary directory. The config is updated to point
        at the resulting files.

        Parameters
        ----------
        config : MutableMapping
            AdvCP configuration mapping; updated in place.
        config_dir : Path or None
            AdvCP YAML directory used to resolve relative paths.
        spoof_paths_explicit : bool
            Whether ``car_mesh_path`` / ``car_mesh_divide_path`` were set
            by the user (as opposed to bundle defaults).

        Returns
        -------
        tuple of Path
            ``(car_mesh_path, car_mesh_divide_path)``.

        Raises
        ------
        ValueError
            If the dimensions cannot be resolved or a divide group would
            produce an empty mesh piece.
        """
        spec, source_path = cls._spoof_spec(config, config_dir)
        cache_value = config.get("asset_cache_dir")
        if cache_value is not None:
            asset_dir = cls._resolve_path(cache_value, config_dir) / "spoof" / cls._spec_key(spec)
            mesh_path, divide_path = asset_dir / "car_mesh.ply", asset_dir / "car_mesh_divide.pkl"
            metadata_path = asset_dir / _CACHE_METADATA_NAME
        elif spoof_paths_explicit and config.get("car_mesh_path") and config.get("car_mesh_divide_path"):
            mesh_path = cls._resolve_path(config["car_mesh_path"], config_dir)
            divide_path = cls._resolve_path(config["car_mesh_divide_path"], config_dir)
            metadata_path = mesh_path.with_name(mesh_path.name + _SIDECAR_SUFFIX)
        else:
            asset_dir = cls._temporary_root() / "spoof" / cls._spec_key(spec)
            mesh_path, divide_path = asset_dir / "car_mesh.ply", asset_dir / "car_mesh_divide.pkl"
            metadata_path = asset_dir / _CACHE_METADATA_NAME

        user_owned_mesh = mesh_path.exists() and cls._read_metadata(metadata_path) is None
        if cls._is_current([mesh_path, divide_path], metadata_path, spec) or (user_owned_mesh and divide_path.exists()):
            logger.info("Reusing AdvCP spoof assets at mesh='%s', divide='%s'.", mesh_path, divide_path)
        elif user_owned_mesh:
            # Only the divide is missing: derive it from the user-provided mesh.
            mesh = read_mesh(mesh_path)
            divide = generate_divide_indices(mesh.vertices, mode="spoof")
            validate_divide_pieces(divide, mesh.faces, f"Spoof mesh divide for '{mesh_path}'")
            dump_divide_pickle(divide, divide_path)
            logger.info("Generated AdvCP spoof divide '%s' for user-provided mesh '%s'.", divide_path, mesh_path)
        else:
            dimensions = (spec["dimensions_m"][0], spec["dimensions_m"][1], spec["dimensions_m"][2])
            mesh = copy_or_generate_mesh(source_path, dimensions=dimensions, preserve_aspect=spec["preserve_aspect"])
            divide = generate_divide_indices(mesh.vertices, mode="spoof")
            validate_divide_pieces(divide, mesh.faces, "Generated spoof mesh divide")
            write_mesh(mesh_path, mesh)
            dump_divide_pickle(divide, divide_path)
            cls._write_metadata(metadata_path, spec)
            logger.info("Generated AdvCP runtime spoof assets at mesh='%s', divide='%s'.", mesh_path, divide_path)

        config["car_mesh_path"] = str(mesh_path)
        config["car_mesh_divide_path"] = str(divide_path)
        return mesh_path, divide_path

    @classmethod
    def ensure_remove_advshape_assets(
        cls,
        config: MutableMapping[str, Any],
        config_dir: Path | None = None,
    ) -> tuple[Path, Path | None] | None:
        """Ensure that removal adv-shape assets exist.

        Generates the removal divide (and, when
        ``remove_adv_shape_generate_zero_perturb`` is set, a zero
        perturbation) for the shared adv-shape template. Output goes to the
        explicit ``remove_adv_shape_*_path`` keys when set, otherwise to
        ``asset_cache_dir``. Without either nothing is generated: the
        runtime falls back to the identical default partition. The config
        is updated to point at generated files.

        Parameters
        ----------
        config : MutableMapping
            AdvCP configuration mapping; updated in place.
        config_dir : Path or None
            AdvCP YAML directory used to resolve relative paths.

        Returns
        -------
        tuple or None
            ``(divide_path, perturb_path)`` (``perturb_path`` is ``None``
            when no perturbation is used), or ``None`` when nothing needed
            to be generated.
        """
        generate_perturb = bool(config.get("remove_adv_shape_generate_zero_perturb", False))
        spec = {
            "format_version": ASSET_FORMAT_VERSION,
            "kind": "remove",
            "template_dimensions_m": list(ADVSHAPE_TEMPLATE_DIMENSIONS_M),
            "template_subdivision_levels": ADVSHAPE_TEMPLATE_SUBDIVISION_LEVELS,
            "zero_perturb": generate_perturb,
        }
        divide_value = config.get("remove_adv_shape_divide_path")
        perturb_value = config.get("remove_adv_shape_perturb_path")
        cache_value = config.get("asset_cache_dir")

        if divide_value is not None:
            divide_path = cls._resolve_path(divide_value, config_dir)
            perturb_path = cls._resolve_path(perturb_value, config_dir) if perturb_value is not None else None
            if generate_perturb and perturb_path is None:
                perturb_path = divide_path.with_name("mesh_perturb.npy")
            metadata_path = divide_path.with_name(divide_path.name + _SIDECAR_SUFFIX)
        elif cache_value is not None:
            asset_dir = cls._resolve_path(cache_value, config_dir) / "remove" / cls._spec_key(spec)
            divide_path = asset_dir / "mesh_divide.pkl"
            perturb_path = asset_dir / "mesh_perturb.npy" if generate_perturb else None
            metadata_path = asset_dir / _CACHE_METADATA_NAME
        else:
            logger.debug("No removal asset path or cache configured; the runtime default adv-shape partition is used.")
            return None

        outputs = [divide_path] + ([perturb_path] if generate_perturb and perturb_path is not None else [])
        if cls._is_current(outputs, metadata_path, spec):
            logger.info("Reusing AdvCP removal assets at divide='%s', perturb='%s'.", divide_path, perturb_path)
        else:
            generated_set = cls._read_metadata(metadata_path) is not None
            # Files without metadata are user-provided: never overwrite them, only fill in what is missing.
            user_owned = [path for path in outputs if path.exists() and not generated_set]
            template = advshape_template_mesh()
            if divide_path not in user_owned:
                divide = generate_divide_indices(template.vertices, mode="remove")
                validate_divide_pieces(divide, template.faces, "Generated removal mesh divide")
                dump_divide_pickle(divide, divide_path)
            if generate_perturb and perturb_path is not None and perturb_path not in user_owned:
                save_perturbation(perturb_path, np.zeros(template.vertices.shape, dtype=np.float32))
            if not user_owned:
                cls._write_metadata(metadata_path, spec)
            logger.info("Prepared AdvCP runtime removal assets at divide='%s', perturb='%s'.", divide_path, perturb_path)

        config["remove_adv_shape_divide_path"] = str(divide_path)
        if perturb_path is not None and perturb_path.exists():
            config["remove_adv_shape_perturb_path"] = str(perturb_path)
        return divide_path, perturb_path

    @classmethod
    def prepare_assets(
        cls,
        config: MutableMapping[str, Any],
        config_dir: Path | None,
        spoof_paths_explicit: bool,
    ) -> None:
        """Generate or reuse all runtime assets requested by the config.

        Does nothing unless ``asset_runtime_generation`` is true. Removal
        assets are only prepared when ``advshape`` is enabled.

        Parameters
        ----------
        config : MutableMapping
            AdvCP configuration mapping; asset path keys are updated in
            place.
        config_dir : Path or None
            AdvCP YAML directory used to resolve relative paths.
        spoof_paths_explicit : bool
            Whether the spoof asset paths were set by the user.
        """
        if not bool(config.get("asset_runtime_generation", False)):
            return
        cls.ensure_spoof_assets(config, config_dir, spoof_paths_explicit)
        if config.get("advshape") is True:
            cls.ensure_remove_advshape_assets(config, config_dir)
