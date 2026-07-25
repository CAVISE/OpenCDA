"""
Runtime asset generation helper for AdvCP mesh-derived assets.

Provides :class:`AdvCPRuntimeAssetHelper`, which is used by the AdvCP
attack pipeline to generate spoof and removal assets on demand during
simulation. This avoids requiring pre-generated ``.ply``, ``.pkl``, and
``.npy`` files to be committed to the repository.

The helper supports two asset categories:

1. **Spoof assets** — car mesh (``.ply``) and spoof mesh-divide (``.pkl``).
2. **Removal advshape assets** — removal mesh-divide (``.pkl``) and
   optional perturbation tensor (``.npy``).

Assets are generated once and cached on disk. Subsequent runs reuse the
cached files. The cache location is configurable via the
``asset_cache_dir`` config key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from opencda.core.attack.advcp.types import AdvCPConfig
from opencda.core.attack.advcp.utils.asset_utils import (
    advshape_template_mesh,
    blueprint_dimensions_m,
    copy_or_generate_mesh,
    dump_divide_pickle,
    generate_divide_indices,
    parse_dimensions_arg,
    save_perturbation,
    write_mesh,
)

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.advcp_manager")


class AdvCPRuntimeAssetHelper:
    """Runtime generation helper for AdvCP mesh-derived assets.

    Generates and caches the 3D assets required by the AdvCP attack
    pipeline (car mesh, mesh-divide metadata, and adversarial-shape
    perturbation tensors) on demand during simulation.

    Assets are written to the paths specified in the AdvCP config, or
    to a configurable cache directory when ``asset_cache_dir`` is set.
    Generation is skipped when the target files already exist.

    Typical usage::

        car_mesh_path, divide_path = AdvCPRuntimeAssetHelper.ensure_spoof_assets(config)
        remove_paths = AdvCPRuntimeAssetHelper.ensure_remove_advshape_assets(config)
    """

    @staticmethod
    def _resolve_absolute(path_value: Any) -> Path:
        """Resolve a config value to an absolute :class:`Path`.

        Expands the user home directory (``~``) and resolves relative
        paths against the current working directory.

        Parameters
        ----------
        path_value : Any
            Path value from the AdvCP config (typically a ``str`` or
            ``Path``).

        Returns
        -------
        Path
            Absolute, resolved path.
        """
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    @classmethod
    def _resolve_generation_roots(cls, config: Mapping[str, Any]) -> tuple[Path, Path]:
        """Resolve the output paths for the car mesh and spoof mesh-divide.

        When ``asset_cache_dir`` is set in the config, the paths are
        derived from the cache directory and the vehicle blueprint name.
        Otherwise, the paths from ``car_mesh_path`` and
        ``car_mesh_divide_path`` in the config are used directly.

        Parameters
        ----------
        config : Mapping
            AdvCP configuration mapping.

        Returns
        -------
        tuple of Path
            ``(car_mesh_path, car_mesh_divide_path)``.
        """
        car_mesh_path = cls._resolve_absolute(config.get("car_mesh_path"))
        car_mesh_divide_path = cls._resolve_absolute(config.get("car_mesh_divide_path"))
        cache_dir_value = config.get("asset_cache_dir")
        if cache_dir_value is None:
            return car_mesh_path, car_mesh_divide_path

        cache_dir = cls._resolve_absolute(cache_dir_value)
        blueprint = str(config.get("vehicle_blueprint") or "vehicle.default").replace(".", "_")
        mesh_name = f"car_mesh_{blueprint}.ply"
        divide_name = f"car_mesh_divide_{blueprint}.pkl"
        return cache_dir / mesh_name, cache_dir / "spoof" / divide_name

    @staticmethod
    def _resolve_dimensions(config: Mapping[str, Any]) -> tuple[float, float, float]:
        """Resolve the target vehicle dimensions from the AdvCP config.

        Prefers an explicit ``car_mesh_dimensions`` override. Falls back
        to the blueprint dimensions from :func:`blueprint_dimensions_m`.

        Parameters
        ----------
        config : Mapping
            AdvCP configuration mapping.

        Returns
        -------
        tuple of float
            ``(length, width, height)`` in meters.
        """
        dimensions_override = parse_dimensions_arg(config.get("car_mesh_dimensions"))
        if dimensions_override is not None:
            return dimensions_override
        return blueprint_dimensions_m(config.get("vehicle_blueprint"))

    @classmethod
    def ensure_spoof_assets(cls, config: AdvCPConfig) -> tuple[Path, Path]:
        """Ensure that spoof assets (car mesh and mesh-divide) exist.

        If the assets already exist on disk, or if runtime generation is
        disabled (``asset_runtime_generation: false``), the configured
        paths are returned without modification.

        Otherwise, the car mesh is generated (from a source mesh or as a
        box), the spoof mesh-divide indices are computed, and both are
        written to disk.

        Parameters
        ----------
        config : AdvCPConfig
            Resolved AdvCP configuration. Relevant keys:
            ``car_mesh_path``, ``car_mesh_divide_path``,
            ``asset_cache_dir``, ``asset_runtime_generation``,
            ``vehicle_blueprint``, ``car_mesh_dimensions``,
            ``car_mesh_source_path``, ``car_mesh_preserve_aspect``.

        Returns
        -------
        tuple of Path
            ``(car_mesh_path, car_mesh_divide_path)`` — the paths to the
            generated or existing assets.
        """
        car_mesh_path, car_mesh_divide_path = cls._resolve_generation_roots(config)
        runtime_enabled = bool(config.get("asset_runtime_generation", True))
        if (car_mesh_path.exists() and car_mesh_divide_path.exists()) or not runtime_enabled:
            return car_mesh_path, car_mesh_divide_path

        dimensions = cls._resolve_dimensions(config)
        source_path_value = config.get("car_mesh_source_path")
        source_path = cls._resolve_absolute(source_path_value) if source_path_value else None
        preserve_aspect = bool(config.get("car_mesh_preserve_aspect", True))

        mesh = copy_or_generate_mesh(source_path, dimensions=dimensions, preserve_aspect=preserve_aspect)
        write_mesh(car_mesh_path, mesh)
        divide = generate_divide_indices(mesh.vertices, mode="spoof")
        dump_divide_pickle(divide, car_mesh_divide_path)
        logger.info(
            "Generated AdvCP runtime spoof assets at mesh='%s', divide='%s'.",
            car_mesh_path,
            car_mesh_divide_path,
        )
        return car_mesh_path, car_mesh_divide_path

    @classmethod
    def ensure_remove_advshape_assets(cls, config: AdvCPConfig) -> tuple[Path, Path] | None:
        """Ensure that removal advshape assets (mesh-divide and perturbation) exist.

        Generates the removal mesh-divide from the AdvCP template mesh
        and, optionally, a zero-initialised perturbation tensor when
        ``remove_adv_shape_generate_zero_perturb`` is ``True``.

        Assets are written to the ``remove/`` subdirectory of the
        configured ``asset_cache_dir``. Returns ``None`` when runtime
        generation is disabled or no cache directory is configured.

        Parameters
        ----------
        config : AdvCPConfig
            Resolved AdvCP configuration. Relevant keys:
            ``asset_runtime_generation``, ``asset_cache_dir``,
            ``remove_adv_shape_generate_zero_perturb``.

        Returns
        -------
        tuple of Path or None
            ``(perturb_path, divide_path)`` when assets were generated
            or already exist, or ``None`` when runtime generation is
            disabled or no cache directory is configured.
        """
        runtime_enabled = bool(config.get("asset_runtime_generation", True))
        if not runtime_enabled:
            return None

        cache_dir_value = config.get("asset_cache_dir")
        if cache_dir_value is None:
            return None

        cache_dir = cls._resolve_absolute(cache_dir_value)
        divide_path = cache_dir / "remove" / "mesh_divide.pkl"
        perturb_path = cache_dir / "remove" / "mesh_perturb.npy"

        template = advshape_template_mesh()
        if not divide_path.exists():
            divide = generate_divide_indices(template.vertices, mode="remove")
            dump_divide_pickle(divide, divide_path)

        if bool(config.get("remove_adv_shape_generate_zero_perturb", False)) and not perturb_path.exists():
            zero_perturb = np.zeros(template.vertices.shape, dtype=np.float32)
            save_perturbation(perturb_path, zero_perturb)

        if "remove_adv_shape_divide_path" not in config:
            config["remove_adv_shape_divide_path"] = str(divide_path)
        if "remove_adv_shape_perturb_path" not in config and perturb_path.exists():
            config["remove_adv_shape_perturb_path"] = str(perturb_path)

        return perturb_path, divide_path