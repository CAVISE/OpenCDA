"""
Top-level AdvCP model manager and visualizer.

``AdvCoperceptionModelManager`` is the entry point that wires AdvCP
into the cooperative-perception inference pipeline. Use it in place of
the regular ``CoperceptionModelManager`` to enable the attack family.

It holds:

- The resolved AdvCP YAML config.
- The current per-tick memory data (set by the calling scenario
  pipeline before each inference call).
- The persistent intermediate-fusion attack state (best perturbations
  to use as warm starts on subsequent ticks).

For each fusion mode the manager overrides the corresponding
``_run_*_inference`` hook so that AdvCP's attack runner is called
instead of the vanilla OpenCOOD inference helper.

``AdvCoperceptionVisualizer`` extends the regular cooperative
visualizer with attacker-aware point coloring and dedicated colors
for fake / removed boxes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import numpy as np
import numpy.typing as npt
import yaml

from opencda.core.attack.advcp.attack_helper import AdvCPAttackHelper
from opencda.core.attack.advcp.early_fusion_attack import AdvCoperceptionEarlyFusionAttack
from opencda.core.attack.advcp.intermediate_fusion_attack import AdvCoperceptionIntermediateFusionAttack
from opencda.core.attack.advcp.late_fusion_attack import AdvCoperceptionLateFusionAttack
from opencda.core.attack.advcp.types import (
    AdvCPAttackResult,
    AdvCPConfig,
    AdvCPIntermediateAttackState,
    AdvCPMemoryData,
    AdvCPVisualizationContext,
    AgentId,
    AttackerId,
)
from opencda.core.common.coperception_model_manager import (
    CoperceptionInferenceResult,
    CoperceptionModelManager,
    CoperceptionVisualizationConfig,
    CoperceptionVisualizer,
)

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.advcp_manager")


class AdvCoperceptionVisualizer(CoperceptionVisualizer):
    """
    Cooperative-perception visualizer with AdvCP-aware point coloring.

    Two visual additions on top of the base visualizer:

    - Dedicated bbox colors for ``fake`` (spoofed) and ``removed``
      target boxes pulled from
      :class:`AdvCPVisualizationContext`.
    - Per-CAV point coloring that treats configured attackers and
      the spoofing-injected points (marked by
      ``origin_lidar_spoofing_masks``) differently from regular CAVs.
    """

    _DEFAULT_VISUALIZATION_CONFIG: CoperceptionVisualizationConfig = {
        "background": (0, 0, 0),
        "lidar_point_colors": {
            "other": (255, 255, 255),
        },
        "bbox_colors": {
            "gt": (0, 255, 0),
            "pred": (255, 0, 0),
            "fake": (180, 0, 255),
            "removed": (56, 189, 248),
        },
        "bbox_line_thickness": 5,
        "image_dpi": 400,
    }

    @classmethod
    def _get_extra_box_tensors(cls, visualization_context: AdvCPVisualizationContext | None = None) -> dict[str, Any]:
        """
        Provide the visualizer's optional ``fake`` and ``removed`` boxes.

        Returns
        -------
        dict
            Mapping from ``"fake"`` / ``"removed"`` to the corresponding
            tensor on the context (possibly ``None``). An empty dict
            when no context is supplied.
        """
        if not visualization_context:
            return {}
        return {
            "fake": visualization_context.fake_box_tensor,
            "removed": visualization_context.removed_box_tensor,
        }

    @staticmethod
    def _require_visualization_value(config: Mapping[str, Any], section: str, key: str) -> Any:
        """
        Look up ``config[section][key]``, raising if either level is missing.

        Used to surface human-readable errors when the visualization
        YAML is incomplete.
        """
        section_mapping = config.get(section)
        if not isinstance(section_mapping, Mapping):
            raise ValueError(f"Unexpected None in AdvCP visualization config for '{section}'.")
        return AdvCPAttackHelper.require_config_value(
            section_mapping,
            key,
            config_name=f"AdvCP visualization config for '{section}'",
        )

    @classmethod
    def _get_lidar_points_and_colors(
        cls,
        batch_data: Any,
        fallback_pcd: Any,
        config: Mapping[str, Any],
        visualization_context: AdvCPVisualizationContext | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Compute colored lidar points for visualization.

        When the batch carries the AdvCP-augmented per-agent bundles
        (``origin_lidar_by_agent`` and ``origin_lidar_spoofing_masks``),
        each agent's points are colored by role (ego, attacker,
        other) and points marked by the spoofing mask are highlighted
        with the dedicated ``spoofing`` color.

        If the batch is not a mapping, has no ego entry, lacks either
        the per-agent point bundle or the spoofing masks, or produces
        no drawable AdvCP points, this method delegates to
        ``CoperceptionVisualizer._get_lidar_points_and_colors``. That
        base path may still visualize generic cooperative-perception
        lidar fields; otherwise it uses ``fallback_pcd``.

        Parameters
        ----------
        batch_data : Any
            OpenCOOD-collated batch for the current inference tick.
            AdvCP expects the ego entry to optionally contain
            per-agent visualization bundles:
            ``origin_lidar_by_agent`` (one point cloud per CAV),
            ``origin_lidar_roles`` (``"ego"`` / ``"other"`` labels),
            ``origin_lidar_agent_ids`` (scenario agent ids), and
            ``origin_lidar_spoofing_masks`` (boolean masks for points
            injected by spoofing). When these bundles are present, the
            method colors each agent independently instead of drawing
            one merged point cloud.
        fallback_pcd : Any
            Standard point cloud passed by the base visualizer. Used
            only when ``batch_data`` does not contain the AdvCP
            per-agent bundles, or when those bundles contain no
            drawable points.
        config : Mapping
            Resolved visualization config. In addition to the base
            keys, AdvCP reads ``lidar_point_colors["other"]`` and may
            use optional ``"ego"``, ``"attackers"``, ``"spoofing"``,
            or per-agent entries such as ``"cav-2"``.
        visualization_context : Optional[AdvCPVisualizationContext]
            Context for the current tick. Its
            ``attacker_ids`` field is used to color active attackers;
            if it is ``None`` or has no attackers, points are colored
            only by role / per-agent config.

        Returns
        -------
        tuple of npt.NDArray
            ``(points, colors)`` arrays ready for the base visualizer.
        """
        if not isinstance(batch_data, Mapping):
            return super()._get_lidar_points_and_colors(
                batch_data,
                fallback_pcd,
                config,
                visualization_context=visualization_context,
            )

        ego_entry = batch_data.get("ego")
        if not isinstance(ego_entry, Mapping):
            return super()._get_lidar_points_and_colors(
                batch_data,
                fallback_pcd,
                config,
                visualization_context=visualization_context,
            )

        if "origin_lidar_by_agent" not in ego_entry or "origin_lidar_spoofing_masks" not in ego_entry:
            return super()._get_lidar_points_and_colors(
                batch_data,
                fallback_pcd,
                config,
                visualization_context=visualization_context,
            )

        lidar_point_colors = config["lidar_point_colors"]
        other_color = cls._as_uint8_color(cls._require_visualization_value(config, "lidar_point_colors", "other"))
        ego_color = cls._as_uint8_color(lidar_point_colors.get("ego", other_color))
        spoofing_color = cls._as_uint8_color(lidar_point_colors.get("spoofing", other_color))
        points_by_agent = ego_entry["origin_lidar_by_agent"]
        roles = list(ego_entry.get("origin_lidar_roles", []))
        agent_ids = list(ego_entry.get("origin_lidar_agent_ids", []))
        spoofing_masks = list(ego_entry.get("origin_lidar_spoofing_masks", []))
        colored_points = []
        colored_values = []

        for idx, points in enumerate(points_by_agent):
            agent_points = cls._to_numpy_points(points)
            if agent_points.size == 0:
                continue

            role = roles[idx] if idx < len(roles) else "default"
            agent_id = str(agent_ids[idx]) if idx < len(agent_ids) else None
            base_color = cls._resolve_point_color(
                config,
                agent_id=agent_id,
                role=role,
                other_color=other_color,
                ego_color=ego_color,
                visualization_context=visualization_context,
            )
            agent_colors = np.tile(np.asarray(base_color, dtype=np.uint8), (agent_points.shape[0], 1))
            if idx < len(spoofing_masks):
                spoofing_mask = cls._to_numpy_array(spoofing_masks[idx]).astype(bool).reshape(-1)
                if spoofing_mask.shape[0] == agent_points.shape[0]:
                    agent_colors[spoofing_mask] = np.asarray(spoofing_color, dtype=np.uint8)

            colored_points.append(agent_points)
            colored_values.append(agent_colors)

        if colored_points:
            return np.vstack(colored_points), np.vstack(colored_values)

        return super()._get_lidar_points_and_colors(
            batch_data,
            fallback_pcd,
            config,
            visualization_context=visualization_context,
        )

    @classmethod
    def _resolve_point_color(
        cls,
        config: Mapping[str, Any],
        agent_id: str | None,
        role: str,
        other_color: tuple[int, int, int],
        ego_color: tuple[int, int, int],
        visualization_context: AdvCPVisualizationContext | None = None,
    ) -> tuple[int, int, int]:
        """
        Pick the point color for a single CAV.

        Resolution order:
        1. Per-agent override via ``lidar_point_colors[agent_id]``.
        2. The dedicated ``attackers`` color when ``agent_id`` is in
           the active attacker list from the visualization context.
        3. The ego color when ``role == "ego"``.
        4. The fallback ``other`` color.

        Returns
        -------
        tuple of int
            RGB triple in ``uint8`` range.
        """
        lidar_point_colors = config["lidar_point_colors"]
        if agent_id is not None and agent_id in lidar_point_colors:
            return cls._as_uint8_color(lidar_point_colors[agent_id])
        attacker_ids = set(visualization_context.attacker_ids if visualization_context else [])
        attacker_color = cls._as_uint8_color(lidar_point_colors.get("attackers", other_color))
        if agent_id is not None and agent_id in attacker_ids:
            return attacker_color
        if role == "ego":
            return ego_color
        return other_color


class AdvCoperceptionModelManager(CoperceptionModelManager):
    """
    Cooperative-perception model manager with AdvCP attacks enabled.

    Overrides ``CoperceptionModelManager`` to extend base model
    manager with support for AdvCP framework. Reads the AdvCP
    YAML config from ``opt.advcp_config``, and redirects the
    per-fusion inference hooks to the corresponding AdvCP
    attack runner.

    Attributes
    ----------
    advcp_config : AdvCPConfig
        Resolved AdvCP config (defaults applied, paths normalised).
    current_memory_data : Optional[AdvCPMemoryData]
        Per-tick memory data set by the calling pipeline before each
        inference call. AdvCP needs it to access raw lidar data and
        per-agent poses.
    intermediate_attack_state : AdvCPIntermediateAttackState
        Persistent state used by the intermediate-fusion attack to
        carry memory snapshots and warm-start perturbations between
        inference calls.
    """

    VISUALIZER_CLASS = AdvCoperceptionVisualizer
    SEQUENCE_BOX_GROUP_NAMES: tuple[str, ...] = ("pred", "gt", "fake", "removed")

    def __init__(
        self,
        opt: Any,
        current_time: str,
        payload_handler: Any = None,
        coperception_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize the manager.

        Parameters
        ----------
        opt : Any
            CLI options. ``opt.advcp_config`` is the path to the AdvCP
            YAML.
        current_time : str
        payload_handler : Optional[Any]
        coperception_config : Optional[Mapping]
            Cooperative perception configuration overrides.
        """
        self.advcp_config = self.load_config(getattr(opt, "advcp_config", None))
        self.current_memory_data: Optional[AdvCPMemoryData] = None
        self.intermediate_attack_state: AdvCPIntermediateAttackState = {}
        super().__init__(
            opt,
            current_time,
            payload_handler=payload_handler,
            coperception_config=coperception_config,
        )

    @staticmethod
    def load_config(config_path: str | None) -> AdvCPConfig:
        """
        Load and normalise the AdvCP YAML config.

        Applies defaults for every recognised key, resolves
        relative asset paths against the YAML directory, validates
        attacker ids, and asserts that all required keys end up
        non-``None``.

        Parameters
        ----------
        config_path : Optional[str]
            Path to the AdvCP YAML. If ``None`` or unloadable, the
            full default config is returned.

        Returns
        -------
        AdvCPConfig
            Resolved config, ready to be consumed by attack runners.
        """
        config: dict[str, object] = {}
        config_dir: Path | None = None
        local_model_root = Path(__file__).resolve().parent / "3d_models"

        if not config_path:
            logger.warning("AdvCP config path is not provided. Falling back to default AdvCP config.")
        else:
            config_dir = Path(config_path).expanduser().resolve().parent
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    loaded_config = yaml.safe_load(handle) or {}
            except OSError as exc:
                logger.warning("Unable to load AdvCP config '%s': %s. Falling back to defaults.", config_path, exc)
            else:
                if isinstance(loaded_config, dict):
                    config = loaded_config
                else:
                    logger.warning("AdvCP config '%s' is not a mapping. Falling back to defaults.", config_path)

        config.setdefault("mode", "spoofing")
        config.setdefault("default_size", (4.5, 2.0, 1.6))
        config.setdefault("boxes", [{"relative": (5.0, 0.0, 0.0, 0.0, 90.0, 0.0)}])
        config.setdefault("attacker_ids", ["cav-1"])
        config.setdefault("advshape", False)
        config.setdefault("density", "sampled")
        config.setdefault("dense_distance", 10.0)
        config.setdefault("sync", True)
        config.setdefault("init", True)
        config.setdefault("online", True)
        config.setdefault("step", 25)
        config.setdefault("max_perturb", 10.0)
        step_value = cast(int | str, config["step"])
        config.setdefault("lr", 1.0 if int(step_value) <= 2 else 0.05)
        config.setdefault("feature_size", 10)
        config.setdefault("car_mesh_path", config.get("model_path", str(local_model_root / "car_mesh_0200.ply")))
        config.setdefault(
            "car_mesh_divide_path",
            config.get("mesh_divide_path", str(local_model_root / "spoof" / "car_mesh_divide.pkl")),
        )

        config["attacker_ids"] = AdvCPAttackHelper.resolve_configured_attacker_ids(cast(AdvCPConfig, config))

        for required_key in (
            "mode",
            "default_size",
            "boxes",
            "attacker_ids",
            "advshape",
            "density",
            "dense_distance",
            "sync",
            "init",
            "online",
            "step",
            "max_perturb",
            "lr",
            "feature_size",
            "car_mesh_path",
            "car_mesh_divide_path",
        ):
            AdvCPAttackHelper.require_config_value(config, required_key)

        for path_key in ("car_mesh_path", "car_mesh_divide_path"):
            path_value = config.get(path_key)
            if path_value is not None and config_dir is not None:
                path = Path(str(path_value)).expanduser()
                if not path.is_absolute():
                    config[path_key] = str((config_dir / path).resolve())

        for optional_path_key in ("remove_adv_shape_perturb_path", "remove_adv_shape_divide_path"):
            path_value = config.get(optional_path_key)
            if path_value is not None and config_dir is not None:
                path = Path(str(path_value)).expanduser()
                if not path.is_absolute():
                    config[optional_path_key] = str((config_dir / path).resolve())
        return cast(AdvCPConfig, config)

    def validate_advcp_agents(self, valid_agent_ids: list[AgentId]) -> bool:
        """
        Cross-check configured attackers against known scenario agents.

        Drops any attacker id that does not exist in
        ``valid_agent_ids`` (with a warning) and overwrites
        ``self.advcp_config["attacker_ids"]`` with the survivors.

        Parameters
        ----------
        valid_agent_ids : list of AgentId
            Agent ids actually present in the scenario.

        Returns
        -------
        bool
            ``True`` once validation succeeded.

        Raises
        ------
        ValueError
            If no configured attacker survives the filter.
        """
        mode = AdvCPAttackHelper.require_config_value(self.advcp_config, "mode")
        configured_attacker_ids = AdvCPAttackHelper.resolve_configured_attacker_ids(self.advcp_config)
        attacker_ids: list[AttackerId] = []
        for attacker_id in configured_attacker_ids:
            if attacker_id in valid_agent_ids:
                attacker_ids.append(attacker_id)
            else:
                logger.warning(
                    "AdvCP attacker_id '%s' does not exist. Known agents: %s. AdvCP attack will not be applied for this attacker.",
                    attacker_id,
                    ", ".join(valid_agent_ids),
                )

        self.advcp_config["attacker_ids"] = attacker_ids

        if not attacker_ids:
            raise ValueError(
                "AdvCP is enabled, but no valid attackers were resolved. "
                f"Configured attackers: {', '.join(configured_attacker_ids)}. Known agents: {', '.join(valid_agent_ids)}."
            )

        logger.info("AdvCP mode: %s", mode)
        logger.info("AdvCP attacks are enabled and will be applied during cooperative perception inference.")
        logger.info("AdvCP attackers: %s", ", ".join(attacker_ids))
        return True

    def _run_late_inference(self, batch_data: Any) -> CoperceptionInferenceResult:  # noqa: DC04
        """Late-fusion inference hook: dispatches to the AdvCP late attack runner."""
        return self._build_inference_result(
            *self._inference_late_fusion_attack(
                batch_data,
                self.model,
                self.opencood_dataset,
                self.device,
                advcp_config=self.advcp_config,
                memory_data=self.current_memory_data,
            )
        )

    def _run_early_inference(self, batch_data: Any) -> CoperceptionInferenceResult:  # noqa: DC04
        """Early-fusion inference hook: dispatches to the AdvCP early attack runner."""
        return self._build_inference_result(
            *self._inference_early_fusion_attack(
                batch_data,
                self.model,
                self.opencood_dataset,
                self.device,
                advcp_config=self.advcp_config,
                memory_data=self.current_memory_data,
            )
        )

    def _run_intermediate_inference(self, batch_data: Any) -> CoperceptionInferenceResult:  # noqa: DC04
        """
        Intermediate-fusion inference hook.

        Dispatches to the AdvCP intermediate attack runner and threads
        the persistent attack state through so warm starts and
        sync-mode optimization across ticks can work.
        """
        return self._build_inference_result(
            *self._inference_intermediate_fusion_attack(
                batch_data,
                self.model,
                self.opencood_dataset,
                self.device,
                advcp_config=self.advcp_config,
                memory_data=self.current_memory_data,
                attack_state=self.intermediate_attack_state,
            )
        )

    def _requires_grad_for_inference(self) -> bool:
        """
        Whether the inference pipeline must keep gradients enabled.

        Required for intermediate fusion because the attack performs
        gradient-based optimization on the spatial features. Early and
        late fusion are gradient-free.

        Returns
        -------
        bool
        """
        core_method = self.hypes.get("fusion", {}).get("core_method")
        return core_method in {"IntermediateFusionDataset", "IntermediateFusionDatasetV2"}

    def _build_metric_update_context(
        self,
        pred_box_tensor: Any,
        pred_score: Any,
        gt_box_tensor: Any,
        visualization_context: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """
        Augment the base metric context with AdvCP-specific fields.

        Adds ``advcp_config`` and ``memory_data`` so that AdvCP-specific
        metrics can access them.

        Returns
        -------
        dict
            Metric update context.
        """
        context = super()._build_metric_update_context(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            visualization_context,
        )
        context.update(
            {
                "advcp_config": self.advcp_config,
                "memory_data": self.current_memory_data,
            }
        )
        return context

    @staticmethod
    def _inference_late_fusion_attack(*args: Any, **kwargs: Any) -> AdvCPAttackResult:
        """Adapter forwarding to ``AdvCoperceptionLateFusionAttack.run``."""
        return AdvCoperceptionLateFusionAttack.run(*args, **kwargs)

    @staticmethod
    def _inference_early_fusion_attack(*args: Any, **kwargs: Any) -> AdvCPAttackResult:
        """Adapter forwarding to ``AdvCoperceptionEarlyFusionAttack.run``."""
        return AdvCoperceptionEarlyFusionAttack.run(*args, **kwargs)

    @staticmethod
    def _inference_intermediate_fusion_attack(*args: Any, **kwargs: Any) -> AdvCPAttackResult:
        """Adapter forwarding to ``AdvCoperceptionIntermediateFusionAttack.run``."""
        return AdvCoperceptionIntermediateFusionAttack.run(*args, **kwargs)
