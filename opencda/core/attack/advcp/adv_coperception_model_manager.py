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
)
from opencda.core.common.coperception_model_manager import (
    CoperceptionInferenceResult,
    CoperceptionModelManager,
    CoperceptionVisualizationConfig,
    CoperceptionVisualizer,
)

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.advcp_manager")


class AdvCoperceptionVisualizer(CoperceptionVisualizer):
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

    @staticmethod
    def _get_context_value(visualization_context: Any, key: str, default: Any = None) -> Any:
        if visualization_context is None:
            return default
        return getattr(visualization_context, key, default)

    @classmethod
    def _get_extra_box_tensors(cls, visualization_context: Optional[Any] = None) -> dict[str, Any]:
        if not visualization_context:
            return {}
        return {
            "fake": cls._get_context_value(visualization_context, "fake_box_tensor"),
            "removed": cls._get_context_value(visualization_context, "removed_box_tensor"),
        }

    @staticmethod
    def _require_visualization_value(config: Mapping[str, Any], section: str, key: str) -> Any:
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
        visualization_context: Optional[Any] = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
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
        visualization_context: Optional[Any] = None,
    ) -> tuple[int, int, int]:
        lidar_point_colors = config["lidar_point_colors"]
        if agent_id is not None and agent_id in lidar_point_colors:
            return cls._as_uint8_color(lidar_point_colors[agent_id])
        attacker_ids = set(cls._get_context_value(visualization_context, "attacker_ids", []))
        attacker_color = cls._as_uint8_color(lidar_point_colors.get("attackers", other_color))
        if agent_id is not None and agent_id in attacker_ids:
            return attacker_color
        if role == "ego":
            return ego_color
        return other_color


class AdvCoperceptionModelManager(CoperceptionModelManager):
    VISUALIZER_CLASS = AdvCoperceptionVisualizer
    SEQUENCE_BOX_GROUP_NAMES: tuple[str, ...] = ("pred", "gt", "fake", "removed")

    def __init__(
        self,
        opt: Any,
        current_time: str,
        payload_handler: Any = None,
        visualization_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.advcp_config = self.load_config(getattr(opt, "advcp_config", None))
        self.current_memory_data: Optional[AdvCPMemoryData] = None
        self.intermediate_attack_state: AdvCPIntermediateAttackState = {}
        super().__init__(opt, current_time, payload_handler=payload_handler, visualization_config=visualization_config)

    @staticmethod
    def load_config(config_path: str | None) -> AdvCPConfig:
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

        config.setdefault("mode", "spoof")
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

    def validate_advcp_agents(self, valid_agent_ids: list[str]) -> bool:
        mode = AdvCPAttackHelper.require_config_value(self.advcp_config, "mode")
        configured_attacker_ids = AdvCPAttackHelper.resolve_configured_attacker_ids(self.advcp_config)
        attacker_ids: list[str] = []
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
        core_method = self.hypes.get("fusion", {}).get("core_method")
        return core_method in {"IntermediateFusionDataset", "IntermediateFusionDatasetV2"}

    @staticmethod
    def _inference_late_fusion_attack(*args: Any, **kwargs: Any) -> AdvCPAttackResult:
        return AdvCoperceptionLateFusionAttack.run(*args, **kwargs)

    @staticmethod
    def _inference_early_fusion_attack(*args: Any, **kwargs: Any) -> AdvCPAttackResult:
        return AdvCoperceptionEarlyFusionAttack.run(*args, **kwargs)

    @staticmethod
    def _inference_intermediate_fusion_attack(*args: Any, **kwargs: Any) -> AdvCPAttackResult:
        return AdvCoperceptionIntermediateFusionAttack.run(*args, **kwargs)
