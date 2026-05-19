"""Attacker-to-benign target visibility ratio for AdvCP attacks."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Mapping

import numpy as np
import numpy.typing as npt

from opencda.metrics_tools.base_metric import BaseMetric
from opencda.metrics_tools.collection_models import MetricSeries
from opencda.metrics_tools.metric_sample import MetricSample
from opencda.metrics_tools.report_models import MetricReportSpec, MetricSummarySpec
from opencda.core.attack.advcp.types import AdvCPVisualizationContext

logger = logging.getLogger("cavise.opencda.opencda.metrics_tools.metrics.coperception.attacker_benign_visibility_ratio")


def _load_advcp_attack_helper() -> Any:
    from opencda.core.attack.advcp.attack_helper import AdvCPAttackHelper

    return AdvCPAttackHelper


class AttackerBenignVisibilityRatioMetric(BaseMetric):
    """Measure how much of the attack target is observed by attackers versus benign agents."""

    metric_name = "attacker_benign_visibility_ratio"
    _ATTACK_MODES: ClassVar[set[str]] = {"removal", "spoofing"}
    _RATIO_SERIES = "attacker_benign_visibility_ratio"
    _ATTACKER_POINTS_SERIES = "attacker_points_on_target"
    _BENIGN_POINTS_SERIES = "benign_points_on_target"

    def __init__(self, warmup_steps: int = 0, epsilon: float = 1.0):
        super().__init__(warmup_steps=warmup_steps)
        self.epsilon = float(epsilon)
        self._series_samples: dict[str, list[MetricSample]] = {
            self._RATIO_SERIES: [],
            self._ATTACKER_POINTS_SERIES: [],
            self._BENIGN_POINTS_SERIES: [],
        }

    def _process_context(self, context: Mapping[str, Any]) -> None:
        visualization_context = context.get("visualization_context")
        if not isinstance(visualization_context, AdvCPVisualizationContext):
            return

        mode = self._normalize_mode(visualization_context.mode)
        if mode not in self._ATTACK_MODES:
            return

        attacker_ids = self._resolve_attacker_ids(visualization_context)
        if not attacker_ids:
            return

        advcp_config = context.get("advcp_config")
        memory_data = context.get("memory_data")
        if advcp_config is None or memory_data is None:
            return

        visibility = self._compute_visibility(advcp_config, memory_data, attacker_ids)
        if visibility is None:
            return

        attacker_points, benign_points = visibility
        ratio = attacker_points / (benign_points + self.epsilon)
        self._series_samples[self._RATIO_SERIES].append(self._make_sample(ratio))
        self._series_samples[self._ATTACKER_POINTS_SERIES].append(self._make_sample(attacker_points))
        self._series_samples[self._BENIGN_POINTS_SERIES].append(self._make_sample(benign_points))

    def get_raw(self) -> tuple[MetricSeries, ...]:
        return tuple(
            MetricSeries(name=series_name, samples=tuple(self._series_samples[series_name]))
            for series_name in (self._RATIO_SERIES, self._ATTACKER_POINTS_SERIES, self._BENIGN_POINTS_SERIES)
        )

    @classmethod
    def get_report_spec(cls) -> MetricReportSpec:
        return MetricReportSpec(
            metric_name=cls.metric_name,
            display_name="Attacker-Benign Visibility Ratio",
            series_names=(cls._RATIO_SERIES, cls._ATTACKER_POINTS_SERIES, cls._BENIGN_POINTS_SERIES),
            summary_specs=(
                MetricSummarySpec(
                    series_name=cls._RATIO_SERIES,
                    display_name="Attacker-Benign Visibility Ratio",
                ),
                MetricSummarySpec(
                    series_name=cls._ATTACKER_POINTS_SERIES,
                    display_name="Attacker Points on Target",
                ),
                MetricSummarySpec(
                    series_name=cls._BENIGN_POINTS_SERIES,
                    display_name="Benign Points on Target",
                ),
            ),
        )

    def _compute_visibility(self, advcp_config: Any, memory_data: Any, attacker_ids: set[str]) -> tuple[float, float] | None:
        try:
            attack_helper = _load_advcp_attack_helper()
            scenario_data = next(iter(memory_data.values()))
            ego_agent_id = attack_helper.resolve_ego_agent_id(scenario_data)
            ego_state = attack_helper.load_agent_state(scenario_data, ego_agent_id)
        except Exception as error:
            logger.debug("Unable to initialize attacker/benign visibility metric context: %s", error)
            return None

        attacker_points = 0
        benign_points = 0
        for agent_id in scenario_data:
            try:
                agent_state = attack_helper.load_agent_state(scenario_data, agent_id)
                agent_snapshot = attack_helper.resolve_agent_snapshot(scenario_data, agent_id)
                lidar = np.asarray(agent_snapshot.get("lidar_np"), dtype=np.float32)
                target_boxes = self._resolve_target_boxes_for_agent(
                    attack_helper,
                    advcp_config,
                    ego_state,
                    agent_state["lidar_pose"],
                )
            except Exception as error:
                logger.debug("Skipping attacker/benign visibility for agent '%s': %s", agent_id, error)
                continue

            points_on_target = self._count_points_in_target_boxes(lidar, target_boxes)
            if str(agent_id) in attacker_ids:
                attacker_points += points_on_target
            else:
                benign_points += points_on_target

        return float(attacker_points), float(benign_points)

    @staticmethod
    def _resolve_target_boxes_for_agent(
        attack_helper: Any,
        advcp_config: Any,
        ego_state: Mapping[str, Any],
        lidar_pose: Any,
    ) -> list[npt.NDArray[np.float32]]:
        box_specs = attack_helper.require_config_value(advcp_config, "boxes")
        if not isinstance(box_specs, list) or len(box_specs) == 0:
            return []
        return [
            attack_helper.resolve_box_spec_for_sensor_pose(
                spec,
                index,
                advcp_config,
                ego_state,
                lidar_pose,
            )
            for index, spec in enumerate(box_specs)
        ]

    @classmethod
    def _count_points_in_target_boxes(cls, lidar: npt.NDArray[np.float32], target_boxes: list[npt.NDArray[np.float32]]) -> int:
        if lidar.size == 0 or lidar.ndim != 2 or lidar.shape[1] < 3 or not target_boxes:
            return 0
        points_xyz = np.asarray(lidar[:, :3], dtype=np.float32)
        inside_any_target = np.zeros((points_xyz.shape[0],), dtype=np.bool_)
        for target_box in target_boxes:
            if np.asarray(target_box).shape != (7,):
                continue
            inside_any_target |= cls._compute_points_inside_box_mask(points_xyz, np.asarray(target_box, dtype=np.float32))
        return int(np.count_nonzero(inside_any_target))

    @staticmethod
    def _compute_points_inside_box_mask(
        points_xyz: npt.NDArray[np.float32],
        box_lwh_bottom_center: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.bool_]:
        center_x, center_y, center_z = [float(value) for value in box_lwh_bottom_center[:3]]
        length, width, height = [float(value) for value in box_lwh_bottom_center[3:6]]
        yaw = float(box_lwh_bottom_center[6])

        translated_x = points_xyz[:, 0] - center_x
        translated_y = points_xyz[:, 1] - center_y
        translated_z = points_xyz[:, 2] - center_z

        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        local_x = cos_yaw * translated_x + sin_yaw * translated_y
        local_y = -sin_yaw * translated_x + cos_yaw * translated_y

        half_length = length / 2.0
        half_width = width / 2.0
        epsilon = 1e-4
        return (
            (np.abs(local_x) <= (half_length + epsilon))
            & (np.abs(local_y) <= (half_width + epsilon))
            & (translated_z >= -epsilon)
            & (translated_z <= (height + epsilon))
        )

    @staticmethod
    def _resolve_attacker_ids(visualization_context: AdvCPVisualizationContext) -> set[str]:
        return {str(attacker_id) for attacker_id in visualization_context.attacker_ids}

    @staticmethod
    def _normalize_mode(mode: Any) -> str:
        return "" if mode is None else str(mode).strip().lower()
