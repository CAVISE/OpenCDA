"""
AdvCP Visualization Manager - manages visualization for attacks and defenses.

This module provides visualization capabilities for the AdvCP (Advanced Collaborative Perception)
module, activated only when --advcp-vis CLI flag is set.
"""

import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from mvp.visualize.general import (
    draw_multi_vehicle_case,
)
from mvp.visualize.attack import draw_attack
from mvp.visualize.defense import (
    draw_ground_segmentation,
    draw_polygon_areas,
    draw_object_tracking,
    visualize_defense,
)
from mvp.visualize.evaluate import draw_distribution, draw_detection_roc

logger = logging.getLogger("cavise.advcp_visualization_manager")


class AdvCPVisualizationManager:
    """
    Manages visualization for AdvCP attacks and defenses.

    This class is activated only when --advcp-vis flag is set in CLI.
    It collects data during simulation ticks and generates visualizations
    using the functions from mvp/visualize module.

    Attributes:
        enabled: Whether visualization is enabled
        mode: Visualization backend ('matplotlib', 'open3d', or 'both')
        save: Whether to save visualizations to disk
        show: Whether to show interactive visualizations
        output_dir: Directory for saving visualization outputs
        vis_types: List of visualization types to generate
    """

    def __init__(self, opt: Dict[str, Any], output_dir: str = "simulation_output/advcp_vis") -> None:
        """
        Initialize AdvCP Visualization Manager.

        Args:
            opt: Configuration options dictionary containing visualization settings
            output_dir: Directory to save visualization outputs
        """
        self.opt = opt
        self.enabled = opt.get("advcp_vis", False)
        self.mode = opt.get("advcp_vis_mode", "matplotlib")
        self.save = opt.get("advcp_vis_save", True)
        self.show = opt.get("advcp_vis_show", True)
        self.vis_types = opt.get("advcp_vis_types", ["attack", "defense"])

        self.output_dir = output_dir
        self.tick_counter = 0

        # Data storage for multi-frame visualization and evaluation
        self.attack_history: List[Dict] = []
        self.defense_history: List[Dict] = []
        self.normal_case_history: List[Dict] = []
        self.attack_case_history: List[Dict] = []

        # Storage for ROC evaluation
        self.normal_values: List[float] = []
        self.attack_values: List[float] = []

        # Storage for tracking visualization
        self.tracking_point_clouds: List[np.ndarray] = []
        self.tracking_detections: List[np.ndarray] = []
        self.tracking_predictions: List[np.ndarray] = []

        if self.enabled:
            os.makedirs(output_dir, exist_ok=True)
            logger.info("AdvCP Visualization Manager initialized:")
            logger.info(f"  mode: {self.mode}")
            logger.info(f"  save: {self.save}")
            logger.info(f"  show: {self.show}")
            logger.info(f"  output_dir: {output_dir}")
            logger.info(f"  vis_types: {self.vis_types}")

    def process_tick(
        self,
        tick_number: int,
        raw_data: Optional[Dict] = None,
        attacked_data: Optional[Dict] = None,
        defended_data: Optional[Dict] = None,
        attack_info: Optional[Dict] = None,
        defense_metrics: Optional[Dict] = None,
        predictions: Optional[Dict] = None,
        ground_inliers: Optional[np.ndarray] = None,
    ) -> None:
        """
        Collect and visualize data for a single simulation tick.

        Args:
            tick_number: Current simulation tick number
            raw_data: Original perception data before attack
            attacked_data: Data after attack was applied
            defended_data: Data after defense was applied
            attack_info: Information about the attack (metadata, target, etc.)
            defense_metrics: Metrics from defense mechanism
            predictions: Prediction results (bboxes, scores)
            ground_inliers: Ground segmentation inliers indices
        """
        if not self.enabled:
            return

        self.tick_counter = tick_number

        # Store data for multi-frame analysis
        if raw_data:
            self.normal_case_history.append({tick_number: raw_data})
        if attacked_data:
            self.attack_case_history.append({tick_number: attacked_data})
        if attack_info:
            self.attack_history.append(attack_info)
        if defense_metrics:
            self.defense_history.append(defense_metrics)

        # Generate visualizations based on vis_types
        if "attack" in self.vis_types and attack_info:
            self._visualize_attack(tick_number, raw_data, attacked_data, attack_info)

        if "defense" in self.vis_types and defense_metrics:
            self._visualize_defense(tick_number, attacked_data, defended_data, defense_metrics)

        if "ground_seg" in self.vis_types and raw_data and ground_inliers is not None:
            self._visualize_ground_segmentation(tick_number, raw_data, ground_inliers)

        if "tracking" in self.vis_types and raw_data and predictions:
            self._collect_tracking_data(tick_number, raw_data, predictions)

        if "roc" in self.vis_types and defense_metrics:
            self._collect_roc_data(defense_metrics)

    def _visualize_attack(self, tick_number: int, normal_case: Optional[Dict], attack_case: Optional[Dict], attack_info: Dict) -> None:
        """
        Generate attack visualization comparing normal vs attack case.

        Args:
            tick_number: Current simulation tick number
            normal_case: Original perception data
            attack_case: Data after attack
            attack_info: Attack metadata
        """
        if normal_case is None or attack_case is None:
            return

        save_path = None
        if self.save:
            save_path = os.path.join(self.output_dir, f"attack_tick_{tick_number:04d}.png")

        try:
            draw_attack(
                attack=attack_info,
                normal_case={tick_number: normal_case},
                attack_case={tick_number: attack_case},
                mode="multi_frame",
                show=self.show,
                save=save_path,
            )
            logger.debug(f"Generated attack visualization for tick {tick_number}")
        except Exception as e:
            logger.warning(f"Failed to generate attack visualization: {e}")

    def _visualize_defense(self, tick_number: int, attacked_data: Optional[Dict], defended_data: Optional[Dict], defense_metrics: Dict) -> None:
        """
        Generate defense visualization showing occupied/free areas and error regions.

        Args:
            tick_number: Current simulation tick number
            attacked_data: Data after attack
            defended_data: Data after defense
            defense_metrics: Defense metrics including error areas
        """
        case_data = defended_data if defended_data else attacked_data
        if case_data is None:
            return

        save_path = None
        if self.save:
            save_path = os.path.join(self.output_dir, f"defense_tick_{tick_number:04d}.png")

        try:
            visualize_defense(case=[{tick_number: case_data}], metrics=[defense_metrics], show=self.show, save=save_path)
            logger.debug(f"Generated defense visualization for tick {tick_number}")
        except Exception as e:
            logger.warning(f"Failed to generate defense visualization: {e}")

    def _visualize_ground_segmentation(self, tick_number: int, raw_data: Optional[Dict], ground_inliers: np.ndarray) -> None:
        """
        Visualize ground segmentation results.

        Args:
            tick_number: Current simulation tick number
            raw_data: Raw perception data containing LiDAR
            ground_inliers: Indices of ground points
        """
        if raw_data is None:
            return

        # Extract point cloud from first vehicle
        pcd_data = None
        for vehicle_id, vehicle_data in raw_data.items():
            if "lidar" in vehicle_data and vehicle_data["lidar"] is not None:
                pcd_data = vehicle_data["lidar"]
                break

        if pcd_data is None:
            return

        save_path = None
        if self.save:
            save_path = os.path.join(self.output_dir, f"ground_seg_tick_{tick_number:04d}.png")

        try:
            draw_ground_segmentation(pcd_data=pcd_data, inliers=ground_inliers, show=self.show, save=save_path)
            logger.debug(f"Generated ground segmentation visualization for tick {tick_number}")
        except Exception as e:
            logger.warning(f"Failed to generate ground segmentation visualization: {e}")

    def _collect_tracking_data(self, tick_number: int, raw_data: Dict, predictions: Dict) -> None:
        """
        Collect data for object tracking visualization.

        Args:
            tick_number: Current simulation tick number
            raw_data: Raw perception data
            predictions: Prediction results
        """
        # Extract point cloud
        for vehicle_id, vehicle_data in raw_data.items():
            if "lidar" in vehicle_data and vehicle_data["lidar"] is not None:
                self.tracking_point_clouds.append(vehicle_data["lidar"][:, :3])
                break

        # Extract detections and predictions
        if "pred_bboxes" in predictions:
            pred_bboxes = predictions["pred_bboxes"]
            if hasattr(pred_bboxes, "cpu"):
                pred_bboxes = pred_bboxes.cpu().numpy()
            # Extract centers for visualization
            centers = pred_bboxes[:, :3] if len(pred_bboxes) > 0 else np.array([])
            self.tracking_detections.append(centers)
            self.tracking_predictions.append(centers)  # Same for now

    def _collect_roc_data(self, defense_metrics: Dict) -> None:
        """
        Collect data for ROC curve generation.

        Args:
            defense_metrics: Defense metrics containing scores
        """
        # Extract scores from defense metrics
        # This depends on the structure of defense_metrics
        if "scores" in defense_metrics:
            scores = defense_metrics["scores"]
            labels = defense_metrics.get("labels", [])

            for score, label in zip(scores, labels):
                if label > 0:
                    self.attack_values.append(score)
                else:
                    self.normal_values.append(score)

    def visualize_multi_vehicle_case(
        self,
        tick_number: int,
        case: Dict,
        ego_id: str,
        gt_bboxes: Optional[np.ndarray] = None,
        pred_bboxes: Optional[np.ndarray] = None,
        system: str = "map",
    ) -> None:
        """
        Generate multi-vehicle case visualization.

        Args:
            tick_number: Current simulation tick number
            case: Multi-vehicle case data
            ego_id: Ego vehicle ID
            gt_bboxes: Ground truth bounding boxes
            pred_bboxes: Predicted bounding boxes
            system: Coordinate system ('map' or 'ego')
        """
        if not self.enabled:
            return

        save_path = None
        if self.save:
            save_path = os.path.join(self.output_dir, f"multi_vehicle_tick_{tick_number:04d}.png")

        try:
            if self.mode in ["matplotlib", "both"]:
                draw_multi_vehicle_case(
                    case=case,
                    ego_id=ego_id,
                    mode="matplotlib",
                    gt_bboxes=gt_bboxes,
                    pred_bboxes=pred_bboxes,
                    system=system,
                    show=self.show,
                    save=save_path,
                )

            if self.mode in ["open3d", "both"]:
                draw_multi_vehicle_case(
                    case=case,
                    ego_id=ego_id,
                    mode="open3d",
                    gt_bboxes=gt_bboxes,
                    pred_bboxes=pred_bboxes,
                    system=system,
                    show=self.show and self.mode == "open3d",
                    save=None,  # Open3D doesn't support save directly
                )

            logger.debug(f"Generated multi-vehicle visualization for tick {tick_number}")
        except Exception as e:
            logger.warning(f"Failed to generate multi-vehicle visualization: {e}")

    def visualize_polygon_areas(self, tick_number: int, case: Dict, tag: str = "") -> None:
        """
        Visualize polygon areas (free/occupied areas from defense).

        Args:
            tick_number: Current simulation tick number
            case: Case data with polygon areas
            tag: Optional tag for filename
        """
        if not self.enabled or "ground_seg" not in self.vis_types:
            return

        save_path = None
        if self.save:
            save_path = os.path.join(self.output_dir, f"polygon_areas_{tag}tick_{tick_number:04d}.png")

        try:
            draw_polygon_areas(case=case, show=self.show, save=save_path, tag=tag)
            logger.debug(f"Generated polygon areas visualization for tick {tick_number}")
        except Exception as e:
            logger.warning(f"Failed to generate polygon areas visualization: {e}")

    def visualize_object_tracking(self, tick_number: int) -> None:
        """
        Generate object tracking visualization from collected data.

        Args:
            tick_number: Current simulation tick number (for filename)
        """
        if not self.enabled or "tracking" not in self.vis_types:
            return

        if len(self.tracking_point_clouds) == 0:
            return

        save_path = None
        if self.save:
            save_path = os.path.join(self.output_dir, f"tracking_tick_{tick_number:04d}.png")

        try:
            draw_object_tracking(
                point_clouds=self.tracking_point_clouds,
                detections=self.tracking_detections,
                predictions=self.tracking_predictions,
                show=self.show,
                save=save_path,
            )
            logger.debug(f"Generated object tracking visualization for tick {tick_number}")
        except Exception as e:
            logger.warning(f"Failed to generate object tracking visualization: {e}")

    def generate_final_report(self) -> Dict[str, str]:
        """
        Generate final evaluation visualizations after simulation ends.

        This method should be called at the end of simulation to generate
        aggregate visualizations like ROC curves and distribution plots.

        Returns:
            Dictionary mapping visualization type to output file path
        """
        output_files: Dict[str, str] = {}

        if not self.enabled:
            return output_files

        # Generate ROC curve if we have collected data
        if "roc" in self.vis_types and len(self.normal_values) > 0 and len(self.attack_values) > 0:
            roc_path = os.path.join(self.output_dir, "roc_curve.png")
            try:
                normal_arr = np.array(self.normal_values)
                attack_arr = np.array(self.attack_values)
                draw_detection_roc(normal_values=normal_arr, attack_values=attack_arr, show=self.show, save=roc_path)
                output_files["roc"] = roc_path
                logger.info(f"Generated ROC curve: {roc_path}")
            except Exception as e:
                logger.warning(f"Failed to generate ROC curve: {e}")

        # Generate distribution plots
        if "evaluation" in self.vis_types:
            if len(self.normal_values) > 0 or len(self.attack_values) > 0:
                dist_path = os.path.join(self.output_dir, "distribution.png")
                try:
                    data_items = []
                    labels = []
                    if len(self.normal_values) > 0:
                        data_items.append(np.array(self.normal_values))
                        labels.append("Normal")
                    if len(self.attack_values) > 0:
                        data_items.append(np.array(self.attack_values))
                        labels.append("Attack")

                    if len(data_items) > 0:
                        draw_distribution(data_items=data_items, labels=labels, show=self.show, save=dist_path, bins=50)
                        output_files["distribution"] = dist_path
                        logger.info(f"Generated distribution plot: {dist_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate distribution plot: {e}")

        # Generate final tracking visualization
        if "tracking" in self.vis_types and len(self.tracking_point_clouds) > 0:
            tracking_path = os.path.join(self.output_dir, "tracking_final.png")
            try:
                draw_object_tracking(
                    point_clouds=self.tracking_point_clouds,
                    detections=self.tracking_detections,
                    predictions=self.tracking_predictions,
                    show=self.show,
                    save=tracking_path,
                )
                output_files["tracking"] = tracking_path
                logger.info(f"Generated tracking visualization: {tracking_path}")
            except Exception as e:
                logger.warning(f"Failed to generate tracking visualization: {e}")

        logger.info(f"Generated final report files: {list(output_files.keys())}")
        return output_files

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected visualization data.

        Returns:
            Dictionary with statistics about collected data
        """
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "ticks_processed": self.tick_counter,
            "attack_cases_collected": len(self.attack_history),
            "defense_cases_collected": len(self.defense_history),
            "normal_values_collected": len(self.normal_values),
            "attack_values_collected": len(self.attack_values),
            "tracking_frames_collected": len(self.tracking_point_clouds),
        }

    def clear_history(self) -> None:
        """Clear all collected history data."""
        self.attack_history.clear()
        self.defense_history.clear()
        self.normal_case_history.clear()
        self.attack_case_history.clear()
        self.normal_values.clear()
        self.attack_values.clear()
        self.tracking_point_clouds.clear()
        self.tracking_detections.clear()
        self.tracking_predictions.clear()
        logger.debug("Cleared all visualization history data")
