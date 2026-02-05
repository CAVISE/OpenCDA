"""
Visualization - Handles 3D visualizations for AdvCP.
Provides visual feedback for attacks, defenses, and system behavior.
"""

import logging
import os
import datetime
from typing import Dict, List, Any
import numpy as np
import open3d as o3d

logger = logging.getLogger("cavise.advcp.visualization")


class Visualization:
    def __init__(self):
        """Initialize visualization module."""
        self.vis = None
        self.attack_geometries = []
        self.defense_geometries = []
        self.history = []

        logger.info("Visualization initialized")

    def initialize_visualizer(self):
        """Initialize Open3D visualizer."""
        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            self.vis.get_render_option().point_size = 1.0
            self.vis.get_render_option().show_coordinate_frame = True

            logger.info("Open3D visualizer initialized")

    def visualize_results(self, pred_box_tensor: Any, gt_box_tensor: Any, batch_data: Dict, tick_number: int):
        """
        Visualize prediction results with attack/defense overlays.

        Args:
            pred_box_tensor: Predicted bounding boxes
            gt_box_tensor: Ground truth bounding boxes
            batch_data: Batch data with attack/defense information
            tick_number: Current simulation tick
        """
        self.initialize_visualizer()

        # Clear previous geometries
        self.vis.clear_geometries()

        # Visualize point clouds
        for cav_id, cav_data in batch_data.items():
            if "lidar" in cav_data:
                lidar_pcd = o3d.geometry.PointCloud()
                lidar_pcd.points = o3d.utility.Vector3dVector(cav_data["lidar"][:, :3])
                lidar_pcd.paint_uniform_color([0, 0, 1])  # Blue for normal data

                # Check if CAV is under attack
                if cav_data.get("advcp_metadata", {}).get("attack_enabled", False):
                    lidar_pcd.paint_uniform_color([1, 0, 0])  # Red for attacked data

                # Check trust score for defense visualization
                trust_score = cav_data.get("advcp_metadata", {}).get("trust_scores", {}).get(cav_id, 1.0)
                if trust_score < 0.5:
                    lidar_pcd.paint_uniform_color([0, 1, 0])  # Green for trusted data

                self.vis.add_geometry(lidar_pcd)

        # Visualize bounding boxes
        if pred_box_tensor is not None:
            pred_boxes = self._tensor_to_boxes(pred_box_tensor)
            for box in pred_boxes:
                self._add_box_geometry(box, color=[1, 0.7, 0])  # Orange for predictions

        if gt_box_tensor is not None:
            gt_boxes = self._tensor_to_boxes(gt_box_tensor)
            for box in gt_boxes:
                self._add_box_geometry(box, color=[0, 1, 0])  # Green for ground truth

        # Visualize attack indicators
        self._visualize_attacks(batch_data)

        # Visualize defense indicators
        self._visualize_defenses(batch_data)

        # Update visualization
        self.vis.poll_events()
        self.vis.update_renderer()

        # Store in history
        self.history.append({"tick": tick_number, "geometries": self.vis.get_geometries(), "timestamp": datetime.now().isoformat()})

    def _tensor_to_boxes(self, box_tensor: Any) -> List[np.ndarray]:
        """Convert tensor boxes to numpy arrays."""
        boxes = []
        if box_tensor is not None:
            for i in range(box_tensor.shape[0]):
                box = box_tensor[i].cpu().numpy()
                boxes.append(box)
        return boxes

    def _add_box_geometry(self, box: np.ndarray, color: List[float]):
        """Add box geometry to visualizer."""
        bbox = o3d.geometry.OrientedBoundingBox(center=box[:3], R=self._rotation_matrix(box[6]), extent=box[3:6])
        bbox.paint_uniform_color(color)
        self.vis.add_geometry(bbox)

    def _rotation_matrix(self, yaw: float) -> np.ndarray:
        """Create rotation matrix from yaw angle."""
        return np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    def _visualize_attacks(self, batch_data: Dict):
        """Visualize active attacks."""
        for cav_id, cav_data in batch_data.items():
            if cav_data.get("advcp_metadata", {}).get("attack_enabled", False):
                # Get attack information
                attack_info = cav_data.get("advcp_metadata", {}).get("attack_info", {})

                # Visualize attack indicators
                for attack in attack_info.get("active_attacks", []):
                    # Visualize attack source
                    if "attacker_position" in attack:
                        self._add_sphere_geometry(
                            attack["attacker_position"],
                            radius=0.5,
                            color=[1, 0, 0],  # Red for attacker
                        )

                    # Visualize attack target
                    if "target_position" in attack:
                        self._add_sphere_geometry(
                            attack["target_position"],
                            radius=0.3,
                            color=[1, 0.5, 0],  # Orange for target
                        )

                    # Visualize attack path
                    if "attack_path" in attack:
                        self._add_line_geometry(attack["attacker_position"], attack["target_position"], color=[1, 0, 0])

    def _visualize_defenses(self, batch_data: Dict):
        """Visualize defense mechanisms."""
        for cav_id, cav_data in batch_data.items():
            trust_score = cav_data.get("advcp_metadata", {}).get("trust_scores", {}).get(cav_id, 1.0)

            if trust_score < 0.7:
                # Visualize low trust score
                self._add_text_geometry(
                    f"CAV {cav_id}: Trust {trust_score:.2f}", position=cav_data.get("advcp_metadata", {}).get("position", [0, 0, 0]), color=[1, 0, 0]
                )

            # Visualize blocked attacks
            if cav_data.get("advcp_metadata", {}).get("blocked_attacks", 0) > 0:
                blocked_attacks = cav_data.get("advcp_metadata", {}).get("blocked_attacks", 0)
                self._add_text_geometry(
                    f"Blocked {blocked_attacks} attacks", position=cav_data.get("advcp_metadata", {}).get("position", [0, 0, 0]), color=[0, 1, 0]
                )

    def _add_sphere_geometry(self, position: List[float], radius: float, color: List[float]):
        """Add sphere geometry to visualizer."""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        self.vis.add_geometry(sphere)

    def _add_line_geometry(self, start: List[float], end: List[float], color: List[float]):
        """Add line geometry to visualizer."""
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([start, end])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.paint_uniform_color(color)
        self.vis.add_geometry(line)

    def _add_text_geometry(self, text: str, position: List[float], color: List[float]):
        """Add text geometry to visualizer."""
        text_3d = o3d.geometry.Text(text, position)
        text_3d.color = color
        self.vis.add_geometry(text_3d)

    def save_visualizations(self, directory: str, tick_number: int):
        """
        Save current visualizations to files.

        Args:
            directory: Directory to save visualizations
            tick_number: Current simulation tick
        """
        if self.vis is None:
            return

        # Create directory
        os.makedirs(directory, exist_ok=True)

        # Save screenshot
        screenshot_path = os.path.join(directory, f"visualization_{tick_number:06d}.png")
        self.vis.capture_screen_image(screenshot_path)

        # Save point cloud data
        # pointcloud_path = os.path.join(directory, f"pointcloud_{tick_number:06d}.pcd")
        # (Implementation would save point cloud data)

        logger.info("Saved visualizations to %s", directory)

    def cleanup(self):
        """Cleanup visualization resources."""
        logger.info("Cleaning up Visualization")

        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None

        self.attack_geometries.clear()
        self.defense_geometries.clear()
        self.history.clear()

        logger.info("Visualization cleanup completed")
