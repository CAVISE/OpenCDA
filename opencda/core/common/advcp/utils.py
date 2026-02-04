"""
AdvCPUtils - Helper functions for AdvCollaborativePerception.
Provides utility functions for attack/defense operations.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("cavise.advcp.utils")


class AdvCPUtils:
    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box [x, y, z, l, w, h, yaw]
            box2: Second bounding box [x, y, z, l, w, h, yaw]
            
        Returns:
            IoU value between 0 and 1
        """
        # Convert boxes to corner points
        corners1 = AdvCPUtils._box_to_corners(box1)
        corners2 = AdvCPUtils._box_to_corners(box2)
        
        # Create polygons
        poly1 = AdvCPUtils._corners_to_polygon(corners1)
        poly2 = AdvCPUtils._corners_to_polygon(corners2)
        
        # Calculate intersection and union
        intersection = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def _box_to_corners(box: np.ndarray) -> np.ndarray:
        """Convert box to corner points."""
        x, y, z, l, w, h, yaw = box
        
        # Calculate half dimensions
        l2 = l / 2
        w2 = w / 2
        h2 = h / 2
        
        # Define corners relative to center
        corners = np.array([
            [-l2, -w2, -h2],
            [l2, -w2, -h2],
            [l2, w2, -h2],
            [-l2, w2, -h2],
            [-l2, -w2, h2],
            [l2, -w2, h2],
            [l2, w2, h2],
            [-l2, w2, h2]
        ])
        
        # Apply rotation around z-axis (yaw)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        rotated_corners = np.dot(corners, rotation_matrix.T)
        
        # Translate to absolute position
        absolute_corners = rotated_corners + np.array([x, y, z])
        
        return absolute_corners
    
    @staticmethod
    def _corners_to_polygon(corners: np.ndarray) -> Any:
        """Convert corners to polygon."""
        # Use bottom face (z = min)
        bottom_corners = corners[corners[:, 2] == np.min(corners[:, 2])]
        
        # Project to 2D (x, y)
        points_2d = bottom_corners[:, :2]
        
        # Create polygon
        from shapely.geometry import Polygon
        return Polygon(points_2d)
    
    @staticmethod
    def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point [x, y, z]
            point2: Second point [x, y, z]
            
        Returns:
            Distance between points
        """
        return np.linalg.norm(point1 - point2)
    
    @staticmethod
    def calculate_angle_difference(angle1: float, angle2: float) -> float:
        """
        Calculate angular difference between two angles.
        
        Args:
            angle1: First angle in radians
            angle2: Second angle in radians
            
        Returns:
            Angular difference in radians
        """
        diff = np.abs(angle1 - angle2)
        return np.minimum(diff, 2 * np.pi - diff)
    
    @staticmethod
    def normalize_point_cloud(point_cloud: np.ndarray, bounds: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize point cloud to [-1, 1] range.
        
        Args:
            point_cloud: Input point cloud [N, 3]
            bounds: Optional bounds [min_x, min_y, min_z, max_x, max_y, max_z]
            
        Returns:
            Normalized point cloud
        """
        if point_cloud.size == 0:
            return point_cloud
        
        # Calculate bounds if not provided
        if bounds is None:
            min_vals = np.min(point_cloud, axis=0)
            max_vals = np.max(point_cloud, axis=0)
        else:
            min_vals = bounds[:3]
            max_vals = bounds[3:]
        
        # Normalize to [-1, 1]
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        normalized = 2 * (point_cloud - min_vals) / range_vals - 1
        
        return normalized
    
    @staticmethod
    def generate_occupancy_map(point_cloud: np.ndarray, resolution: float = 0.5) -> np.ndarray:
        """
        Generate 2D occupancy map from point cloud.
        
        Args:
            point_cloud: Input point cloud [N, 3]
            resolution: Grid resolution in meters
            
        Returns:
            2D occupancy map as numpy array
        """
        if point_cloud.size == 0:
            return np.zeros((100, 100))
        
        # Project to 2D (x, y)
        points_2d = point_cloud[:, :2]
        
        # Calculate grid bounds
        min_x, min_y = np.min(points_2d, axis=0)
        max_x, max_y = np.max(points_2d, axis=0)
        
        # Calculate grid size
        grid_width = int(np.ceil((max_x - min_x) / resolution)) + 1
        grid_height = int(np.ceil((max_y - min_y) / resolution)) + 1
        
        # Create occupancy map
        occupancy_map = np.zeros((grid_height, grid_width))
        
        # Fill occupancy map
        for point in points_2d:
            x_idx = int((point[0] - min_x) / resolution)
            y_idx = int((point[1] - min_y) / resolution)
            
            if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                occupancy_map[y_idx, x_idx] = 1
        
        return occupancy_map
    
    @staticmethod
    def calculate_point_cloud_features(point_cloud: np.ndarray) -> Dict[str, float]:
        """
        Calculate features from point cloud.
        
        Args:
            point_cloud: Input point cloud [N, 3]
            
        Returns:
            Dictionary of point cloud features
        """
        features = {}
        
        if point_cloud.size == 0:
            return features
        
        # Basic statistics
        points = point_cloud[:, :3]
        
        features["num_points"] = len(points)
        features["centroid"] = np.mean(points, axis=0).tolist()
        features["std_dev"] = np.std(points, axis=0).tolist()
        features["density"] = len(points) / np.prod(np.ptp(points, axis=0))
        
        # Range features
        distances = np.linalg.norm(points, axis=1)
        features["min_range"] = float(np.min(distances))
        features["max_range"] = float(np.max(distances))
        features["avg_range"] = float(np.mean(distances))
        
        # Angular features (if available)
        if point_cloud.shape[1] > 3:
            intensities = point_cloud[:, 3]
            features["avg_intensity"] = float(np.mean(intensities))
            features["std_intensity"] = float(np.std(intensities))
        
        return features
    
    @staticmethod
    def detect_anomalies(point_cloud: np.ndarray, reference_cloud: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect anomalies in point cloud compared to reference.
        
        Args:
            point_cloud: Input point cloud to check
            reference_cloud: Reference point cloud
            threshold: Anomaly detection threshold
            
        Returns:
            Dictionary with anomaly detection results
        """
        results = {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "num_anomalous_points": 0,
            "anomalous_points": []
        }
        
        if point_cloud.size == 0 or reference_cloud.size == 0:
            return results
        
        # Calculate point-to-point distances
        distances = []
        for point in point_cloud:
            # Find nearest neighbor in reference cloud
            distances_to_ref = np.linalg.norm(reference_cloud - point, axis=1)
            min_distance = np.min(distances_to_ref)
            distances.append(min_distance)
        
        # Calculate anomaly score
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Anomaly score based on distance from reference
        anomaly_score = avg_distance / (std_distance + 1e-6)
        
        # Detect anomalous points
        anomalous_points = [i for i, d in enumerate(distances) if d > threshold]
        
        results["is_anomaly"] = anomaly_score > threshold
        results["anomaly_score"] = anomaly_score
        results["num_anomalous_points"] = len(anomalous_points)
        results["anomalous_points"] = anomalous_points
        
        return results
    
    @staticmethod
    def calculate_trust_score(consistency_score: float, anomaly_score: float, base_trust: float = 1.0) -> float:
        """
        Calculate trust score based on consistency and anomaly scores.
        
        Args:
            consistency_score: Consistency score (0-1)
            anomaly_score: Anomaly score (0-1)
            base_trust: Base trust value
            
        Returns:
            Trust score (0-1)
        """
        # Combine scores (higher consistency and lower anomaly = higher trust)
        trust = base_trust * consistency_score * (1.0 - anomaly_score)
        
        # Clamp to [0, 1]
        return np.clip(trust, 0.0, 1.0)
    
    @staticmethod
    def generate_attack_signature(attack_type: str, attacker_id: int, target_id: int, tick: int) -> str:
        """
        Generate unique attack signature.
        
        Args:
            attack_type: Type of attack
            attacker_id: ID of attacker CAV
            target_id: ID of target CAV
            tick: Simulation tick
            
        Returns:
            Unique attack signature string
        """
        return f"{attack_type}_{attacker_id}>{target_id}_{tick}"
    
    @staticmethod
    def parse_attack_signature(signature: str) -> Dict[str, Any]:
        """
        Parse attack signature into components.
        
        Args:
            signature: Attack signature string
            
        Returns:
            Dictionary with attack components
        """
        parts = signature.split("_")
        
        if len(parts) != 3:
            return {}
        
        attack_type = parts[0]
        
        # Parse attacker>target
        attacker_target = parts[1].split(">")
        if len(attacker_target) != 2:
            return {}
        
        attacker_id = int(attacker_target[0])
        target_id = int(attacker_target[1])
        
        tick = int(parts[2])
        
        return {
            "attack_type": attack_type,
            "attacker_id": attacker_id,
            "target_id": target_id,
            "tick": tick
        }
    
    @staticmethod
    def create_empty_cav_data() -> Dict[str, Any]:
        """
        Create empty CAV data structure.
        
        Returns:
            Empty CAV data dictionary
        """
        return {
            "lidar": np.zeros((0, 4)),  # Empty LiDAR data
            "pred_bboxes": np.zeros((0, 7)),  # Empty predictions
            "pred_scores": np.zeros(0),  # Empty scores
            "gt_bboxes": np.zeros((0, 7)),  # Empty ground truth
            "advcp_metadata": {
                "attack_enabled": False,
                "defense_enabled": False,
                "trust_scores": {},
                "attack_info": {},
                "blocked_attacks": 0
            }
        }
    
    @staticmethod
    def merge_point_clouds(point_clouds: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Merge multiple point clouds with optional weighting.
        
        Args:
            point_clouds: List of point clouds to merge
            weights: Optional weights for each point cloud
            
        Returns:
            Merged point cloud
        """
        if not point_clouds:
            return np.zeros((0, 4))
        
        # Apply weights if provided
        if weights is not None:
            weighted_clouds = []
            for cloud, weight in zip(point_clouds, weights):
                weighted_clouds.append(cloud * weight)
            
            # Combine weighted clouds
            merged_cloud = np.vstack(weighted_clouds)
        else:
            # Simple concatenation
            merged_cloud = np.vstack(point_clouds)
        
        return merged_cloud
    
    @staticmethod
    def filter_point_cloud(point_cloud: np.ndarray, min_intensity: float = 0.1, max_range: float = 50.0) -> np.ndarray:
        """
        Filter point cloud based on intensity and range.
        
        Args:
            point_cloud: Input point cloud
            min_intensity: Minimum intensity threshold
            max_range: Maximum range threshold
            
        Returns:
            Filtered point cloud
        """
        if point_cloud.size == 0:
            return point_cloud
        
        # Filter by intensity
        if point_cloud.shape[1] > 3:
            intensity_mask = point_cloud[:, 3] > min_intensity
        else:
            intensity_mask = np.ones(len(point_cloud), dtype=bool)
        
        # Filter by range
        distances = np.linalg.norm(point_cloud[:, :3], axis=1)
        range_mask = distances < max_range
        
        # Combine masks
        mask = intensity_mask & range_mask
        
        return point_cloud[mask]
    
    @staticmethod
    def calculate_detection_metrics(pred_bboxes: np.ndarray, gt_bboxes: np.ndarray, iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate detection metrics (precision, recall, F1-score).
        
        Args:
            pred_bboxes: Predicted bounding boxes
            gt_bboxes: Ground truth bounding boxes
            iou_threshold: IoU threshold for matching
            
        Returns:
            Dictionary with detection metrics
        """
        if len(pred_bboxes) == 0 and len(gt_bboxes) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
        
        if len(pred_bboxes) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        if len(gt_bboxes) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_bboxes), len(gt_bboxes)))
        for i, pred in enumerate(pred_bboxes):
            for j, gt in enumerate(gt_bboxes):
                iou_matrix[i, j] = AdvCPUtils.calculate_iou(pred, gt)
        
        # Find matches
        matched_preds = set()
        matched_gts = set()
        
        for i in range(len(pred_bboxes)):
            for j in range(len(gt_bboxes)):
                if iou_matrix[i, j] > iou_threshold:
                    matched_preds.add(i)
                    matched_gts.add(j)
        
        # Calculate metrics
        tp = len(matched_preds)
        fp = len(pred_bboxes) - tp
        fn = len(gt_bboxes) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }