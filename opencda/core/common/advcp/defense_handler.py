"""
DefenseHandler - Manages CAD defense mechanisms for AdvCP.
Implements consistency checking and anomaly detection for collaborative perception.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import open3d as o3d
from shapely.geometry import Polygon, MultiPolygon

from .utils import AdvCPUtils

logger = logging.getLogger("cavise.advcp.defense_handler")


class DefenseHandler:
    def __init__(self, config: Dict):
        """
        Initialize defense handler with configuration.
        
        Args:
            config: AdvCP configuration dictionary
        """
        self.config = config
        self.defense_enabled = config.get("defense", {}).get("enabled", False)
        self.trust_threshold = config.get("defense", {}).get("trust_threshold", 0.7)
        self.consistency_window = config.get("defense", {}).get("consistency_window", 5)
        
        # Initialize defense components
        self.occupancy_map = OccupancyMap(config)
        self.consistency_checker = ConsistencyChecker(config)
        self.anomaly_detector = AnomalyDetector(config)
        
        # Track trust scores and history
        self.trust_scores = {}
        self.history = []
        self.blocked_attacks = 0
        
        logger.info("DefenseHandler initialized with trust threshold: %.2f", self.trust_threshold)
    
    def apply_defense(self, batch_data: Dict, tick_number: int) -> Dict:
        """
        Apply defense mechanisms to the batch data.
        
        Args:
            batch_data: Current batch data
            tick_number: Current simulation tick
            
        Returns:
            Modified batch data with defense applied
        """
        if not self.defense_enabled:
            return batch_data
        
        modified_data = batch_data.copy()
        
        # Update occupancy map
        self.occupancy_map.update(batch_data, tick_number)
        
        # Check consistency
        consistency_results = self.consistency_checker.check_consistency(
            batch_data, tick_number
        )
        
        # Detect anomalies
        anomaly_results = self.anomaly_detector.detect_anomalies(
            batch_data, consistency_results, tick_number
        )
        
        # Update trust scores
        self._update_trust_scores(batch_data, anomaly_results, tick_number)
        
        # Filter malicious data
        modified_data = self._filter_malicious_data(
            modified_data, anomaly_results, tick_number
        )
        
        # Log defense results
        self._log_defense_results(tick_number, consistency_results, anomaly_results)
        
        return modified_data
    
    def _update_trust_scores(self, batch_data: Dict, anomaly_results: Dict, tick_number: int):
        """Update trust scores for each CAV."""
        for cav_id, cav_data in batch_data.items():
            # Calculate trust score based on anomaly detection
            anomaly_score = anomaly_results.get(cav_id, {}).get("anomaly_score", 1.0)
            
            # Update trust score
            trust_score = max(0.0, 1.0 - anomaly_score)
            self.trust_scores[cav_id] = trust_score
            
            # Store in history
            self.history.append({
                "tick": tick_number,
                "cav_id": cav_id,
                "trust_score": trust_score,
                "anomaly_score": anomaly_score
            })
        
        # Keep only recent history
        self.history = [h for h in self.history if h["tick"] > tick_number - self.consistency_window]
    
    def _filter_malicious_data(self, batch_data: Dict, anomaly_results: Dict, tick_number: int) -> Dict:
        """Filter out malicious data based on trust scores."""
        modified_data = {}
        
        for cav_id, cav_data in batch_data.items():
            trust_score = self.trust_scores.get(cav_id, 1.0)
            
            # Check if CAV is trusted
            if trust_score > self.trust_threshold:
                modified_data[cav_id] = cav_data
            else:
                # Block malicious data
                self.blocked_attacks += 1
                logger.warning("Blocked data from CAV %s (trust score: %.2f)", cav_id, trust_score)
                
                # Replace with trusted data or empty
                if self._has_trusted_alternative(cav_id, batch_data):
                    modified_data[cav_id] = self._get_trusted_alternative(cav_id, batch_data)
                else:
                    # Create empty data structure
                    modified_data[cav_id] = self._create_empty_data()
        
        return modified_data
    
    def _has_trusted_alternative(self, cav_id: int, batch_data: Dict) -> bool:
        """Check if there's a trusted alternative for the CAV."""
        for other_id, other_data in batch_data.items():
            if other_id != cav_id and self.trust_scores.get(other_id, 0.0) > self.trust_threshold:
                return True
        return False
    
    def _get_trusted_alternative(self, cav_id: int, batch_data: Dict) -> Dict:
        """Get trusted alternative data for the CAV."""
        for other_id, other_data in batch_data.items():
            if other_id != cav_id and self.trust_scores.get(other_id, 0.0) > self.trust_threshold:
                return other_data.copy()
        return self._create_empty_data()
    
    def _create_empty_data(self) -> Dict:
        """Create empty data structure."""
        return {
            "lidar": np.zeros((0, 4)),  # Empty LiDAR data
            "pred_bboxes": np.zeros((0, 7)),  # Empty predictions
            "pred_scores": np.zeros(0),  # Empty scores
            "gt_bboxes": np.zeros((0, 7)),  # Empty ground truth
        }
    
    def get_trust_scores(self) -> Dict:
        """Get current trust scores for all CAVs."""
        return self.trust_scores.copy()
    
    def get_blocked_attacks(self) -> int:
        """Get number of blocked attacks."""
        return self.blocked_attacks
    
    def _log_defense_results(self, tick_number: int, consistency_results: Dict, anomaly_results: Dict):
        """Log defense results."""
        logger.info("Defense results for tick %d:", tick_number)
        logger.info("  Consistency: %s", consistency_results)
        logger.info("  Anomalies: %s", anomaly_results)
        logger.info("  Trust scores: %s", self.trust_scores)
    
    def cleanup(self):
        """Cleanup defense handler resources."""
        logger.info("Cleaning up defense handler")
        
        # Cleanup components
        self.occupancy_map.cleanup()
        self.consistency_checker.cleanup()
        self.anomaly_detector.cleanup()
        
        logger.info("Defense handler cleanup completed")


class OccupancyMap:
    def __init__(self, config: Dict):
        """Initialize occupancy map."""
        self.resolution = config.get("defense", {}).get("occupancy_resolution", 0.5)
        self.map_size = config.get("defense", {}).get("occupancy_map_size", 100)
        self.occupancy_grid = np.zeros((self.map_size, self.map_size))
        
    def update(self, batch_data: Dict, tick_number: int):
        """Update occupancy map with new data."""
        # Clear previous data
        self.occupancy_grid.fill(0)
        
        # Update with new data
        for cav_id, cav_data in batch_data.items():
            if "lidar" in cav_data:
                lidar_points = cav_data["lidar"][:, :2]  # Use x, y coordinates
                
                # Convert to grid coordinates
                grid_coords = self._world_to_grid(lidar_points)
                
                # Update occupancy
                for coord in grid_coords:
                    if 0 <= coord[0] < self.map_size and 0 <= coord[1] < self.map_size:
                        self.occupancy_grid[coord[0], coord[1]] = 1
    
    def _world_to_grid(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to grid coordinates."""
        grid_coords = np.floor(points / self.resolution).astype(int)
        return grid_coords
    
    def cleanup(self):
        """Cleanup occupancy map."""
        pass


class ConsistencyChecker:
    def __init__(self, config: Dict):
        """Initialize consistency checker."""
        self.window_size = config.get("defense", {}).get("consistency_window", 5)
        self.min_consistency = config.get("defense", {}).get("min_consistency", 0.8)
        self.history = []
        
    def check_consistency(self, batch_data: Dict, tick_number: int) -> Dict:
        """Check consistency across CAVs."""
        results = {}
        
        # Compare each CAV with others
        cav_ids = list(batch_data.keys())
        for i, cav_id in enumerate(cav_ids):
            cav_data = batch_data[cav_id]
            
            # Compare with other CAVs
            consistency_scores = []
            for j, other_id in enumerate(cav_ids):
                if i != j:
                    other_data = batch_data[other_id]
                    score = self._calculate_consistency(cav_data, other_data)
                    consistency_scores.append(score)
            
            # Calculate average consistency
            if consistency_scores:
                avg_consistency = np.mean(consistency_scores)
            else:
                avg_consistency = 1.0
            
            results[cav_id] = {
                "consistency_score": avg_consistency,
                "consistent": avg_consistency > self.min_consistency
            }
        
        # Store in history
        self.history.append({
            "tick": tick_number,
            "results": results.copy()
        })
        
        # Keep only recent history
        self.history = [h for h in self.history if h["tick"] > tick_number - self.window_size]
        
        return results
    
    def _calculate_consistency(self, data1: Dict, data2: Dict) -> float:
        """Calculate consistency between two CAVs."""
        # Compare LiDAR point clouds
        if "lidar" in data1 and "lidar" in data2:
            pc1 = data1["lidar"][:, :2]
            pc2 = data2["lidar"][:, :2]
            
            # Calculate IoU of point clouds
            iou = self._calculate_point_cloud_iou(pc1, pc2)
            return iou
        
        return 0.0
    
    def _calculate_point_cloud_iou(self, pc1: np.ndarray, pc2: np.ndarray) -> float:
        """Calculate IoU of two point clouds."""
        # Create occupancy grids for both point clouds
        grid_size = 50
        resolution = 1.0
        
        grid1 = np.zeros((grid_size, grid_size))
        grid2 = np.zeros((grid_size, grid_size))
        
        # Fill grids
        for point in pc1:
            x, y = int(point[0] / resolution + grid_size/2), int(point[1] / resolution + grid_size/2)
            if 0 <= x < grid_size and 0 <= y < grid_size:
                grid1[x, y] = 1
        
        for point in pc2:
            x, y = int(point[0] / resolution + grid_size/2), int(point[1] / resolution + grid_size/2)
            if 0 <= x < grid_size and 0 <= y < grid_size:
                grid2[x, y] = 1
        
        # Calculate IoU
        intersection = np.sum(grid1 * grid2)
        union = np.sum(grid1) + np.sum(grid2) - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def cleanup(self):
        """Cleanup consistency checker."""
        pass


class AnomalyDetector:
    def __init__(self, config: Dict):
        """Initialize anomaly detector."""
        self.threshold = config.get("defense", {}).get("anomaly_threshold", 0.5)
        self.history = []
        
    def detect_anomalies(self, batch_data: Dict, consistency_results: Dict, tick_number: int) -> Dict:
        """Detect anomalies in the data."""
        results = {}
        
        for cav_id, cav_data in batch_data.items():
            # Get consistency score
            consistency = consistency_results.get(cav_id, {}).get("consistency_score", 1.0)
            
            # Calculate anomaly score
            anomaly_score = 1.0 - consistency
            
            # Check if anomaly detected
            is_anomaly = anomaly_score > self.threshold
            
            results[cav_id] = {
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
                "consistency": consistency
            }
        
        # Store in history
        self.history.append({
            "tick": tick_number,
            "results": results.copy()
        })
        
        return results
    
    def cleanup(self):
        """Cleanup anomaly detector."""
        pass