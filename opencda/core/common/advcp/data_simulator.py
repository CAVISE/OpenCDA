"""
DataSimulator - Adapts CARLA data for AdvCP processing.
Handles data format conversion and preprocessing.
"""

import logging
import numpy as np
from typing import Dict, List, Optional

import open3d as o3d

logger = logging.getLogger("cavise.advcp.data_simulator")


class DataSimulator:
    def __init__(self):
        """Initialize data simulator."""
        self.config = None
        self.dataset = None
        
        logger.info("DataSimulator initialized")
    
    def prepare_dataset(self, dataset: object):
        """
        Prepare dataset for AdvCP processing.
        
        Args:
            dataset: OpenCOOD dataset object
        """
        self.dataset = dataset
        
        # Add AdvCP specific data processing
        self._add_advcp_metadata()
        self._preprocess_point_clouds()
        self._generate_occupancy_maps()
        
        logger.info("Dataset prepared for AdvCP processing")
    
    def _add_advcp_metadata(self):
        """Add AdvCP specific metadata to dataset."""
        for sample_id in range(len(self.dataset)):
            sample = self.dataset[sample_id]
            
            # Add attack/defense metadata
            sample["advcp_metadata"] = {
                "timestamp": sample.get("timestamp", 0),
                "frame_id": sample.get("frame_id", 0),
                "scenario_id": sample.get("scenario_id", "unknown"),
                "attack_enabled": False,
                "defense_enabled": False,
                "trust_scores": {}
            }
            
            # Add CAV specific metadata
            for cav_id in sample["cav_data"]:
                cav_data = sample["cav_data"][cav_id]
                
                cav_data["advcp_cav_metadata"] = {
                    "cav_id": cav_id,
                    "position": cav_data.get("position", [0, 0, 0]),
                    "orientation": cav_data.get("orientation", [0, 0, 0]),
                    "velocity": cav_data.get("velocity", [0, 0, 0]),
                    "is_attacker": False,
                    "trust_score": 1.0,
                    "detection_results": []
                }
    
    def _preprocess_point_clouds(self):
        """Preprocess point clouds for AdvCP."""
        for sample_id in range(len(self.dataset)):
            sample = self.dataset[sample_id]
            
            for cav_id, cav_data in sample["cav_data"].items():
                if "lidar" in cav_data:
                    # Normalize point cloud
                    lidar_data = cav_data["lidar"]
                    normalized_lidar = self._normalize_point_cloud(lidar_data)
                    cav_data["lidar_normalized"] = normalized_lidar
                    
                    # Generate features
                    features = self._extract_features(normalized_lidar)
                    cav_data["lidar_features"] = features
    
    def _normalize_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Normalize point cloud data."""
        if point_cloud.size == 0:
            return point_cloud
        
        # Normalize coordinates to [-1, 1]
        points = point_cloud[:, :3]
        
        # Calculate bounds
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        
        # Normalize
        normalized = 2 * (points - min_vals) / (max_vals - min_vals) - 1
        
        # Keep intensity
        if point_cloud.shape[1] > 3:
            normalized = np.hstack([normalized, point_cloud[:, 3:]])
        
        return normalized
    
    def _extract_features(self, point_cloud: np.ndarray) -> Dict:
        """Extract features from point cloud."""
        features = {}
        
        if point_cloud.size == 0:
            return features
        
        # Basic statistics
        points = point_cloud[:, :3]
        
        features["num_points"] = len(points)
        features["centroid"] = np.mean(points, axis=0)
        features["std_dev"] = np.std(points, axis=0)
        features["density"] = len(points) / np.prod(np.ptp(points, axis=0))
        
        # Range features
        features["min_range"] = np.min(np.linalg.norm(points, axis=1))
        features["max_range"] = np.max(np.linalg.norm(points, axis=1))
        features["avg_range"] = np.mean(np.linalg.norm(points, axis=1))
        
        # Angular features (if available)
        if point_cloud.shape[1] > 3:
            intensities = point_cloud[:, 3]
            features["avg_intensity"] = np.mean(intensities)
            features["std_intensity"] = np.std(intensities)
        
        return features
    
    def _generate_occupancy_maps(self):
        """Generate occupancy maps for each CAV."""
        for sample_id in range(len(self.dataset)):
            sample = self.dataset[sample_id]
            
            for cav_id, cav_data in sample["cav_data"].items():
                if "lidar" in cav_data:
                    lidar_data = cav_data["lidar"]
                    occupancy_map = self._create_occupancy_map(lidar_data)
                    cav_data["occupancy_map"] = occupancy_map
    
    def _create_occupancy_map(self, point_cloud: np.ndarray) -> np.ndarray:
        """Create 2D occupancy map from point cloud."""
        if point_cloud.size == 0:
            return np.zeros((100, 100))
        
        # Create grid
        grid_size = 100
        resolution = 0.5  # 0.5m per cell
        
        occupancy_map = np.zeros((grid_size, grid_size))
        
        # Project points to 2D (x, y)
        points_2d = point_cloud[:, :2]
        
        # Convert to grid coordinates
        grid_coords = np.floor(points_2d / resolution).astype(int) + grid_size // 2
        
        # Fill occupancy map
        for coord in grid_coords:
            if 0 <= coord[0] < grid_size and 0 <= coord[1] < grid_size:
                occupancy_map[coord[0], coord[1]] = 1
        
        return occupancy_map
    
    def adapt_carla_data(self, carla_data: Dict) -> Dict:
        """
        Adapt CARLA data for AdvCP processing.
        
        Args:
            carla_data: Raw CARLA data dictionary
            
        Returns:
            Adapted data for AdvCP
        """
        adapted_data = {}
        
        # Process each CAV
        for cav_id, cav_info in carla_data.items():
            adapted_cav_data = {}
            
            # Process LiDAR data
            if "lidar" in cav_info:
                lidar_data = cav_info["lidar"]
                adapted_lidar = self._adapt_lidar_data(lidar_data)
                adapted_cav_data["lidar"] = adapted_lidar
            
            # Process camera data
            if "camera" in cav_info:
                camera_data = cav_info["camera"]
                adapted_camera = self._adapt_camera_data(camera_data)
                adapted_cav_data["camera"] = adapted_camera
            
            # Add CAV metadata
            adapted_cav_data["metadata"] = {
                "cav_id": cav_id,
                "position": cav_info.get("position", [0, 0, 0]),
                "orientation": cav_info.get("orientation", [0, 0, 0]),
                "velocity": cav_info.get("velocity", [0, 0, 0])
            }
            
            adapted_data[cav_id] = adapted_cav_data
        
        return adapted_data
    
    def _adapt_lidar_data(self, lidar_data: np.ndarray) -> np.ndarray:
        """Adapt raw LiDAR data."""
        if lidar_data.size == 0:
            return lidar_data
        
        # Convert to AdvCP format (x, y, z, intensity)
        if lidar_data.shape[1] < 4:
            # Add intensity if missing
            intensities = np.ones((lidar_data.shape[0], 1)) * 0.1
            lidar_data = np.hstack([lidar_data, intensities])
        
        return lidar_data
    
    def _adapt_camera_data(self, camera_data: np.ndarray) -> np.ndarray:
        """Adapt camera data."""
        # For now, just return as-is
        return camera_data
    
    def cleanup(self):
        """Cleanup data simulator resources."""
        logger.info("Cleaning up DataSimulator")
        self.dataset = None
        logger.info("DataSimulator cleanup completed")