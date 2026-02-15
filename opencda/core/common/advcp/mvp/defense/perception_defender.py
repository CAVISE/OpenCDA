import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
import pickle

from .defender import Defender
from mvp.data.util import bbox_sensor_to_map
from mvp.tools.polygon_space import bbox_to_polygon
from mvp.config import data_root


class PerceptionDefender(Defender):
    thres = 1.7
    sigma = 0

    def __init__(self) -> None:
        super().__init__()
        self.name = "perception"
        self.lane_areas_map: Optional[Dict[str, Any]] = None
        self._load_map()

    def score(self, metrics: Any) -> int:
        return 0

    def run_core(
        self,
        pred_bboxes: np.ndarray,
        gt_bboxes: np.ndarray,
        occupied_areas: List[Any],
        free_areas: Any,
        ego_area: Any,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"spoof": [], "remove": []}
        pred_bbox_areas = []
        gt_bbox_areas = []
        if isinstance(occupied_areas, MultiPolygon):
            occupied_areas = list(occupied_areas.geoms)

        for bbox in pred_bboxes:
            bbox_area = bbox_to_polygon(bbox)
            pred_bbox_areas.append(bbox_area)

        for bbox in gt_bboxes:
            bbox_area = bbox_to_polygon(bbox)
            gt_bbox_areas.append(bbox_area)

        pred_merged_areas = unary_union(pred_bbox_areas)
        gt_merged_areas = unary_union(gt_bbox_areas)

        for i, occupied_area in enumerate(occupied_areas):
            error_area = occupied_area.difference(pred_merged_areas)
            occupied_area_error = error_area.area
            occupied_area_gt_error = 0
            gt_bbox_index = -1
            for j, gt_a in enumerate(gt_bbox_areas):
                if gt_a.intersection(occupied_area).area > 0:
                    occupied_area_gt_error = gt_a.difference(pred_merged_areas).area
                    gt_bbox_index = j
                    break
            metrics["remove"].append((error_area, occupied_area_error, occupied_area_gt_error, gt_bbox_index))

        for i, bbox_area in enumerate(pred_bbox_areas):
            error_area = bbox_area.intersection(free_areas)
            free_area_error = error_area.area
            free_area_gt_error = bbox_area.difference(gt_merged_areas).area
            metrics["spoof"].append((error_area, free_area_error, free_area_gt_error, i))

        metrics["gt_bboxes"] = gt_bboxes
        metrics["pred_bboxes"] = pred_bboxes

        return metrics

    def run(
        self, multi_frame_case: Dict[int, Dict[int, Any]], defend_opts: Dict[str, Any]
    ) -> Tuple[Dict[int, Dict[int, Any]], int, List[Dict[str, Any]]]:
        metrics: List[Dict[str, Any]] = [{} for _ in range(10)]
        try:
            map_name = multi_frame_case[0][list(multi_frame_case[0].keys())[0]]["map"]
            lane_areas = self.lane_areas_map[map_name] if self.lane_areas_map else None
        except (KeyError, IndexError):
            lane_areas = None
        vehicle_ids = list(multi_frame_case[0].keys()) if "vehicle_ids" not in defend_opts else defend_opts["vehicle_ids"]

        for frame_id in defend_opts["frame_ids"]:
            frame_data = multi_frame_case[frame_id]

            # Merge occupancy maps.
            occupied_areas: List[Any] = []
            free_areas: List[Any] = []

            for vehicle_id, vehicle_data in frame_data.items():
                if vehicle_id not in vehicle_ids:
                    continue
                occupied_areas += vehicle_data["occupied_areas"]
                occupied_areas.append(vehicle_data["ego_area"])
                free_areas.append(unary_union(vehicle_data["free_areas"]).difference(vehicle_data["ego_area"]))
            free_areas_merged = unary_union(free_areas)

            # Do consistency check.
            gt_bboxes_list: List[np.ndarray] = []
            all_object_ids: List[List[int]] = []
            for vehicle_id, vehicle_data in frame_data.items():
                gt_bboxes_list.append(bbox_sensor_to_map(vehicle_data["gt_bboxes"], vehicle_data["lidar_pose"]))
                all_object_ids.append(vehicle_data["object_ids"])
            gt_bboxes = np.vstack(gt_bboxes_list)
            all_object_ids_flat = np.hstack(all_object_ids).reshape(-1)
            _, unique_indices = np.unique(all_object_ids_flat, return_index=True)
            gt_bboxes = gt_bboxes[unique_indices]

            for vehicle_id, vehicle_data in frame_data.items():
                if vehicle_id not in vehicle_ids:
                    continue
                if "pred_bboxes" not in vehicle_data:
                    continue
                filtered_occupied_areas = []
                for area in occupied_areas:
                    if lane_areas is None or self.check_in_lane_areas(area, lane_areas):
                        filtered_occupied_areas.append(area)
                filtered_occupied_areas_merged = unary_union(filtered_occupied_areas)
                pred_bboxes = vehicle_data["pred_bboxes"]
                pred_bboxes = bbox_sensor_to_map(pred_bboxes, vehicle_data["lidar_pose"])
                filtered_pred_bbox_indices = []
                for i, pred_bbox in enumerate(pred_bboxes):
                    pred_area = bbox_to_polygon(pred_bbox)
                    if lane_areas is None or self.check_in_lane_areas(pred_area, lane_areas):
                        filtered_pred_bbox_indices.append(i)
                pred_bboxes = pred_bboxes[filtered_pred_bbox_indices]
                vehicle_metrics = self.run_core(pred_bboxes, gt_bboxes, filtered_occupied_areas_merged, free_areas_merged, vehicle_data["ego_area"])
                vehicle_metrics["lidar_pose"] = vehicle_data["lidar_pose"]
                metrics[frame_id][vehicle_id] = vehicle_metrics

        score = self.score(metrics)
        return multi_frame_case, score, metrics

    @staticmethod
    def check_in_lane_areas(area: Any, lane_areas: List[Any]) -> bool:
        intersection = 0
        for lane_area in lane_areas:
            intersection += area.intersection(lane_area).area
        return intersection > 0.95 * area.area

    @staticmethod
    def check_in_perception_range(area: Any, lidar_pose: np.ndarray, lidar_range: List[float]) -> bool:
        perception_range = np.array([*lidar_pose[:3], lidar_range[3] - lidar_range[0], lidar_range[4] - lidar_range[1], 1, np.radians(lidar_pose[4])])
        perception_range = bbox_to_polygon(perception_range)

        return area.intersection(perception_range).area > 0.95 * area.area

    def _load_map(self, map_names: Optional[List[str]] = None) -> None:
        self.lane_areas_map = {}
        if map_names is None:
            map_names = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]

        for map_name in map_names:
            with open(os.path.join(data_root, "carla/{}_lane_areas.pkl".format(map_name)), "rb") as f:
                self.lane_areas_map[map_name] = pickle.load(f)
