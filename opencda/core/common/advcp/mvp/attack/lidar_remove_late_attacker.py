from typing import Any, Dict, List, Optional, Tuple

import copy

from .attacker import Attacker
from mvp.data.util import bbox_map_to_sensor


class LidarRemoveLateAttacker(Attacker):
    def __init__(self, perception: Any, dataset: Optional[Any] = None) -> None:
        super().__init__()
        self.name = "lidar_remove"
        self.dataset = dataset
        self.load_benchmark_meta()
        self.perception = perception
        self.name = "lidar_remove_late"

    def run(self, multi_frame_case: Dict[int, Any], attack_opts: Dict[str, Any]) -> Tuple[Dict[int, Any], List[Dict[str, Any]]]:
        case = copy.deepcopy(multi_frame_case)
        attack_results: List[Dict[str, Any]] = []
        current_frame = max(multi_frame_case.keys())
        for frame_id in sorted(multi_frame_case.keys()):
            attack_results.append({})
            if frame_id == current_frame:
                multi_vehicle_case = multi_frame_case[frame_id]
                attacker_id = attack_opts["attacker_vehicle_id"]
                ego_id = attack_opts["victim_vehicle_id"]
                # Get ego's lidar_pose to transform world bbox to ego coordinates
                ego_pose = multi_vehicle_case[ego_id]["lidar_pose"]
                
                # Get bbox in world coordinates from attack_opts
                #if "bboxes" in attack_opts:
                #    bbox_world = attack_opts["bboxes"][0]  # single bbox
                #    # Transform world bbox to ego sensor coordinates
                #    bbox_ego = bbox_map_to_sensor(bbox_world, ego_pose)
                #else:
                #    bbox_ego = None
                object_index = multi_vehicle_case[attacker_id]["object_ids"].index(attack_opts["object_id"])
                bbox_to_remove = multi_vehicle_case[attacker_id]["gt_bboxes"][object_index]
                bbox_to_remove_ego = bbox_map_to_sensor(bbox_to_remove, ego_pose)
                result = self.perception.attack_late(multi_vehicle_case, ego_id, attacker_id, mode="remove", bbox=bbox_to_remove_ego)
                case[frame_id][ego_id]["pred_bboxes"] = result["pred_bboxes"]
                case[frame_id][ego_id]["pred_scores"] = result["pred_scores"]
                attack_results[-1][ego_id] = {"pred_bboxes": result["pred_bboxes"], "pred_scores": result["pred_scores"]}
        return case, attack_results

    def build_benchmark_meta(self, write: bool = False, max_cnt: int = 500) -> None:
        raise NotImplementedError("Should use the same benchmask as LidarRemoveAttacker.")
