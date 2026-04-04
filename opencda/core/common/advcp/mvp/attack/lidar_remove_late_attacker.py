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
        import logging
        logger = logging.getLogger("cavise.lidar_remove_late_attacker")
        case = copy.deepcopy(multi_frame_case)
        attack_results: List[Dict[str, Any]] = []
        current_frame = max(multi_frame_case.keys())
        logger.info(f"[DEBUG run] Starting attack, current_frame={current_frame}, attack_opts keys={list(attack_opts.keys())}")
        for frame_id in sorted(multi_frame_case.keys()):
            attack_results.append({})
            if frame_id == current_frame:
                multi_vehicle_case = multi_frame_case[frame_id]
                attacker_id = attack_opts["attacker_vehicle_id"]
                ego_id = attack_opts["victim_vehicle_id"]
                logger.info(f"[DEBUG run] Processing frame {frame_id}, attacker={attacker_id}, victim={ego_id}")
                
                # Get attacker's lidar_pose to transform world bbox to attacker's coordinates
                attacker_pose = multi_vehicle_case[attacker_id]["lidar_pose"]
                
                object_index = multi_vehicle_case[attacker_id]["object_ids"].index(attack_opts["object_id"])
                bbox_to_remove = multi_vehicle_case[attacker_id]["gt_bboxes"][object_index]
                bbox_to_remove_attacker = bbox_map_to_sensor(bbox_to_remove, attacker_pose)
                
                # DEBUG LOG: Check bbox before attack
                logger.info(f"[DEBUG] Frame {frame_id}: Attacker={attacker_id}, Victim={ego_id}")
                logger.info(f"[DEBUG] bbox_to_remove (world): {bbox_to_remove}")
                logger.info(f"[DEBUG] attacker_pose: {attacker_pose}")
                logger.info(f"[DEBUG] bbox_to_remove_attacker (attacker coords): {bbox_to_remove_attacker}")
                
                logger.info(f"[DEBUG run] About to call perception.attack_late with ego_id={ego_id}, attacker_id={attacker_id}")
                result = self.perception.attack_late(multi_vehicle_case, ego_id, attacker_id, mode="remove", bbox=bbox_to_remove_attacker)
                logger.info(f"[DEBUG run] attack_late returned, result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                
                # DEBUG LOG: Check result
                orig_pred_count = len(multi_vehicle_case[ego_id].get("pred_bboxes", []))
                attacked_pred_count = len(result["pred_bboxes"])
                logger.info(f"[DEBUG] Frame {frame_id}: Original pred_bboxes count: {orig_pred_count}, After attack: {attacked_pred_count}")
                
                case[frame_id][ego_id]["pred_bboxes"] = result["pred_bboxes"]
                case[frame_id][ego_id]["pred_scores"] = result["pred_scores"]
                attack_results[-1][ego_id] = {"pred_bboxes": result["pred_bboxes"], "pred_scores": result["pred_scores"]}
        return case, attack_results

    def build_benchmark_meta(self, write: bool = False, max_cnt: int = 500) -> None:
        raise NotImplementedError("Should use the same benchmask as LidarRemoveAttacker.")
