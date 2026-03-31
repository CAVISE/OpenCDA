import logging
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

from .general import get_xylims
from mvp.data.util import pcd_sensor_to_map, bbox_sensor_to_map
from .general import draw_bbox_2d

def _prepare_bboxes(bboxes: Any) -> np.ndarray:
    """Ensure bboxes array is 2D (N, 7) by removing batch dimension and converting corners if needed."""
    bboxes = _to_numpy(bboxes)
    
    # If it's already 2D with 7 columns, just ensure proper shape
    if bboxes.ndim == 2 and bboxes.shape[1] == 7:
        return bboxes
    
    # If it's 3D with shape (N, 8, 3), it's corners format - convert to bbox
    if bboxes.ndim == 3 and bboxes.shape[1] == 8 and bboxes.shape[2] == 3:
        from mvp.data.util import corners_to_bbox_batch
        return corners_to_bbox_batch(bboxes)
    
    # Handle batch dimension: (1, N, 7) or (1, N, 8, 3)
    if bboxes.ndim == 4 and bboxes.shape[0] == 1:
        bboxes = bboxes[0]
        return _prepare_bboxes(bboxes)  # Recursive call with squeezed array
    
    # If it's 3D with batch but not corners, try to squeeze
    if bboxes.ndim == 3:
        if bboxes.shape[0] == 1:
            bboxes = bboxes[0]
            return _prepare_bboxes(bboxes)
        else:
            # Multiple batches - flatten to 2D
            logger.warning(f"Flattening batch of size {bboxes.shape[0]}")
            bboxes = bboxes.reshape(-1, bboxes.shape[-1])
            return _prepare_bboxes(bboxes)
    
    # If it's 1D single bbox with 7 elements
    if bboxes.ndim == 1 and len(bboxes) == 7:
        return bboxes[None, :]
    
    raise ValueError(f"Cannot prepare bboxes with shape {bboxes.shape}")

logger = logging.getLogger(__name__)


def _to_numpy(data: Any) -> Any:
    """Convert PyTorch tensors to numpy arrays, handling CPU/GPU devices."""
    if isinstance(data, torch.Tensor):
        if data.device.type == 'cuda':
            data = data.cpu()
        return data.detach().numpy()
    return data


def _normalize_bbox(bbox: Any) -> np.ndarray:
    """Normalize bbox to 1D array with 7 elements (x, y, z, w, l, h, yaw).
    
    Args:
        bbox: Bounding box in various formats (list, numpy array, torch tensor)
        
    Returns:
        1D numpy array with shape (7,)
        
    Raises:
        ValueError: If bbox cannot be normalized to 7 elements
    """
    bbox = _to_numpy(bbox)
    bbox = np.asarray(bbox)
    
    # Flatten to 1D if it's a higher dimensional array
    while bbox.ndim > 1:
        bbox = bbox[0]
    
    # Ensure it's a 1D array
    bbox = np.atleast_1d(bbox)
    
    if len(bbox) != 7:
        raise ValueError(f"Bounding box must have 7 elements (x, y, z, w, l, h, yaw), got shape {bbox.shape}")
    
    return bbox


def draw_attack(
    attack: Dict[str, Any],
    normal_case: Dict[int, Dict[Any, Any]],
    attack_case: Dict[int, Dict[Any, Any]],
    mode: str = "multi_frame",
    show: bool = False,
    save: Optional[str] = None,
) -> None:
    if mode == "multi_frame":
        frame_ids = attack["attack_meta"]["attack_frame_ids"]
        frame_num = len(frame_ids)
        fig, axes = plt.subplots(frame_num, 2, figsize=(40, 20 * frame_num))

        # draw normal case first
        for case_id, case in enumerate([normal_case, attack_case]):
            for frame_id in frame_ids:
                if frame_num <= 1:
                    ax = axes[case_id]
                else:
                    ax = axes[frame_ids.index(frame_id)][case_id]

                # draw point clouds
                # pointcloud_all = pcd_sensor_to_map(case[frame_id][attack["attack_opts"]["attacker_vehicle_id"]]["lidar"], case[frame_id][attack["attack_opts"]["attacker_vehicle_id"]]["lidar_pose"])[:,:3]
                pointcloud_all = np.vstack(
                    [
                        pcd_sensor_to_map(
                            _to_numpy(vehicle_data["lidar"]),
                            _to_numpy(vehicle_data["lidar_pose"])
                        )[:, :3]
                        for vehicle_id, vehicle_data in case[frame_id].items()
                    ]
                )
                xlim, ylim = get_xylims(pointcloud_all)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                # ax.set_aspect('equal', adjustable='box')
                ax.scatter(pointcloud_all[:, 0], pointcloud_all[:, 1], s=0.01, c="black")

                # label the location of attacker and victim
                attacker_vehicle_id = attack["attack_meta"]["attacker_vehicle_id"]
                attacker_vehicle_data = case[frame_id][attacker_vehicle_id]
                victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
                victim_vehicle_data = case[frame_id][victim_vehicle_id]
                # Convert to numpy if needed and get 2D position
                victim_pos = _to_numpy(victim_vehicle_data["lidar_pose"])[:2]
                attacker_pos = _to_numpy(attacker_vehicle_data["lidar_pose"])[:2]
                # Convert to list if numpy array
                if hasattr(victim_pos, 'tolist'):
                    victim_pos = victim_pos.tolist()
                if hasattr(attacker_pos, 'tolist'):
                    attacker_pos = attacker_pos.tolist()
                ax.scatter(*victim_pos, s=100, c="green")
                ax.scatter(*attacker_pos, s=100, c="red")

                # draw gt/result bboxes
                total_bboxes: List[tuple] = []
                if "gt_bboxes" in victim_vehicle_data:
                    gt_bboxes = _prepare_bboxes(victim_vehicle_data["gt_bboxes"])
                    total_bboxes.append(
                        (
                            bbox_sensor_to_map(
                                gt_bboxes,
                                _to_numpy(victim_vehicle_data["lidar_pose"])
                            ),
                            victim_vehicle_data["object_ids"],
                            "g",
                        )
                    )
                if "pred_bboxes" in victim_vehicle_data:
                    pred_bboxes = _prepare_bboxes(victim_vehicle_data["pred_bboxes"])
                    total_bboxes.append((
                        bbox_sensor_to_map(
                            pred_bboxes,
                            _to_numpy(victim_vehicle_data["lidar_pose"])
                        ),
                        None,
                        "r"
                    ))
                # label the position of spoofing/removal
                if attack["attack_meta"].get("bboxes") is not None and len(attack["attack_meta"]["bboxes"]) > 0:
                    frame_idx = frame_ids.index(frame_id)
                    if frame_idx < len(attack["attack_meta"]["bboxes"]):
                        bbox = attack["attack_meta"]["bboxes"][frame_idx]
                        # Normalize bbox to 1D array with 7 elements
                        bbox_1d = _normalize_bbox(bbox)
                        # Convert to 2D array (1, 7) for bbox_sensor_to_map
                        bbox_2d = bbox_1d[None, :]
                        bbox = bbox_sensor_to_map(bbox_2d, _to_numpy(attacker_vehicle_data["lidar_pose"]))
                        total_bboxes.append((bbox, None, "red"))
                    else:
                        logger.debug(f"Frame index {frame_idx} out of range for bboxes (len={len(attack['attack_meta']['bboxes'])}), skipping")
                else:
                    logger.debug(f"No bbox data in attack_meta, skipping attack bbox visualization")

                draw_bbox_2d(ax, total_bboxes)
    else:
        raise NotImplementedError()

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close(fig)


def draw_attack_trace(trace: Any, show: bool = False, save: Optional[str] = None) -> None:
    pass
