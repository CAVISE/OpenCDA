"""
Shared ML model manager for multi-agent systems.

This module provides a centralized ML model manager that allows multiple CAVs
(Connected Autonomous Vehicles) to share the same model instance, avoiding
duplicate memory consumption in multi-agent cooperative perception scenarios.
"""

from typing import Any
import cv2
import torch
import numpy as np
import numpy.typing as npt


class MLManager(object):
    """
    Centralized ML model manager for multi-agent systems.

    Contains and manages ML/DL models shared across multiple CAVs to avoid
    duplicate memory consumption. Currently supports YOLOv5 object detection.

    Attributes
    ----------
    object_detector : torch.nn.Module
        YOLOv5 object detector loaded from PyTorch Hub.
    """

    def __init__(self):
        self.object_detector = torch.hub.load("ultralytics/yolov5", "yolov5m")

    def draw_2d_box(self, result: Any, rgb_image: npt.NDArray[np.uint8], index: int) -> npt.NDArray[np.uint8]:
        """
        Draw 2D bounding boxes on image based on YOLO detection results.

        Parameters
        ----------
        result : Any
            Detection result from YOLOv5 containing bounding boxes and labels.
        rgb_image : npt.NDArray[np.uint8]
            RGB camera image with shape (H, W, 3).
        index : int
            Index indicating which batch result to visualize.

        Returns
        -------
        npt.NDArray[np.uint8]
            RGB image with bounding boxes and labels drawn.
        """
        # torch.Tensor
        bounding_box = result.xyxy[index]
        if bounding_box.is_cuda:
            bounding_box = bounding_box.cpu().detach().numpy()
        else:
            bounding_box = bounding_box.detach().numpy()

        for i in range(bounding_box.shape[0]):
            detection = bounding_box[i]

            # the label has 80 classes, which is the same as coco dataset
            label = int(detection[5])
            label_name = result.names[label]

            if is_vehicle_cococlass(label):
                label_name = "vehicle"

            x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw text on it
            cv2.putText(rgb_image, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

        return rgb_image


def is_vehicle_cococlass(label: int) -> bool:
    """
    Check if label belongs to vehicle class in COCO dataset.

    Parameters
    ----------
    label : int
        YOLO detection class label (0-79 for COCO).

    Returns
    -------
    bool
        True if label corresponds to vehicle class, False otherwise.
    """
    vehicle_class_array = np.array([1, 2, 3, 5, 7], dtype=np.int)
    return True if 0 in (label - vehicle_class_array) else False
