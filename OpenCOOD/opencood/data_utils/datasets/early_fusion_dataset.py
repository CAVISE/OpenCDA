"""
Dataset class for early fusion.

This module implements the early fusion dataset where each CAV transmits raw
point cloud data to the ego vehicle for processing.
"""

import math
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray
import torch

import opencood.data_utils.datasets
from opencood.utils import box_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, shuffle_points, downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2

logger = logging.getLogger("cavise.OpenCOOD.opencood.data_utils.datasets.early_fusion_dataset")


class EarlyFusionDataset(basedataset.BaseDataset):
    """
    Dataset class for early fusion cooperative perception.

    Parameters
    ----------
    params : Dict[str, Any]
        Configuration dictionary containing dataset parameters.
    visualize : bool
        Whether to include visualization data.
    train : bool
        Whether the dataset is used for training. Default is True.
    message_handler : Optional[Any]
        Handler for inter-vehicle communication. Default is None.

    Attributes
    ----------
    pre_processor : Any
        Preprocessor for point cloud data.
    post_processor : Any
        Postprocessor for generating labels and bounding boxes.
    message_handler : Optional[Any]
        Handler for inter-vehicle communication.
    module_name : str
        Identifier for the module.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        visualize: bool,
        train: bool = True,
        message_handler: Optional[Any] = None,
    ):
        super().__init__(params, visualize, train)

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)

        self.message_handler = message_handler
        self.module_name = "OpenCOOD.EarlyFusionDataset"

    def __find_ego_vehicle(
        self, base_data_dict: Dict[str, Dict[str, Any]]
    ) -> Tuple[int, List[float]]:
        """
        Find the ego vehicle in the base data dictionary.

        Parameters
        ----------
        base_data_dict : Dict[str, Dict[str, Any]]
            Dictionary containing data for all CAVs.

        Returns
        -------
        Tuple[int, List[float]]
            Tuple containing ego vehicle ID and its LiDAR pose.
        """
        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                ego_id = cav_id
                ego_lidar_pose = cav_content["params"]["lidar_pose"]
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        return ego_id, ego_lidar_pose

    @staticmethod
    def __wrap_ndarray(ndarray: NDArray[np.float64]) -> Dict[str, Any]:
        return {
            "data": ndarray.tobytes(),
            "shape": ndarray.shape,
            "dtype": str(ndarray.dtype),
        }

    def extract_data(self, idx: int) -> None:
        """
        Extract and prepare data for a given index.

        Parameters
        ----------
        idx : int
            Index of the data sample.
        """
        base_data_dict = self.retrieve_base_data(idx)
        _, ego_lidar_pose = self.__find_ego_vehicle(base_data_dict)

        if self.message_handler is not None:
            for cav_id, selected_cav_base in base_data_dict.items():
                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)

                with self.message_handler.handle_opencda_message(cav_id, self.module_name) as msg:
                    msg["object_ids"] = {
                        "data": selected_cav_processed["object_ids"],  # list
                        "label": "LABEL_REPEATED",
                        "name": "object_ids",
                        "type": "int64",
                    }

                    msg["object_bbx_center"] = {
                        "name": "object_bbx_center",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["object_bbx_center"]),
                    }

                    msg["projected_lidar"] = {
                        "name": "projected_lidar",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["projected_lidar"]),
                    }

    def __process_with_messages(
        self,
        ego_id: int,
        ego_lidar_pose: List[float],
        base_data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process data with message handling for inter-vehicle communication.

        Parameters
        ----------
        ego_id : int
            ID of the ego vehicle.
        ego_lidar_pose : List[float]
            Lidar pose of the ego vehicle.
        base_data_dict : Dict[str, Any]
            Dictionary containing base data for all CAVs.

        Returns
        -------
        Dict[str, Any]
            Processed data containing object stack, object IDs, and projected lidar.
        """
        object_stack = []
        object_id_stack = []
        projected_lidar_stack = []

        ego_cav_base = base_data_dict.get(ego_id)
        ego_cav_processed = self.get_item_single_car(ego_cav_base, ego_lidar_pose)

        object_id_stack += ego_cav_processed["object_ids"]
        object_stack.append(ego_cav_processed["object_bbx_center"])
        projected_lidar_stack.append(ego_cav_processed["projected_lidar"])

        if ego_id in self.message_handler.current_message_artery:
            for cav_id, _ in base_data_dict.items():
                if cav_id in self.message_handler.current_message_artery[ego_id]:
                    with self.message_handler.handle_artery_message(ego_id, cav_id, self.module_name) as msg:
                        object_id_stack += msg["object_ids"]

                        bbx = np.frombuffer(msg["object_bbx_center"]["data"], np.dtype(msg["object_bbx_center"]["dtype"]))
                        bbx = bbx.reshape(msg["object_bbx_center"]["shape"])
                        object_stack.append(bbx)

                        projected = np.frombuffer(msg["projected_lidar"]["data"], np.dtype(msg["projected_lidar"]["dtype"]))
                        projected = projected.reshape(msg["projected_lidar"]["shape"])
                        projected_lidar_stack.append(projected)

        return {"object_stack": object_stack, "object_id_stack": object_id_stack, "projected_lidar_stack": projected_lidar_stack}

    def __process_without_messages(
        self,
        ego_lidar_pose: List[float],
        base_data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process data without message handling.

        Parameters
        ----------
        ego_lidar_pose : List[float]
            Lidar pose of the ego vehicle.
        base_data_dict : Dict[str, Any]
            Dictionary containing base data for all CAVs.

        Returns
        -------
        Dict[str, Any]
            Processed data containing object stack, object IDs, and projected lidar.
        """
        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []

        for _, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            dx = selected_cav_base["params"]["lidar_pose"][0] - ego_lidar_pose[0]
            dy = selected_cav_base["params"]["lidar_pose"][1] - ego_lidar_pose[1]
            distance = math.hypot(dx, dy)

            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)
            projected_lidar_stack.append(selected_cav_processed["projected_lidar"])
            object_stack.append(selected_cav_processed["object_bbx_center"])
            object_id_stack += selected_cav_processed["object_ids"]

        return {"object_stack": object_stack, "object_id_stack": object_id_stack, "projected_lidar_stack": projected_lidar_stack}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample by index.

        Parameters
        ----------
        idx : int
            Index of the data sample.

        Returns
        -------
        Dict[str, Any]
            Processed data dictionary.
        """
        base_data_dict = self.retrieve_base_data(idx)
        processed_data_dict = OrderedDict()
        processed_data_dict["ego"] = {}

        ego_id, ego_lidar_pose = self.__find_ego_vehicle(base_data_dict)

        if self.message_handler is not None:
            data = self.__process_with_messages(ego_id, ego_lidar_pose, base_data_dict)
        else:
            data = self.__process_without_messages(ego_lidar_pose, base_data_dict)

        # exclude all repetitive objects
        unique_indices = [data["object_id_stack"].index(x) for x in set(data["object_id_stack"])]
        object_stack = np.vstack(data["object_stack"])
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = np.zeros((self.params["postprocess"]["max_num"], 7))
        mask = np.zeros(self.params["postprocess"]["max_num"])
        object_bbx_center[: object_stack.shape[0], :] = object_stack
        mask[: object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(data["projected_lidar_stack"])

        # data augmentation
        projected_lidar_stack, object_bbx_center, mask = self.augment(projected_lidar_stack, object_bbx_center, mask)

        # we do lidar filtering in the stacked lidar
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack, self.params["preprocess"]["cav_lidar_range"])
        # augmentation may remove some of the bbx out of range
        object_bbx_center_valid = object_bbx_center[mask == 1]
        object_bbx_center_valid, range_mask = box_utils.mask_boxes_outside_range_numpy(
            object_bbx_center_valid, self.params["preprocess"]["cav_lidar_range"], self.params["postprocess"]["order"], return_mask=True
        )
        mask[object_bbx_center_valid.shape[0] :] = 0
        object_bbx_center[: object_bbx_center_valid.shape[0]] = object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0] :] = 0
        unique_indices = list(np.array(unique_indices)[range_mask])

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask)

        processed_data_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": mask,
                "object_ids": [data["object_id_stack"][i] for i in unique_indices],
                "anchor_box": anchor_box,
                "processed_lidar": lidar_dict,
                "label_dict": label_dict,
            }
        )

        if self.visualize:
            processed_data_dict["ego"].update({"origin_lidar": projected_lidar_stack})

        return processed_data_dict
