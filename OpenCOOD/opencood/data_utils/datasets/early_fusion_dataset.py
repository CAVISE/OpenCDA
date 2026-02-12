"""
Early fusion dataset for cooperative perception.

This module provides the dataset implementation for early fusion approaches in
cooperative autonomous driving, where LiDAR point clouds from multiple vehicles
are fused before feature extraction.
"""

import math
import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional

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
from opencda.core.common.communication.serialize import MessageHandler

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
    message_handler : Optional[message_handler]
        Handler for inter-vehicle communication. Default is None.

    Attributes
    ----------
    pre_processor : Any
        Preprocessor for point cloud data.
    post_processor : Any
        Postprocessor for generating labels and bounding boxes.
    message_handler : Optional[message_handler]
        Handler for inter-vehicle communication.
    module_name : str
        Identifier for the module.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        visualize: bool,
        train: bool = True,
        message_handler: Optional[MessageHandler] = None,
    ):
        super().__init__(params, visualize, train)

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)

        self.message_handler = message_handler
        self.module_name = "OpenCOOD.EarlyFusionDataset"

    def __find_ego_vehicle(self, base_data_dict: Dict[str, Dict[str, Any]]) -> Tuple[str, List[float]]:
        """
        Find the ego vehicle in the base data dictionary.

        Parameters
        ----------
        base_data_dict : Dict[int, Dict[str, Any]]
            Dictionary containing data for all CAVs.

        Returns
        -------
        Tuple[int, List[float]]
            Tuple containing ego vehicle ID and its LiDAR pose.
        """
        ego_id = ""
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                ego_id = cav_id
                ego_lidar_pose = cav_content["params"]["lidar_pose"]
                break

        assert ego_id != ""
        assert len(ego_lidar_pose) > 0

        return ego_id, ego_lidar_pose

    @staticmethod
    def __wrap_ndarray(ndarray: NDArray) -> Dict[str, Any]:
        return {"data": ndarray.tobytes(), "shape": ndarray.shape, "dtype": str(ndarray.dtype)}

    def extract_data(self, idx: int) -> None:
        """
        Extract and package data for V2V communication simulation.

        Retrieves base data, processes each CAV's data, and sends messages
        through the message handler for communication simulation.

        Parameters
        ----------
        idx : int
        Dataset sample index corresponding to a specific scenario frame.
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
        ego_id: str,
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

        message_handler = self.message_handler
        assert message_handler is not None

        ego_cav_base = base_data_dict.get(ego_id)
        if ego_cav_base is None:
            raise ValueError("Ego vehicle data not found in base_data_dict.")
        ego_cav_processed = self.get_item_single_car(ego_cav_base, ego_lidar_pose)
        object_id_stack += ego_cav_processed["object_ids"]
        object_stack.append(ego_cav_processed["object_bbx_center"])
        projected_lidar_stack.append(ego_cav_processed["projected_lidar"])

        if ego_id in message_handler.current_message_artery:
            for cav_id, _ in base_data_dict.items():
                if cav_id in message_handler.current_message_artery[ego_id]:
                    with message_handler.handle_artery_message(ego_id, cav_id, self.module_name) as msg:
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

        # loop over all CAVs to process information
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
        processed_data_dict : dict of {str: dict}
        Dictionary containing:
            - ego : dict
                - object_bbx_center : NDArray (max_num, 7)
                    Padded object bounding boxes.
                - object_bbx_mask : NDArray (max_num,)
                    Binary mask for valid objects.
                - object_ids : list of str
                    Unique object IDs after deduplication.
                - anchor_box : NDArray
                    Generated anchor boxes for detection.
                - processed_lidar : dict
                    Preprocessed LiDAR (voxels/BEV).
                - label_dict : dict
                    Ground truth labels for training.
                - origin_lidar : NDArray, optional
                    Raw fused point cloud (if visualize=True).
        """
        base_data_dict = self.retrieve_base_data(idx)
        processed_data_dict: OrderedDict = OrderedDict()
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

        pre_processor = self.pre_processor
        post_processor = self.post_processor
        assert pre_processor is not None
        assert post_processor is not None

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = pre_processor.preprocess(projected_lidar_stack)

        # generate the anchor boxes
        anchor_box = post_processor.generate_anchor_box()

        # generate targets label
        label_dict = post_processor.generate_label(gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask)

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

    def get_item_single_car(self, selected_cav_base: Dict[str, Any], ego_pose: List[float]) -> Dict[str, Any]:
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
        Processed CAV data:
            - object_bbx_center : NDArray
                Valid object bounding boxes in ego frame.
            - object_ids : list of str
                Object IDs.
            - projected_lidar : NDArray
                LiDAR points (N, 4) in ego frame.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = x1_to_x2(selected_cav_base["params"]["lidar_pose"], ego_pose)

        # retrieve objects under ego coordinates
        post_processor = self.post_processor
        assert post_processor is not None
        object_bbx_center, object_bbx_mask, object_ids = post_processor.generate_object_center([selected_cav_base], ego_pose)

        # filter lidar
        lidar_np = selected_cav_base["lidar_np"]
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)

        selected_cav_processed.update(
            {"object_bbx_center": object_bbx_center[object_bbx_mask == 1], "object_ids": object_ids, "projected_lidar": lidar_np}
        )

        return selected_cav_processed

    def collate_batch_test(self, batch: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        output_dict : dict of {str: dict}
            Reformatted batch dictionary with tensors:
                - {cav_id}: dict
                    - object_bbx_center : torch.Tensor (1, max_num, 7)
                    - object_bbx_mask : torch.Tensor (1, max_num)
                    - anchor_box : torch.Tensor, optional
                    - processed_lidar : dict of tensors
                    - label_dict : dict of tensors
                    - object_ids : list of str
                    - transformation_matrix : torch.Tensor (4, 4)
                    - origin_lidar : torch.Tensor, optional (if visualize=True)
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch_dict = batch[0]

        output_dict: Dict[str, Dict] = {}

        for cav_id, cav_content in batch_dict.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array([cav_content["object_bbx_center"]]))
            object_bbx_mask = torch.from_numpy(np.array([cav_content["object_bbx_mask"]]))
            object_ids = cav_content["object_ids"]

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content["anchor_box"] is not None:
                output_dict[cav_id].update({"anchor_box": torch.from_numpy(np.array(cav_content["anchor_box"]))})
            if self.visualize:
                origin_lidar = [cav_content["origin_lidar"]]

            # processed lidar dictionary
            pre_processor = self.pre_processor
            post_processor = self.post_processor
            assert pre_processor is not None
            assert post_processor is not None
            processed_lidar_torch_dict = pre_processor.collate_batch([cav_content["processed_lidar"]])
            # label dictionary
            label_torch_dict = post_processor.collate_batch([cav_content["label_dict"]])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()

            output_dict[cav_id].update(
                {
                    "object_bbx_center": object_bbx_center,
                    "object_bbx_mask": object_bbx_mask,
                    "processed_lidar": processed_lidar_torch_dict,
                    "label_dict": label_torch_dict,
                    "object_ids": object_ids,
                    "transformation_matrix": transformation_matrix_torch,
                }
            )

            if self.visualize:
                origin_lidar_arr = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar_tensor = torch.from_numpy(origin_lidar_arr)
                output_dict[cav_id].update({"origin_lidar": origin_lidar_tensor})

        return output_dict

    def post_process(self, data_dict: Dict[str, Any], output_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        post_processor = self.post_processor
        assert post_processor is not None
        pred_box_tensor, pred_score = post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor
