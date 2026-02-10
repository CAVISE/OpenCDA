"""
Late fusion dataset for cooperative perception.

This module provides dataset functionality for late fusion approaches where
each connected autonomous vehicle (CAV) transmits detection outputs to the
ego vehicle for collaborative 3D object detection.
"""

import math
import random
import logging
from collections import OrderedDict

import numpy as np
import torch
import opencood.data_utils.datasets
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, shuffle_points, downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
from opencda.core.common.communication.serialize import MessageHandler

from typing import Dict, List, Tuple, Any, Optional
from numpy.typing import npt

logger = logging.getLogger("cavise.OpenCOOD.opencood.data_utils.datasets.late_fusion_dataset")


class LateFusionDataset(basedataset.BaseDataset):
    """
    Dataset class for late fusion where each vehicle transmits detection outputs to the ego vehicle.

    This class handles the processing of LiDAR data and object detection labels for multiple connected
    autonomous vehicles (CAVs) in a cooperative perception setting. It supports both synchronous and
    asynchronous data processing with message handling capabilities.

    Attributes
    ----------
    pre_processor : object
        Module for preprocessing LiDAR data.
    post_processor : object
        Module for post-processing detection results.
    message_handler : Optional[object]
        Handler for inter-vehicle communication.

    Parameters
    ----------
    params : Dict[str, Any]
        Configuration dictionary containing dataset parameters.
    visualize : bool
        Whether to include visualization data.
    train : bool, optional
        Whether the dataset is used for training. Default is True.
    message_handler : Optional[MessageHandler], optional
        Handler for inter-vehicle communication. Default is None.

    Attributes
    ----------
    pre_processor : object
        Module for preprocessing LiDAR data.
    post_processor : object
        Module for post-processing detection results.
    message_handler : MessageHandler or None
        Handler for inter-vehicle communication.
    """

    def __init__(self, params: Dict[str, Any], visualize: bool, train: bool = True, message_handler: Optional[MessageHandler] = None):
        super(LateFusionDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)

        self.message_handler = message_handler

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample by index.

        Parameters
        ----------
        idx : int
            Index of the data sample to retrieve.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the processed data sample.
        """
        base_data_dict = self.retrieve_base_data(idx)
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict)

        return reformat_data_dict

    @staticmethod
    def __wrap_ndarray(ndarray: np.ndarray) -> Dict[str, Any]:
        """
        Convert a numpy array to a serializable dictionary.

        Parameters
        ----------
        ndarray : np.ndarray
            Input numpy array.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing array data, shape, and dtype.
        """
        return {"data": ndarray.tobytes(), "shape": ndarray.shape, "dtype": str(ndarray.dtype)}

    def extract_data(self, idx: int) -> None:
        """
        Extract and process data for a single frame.

        Parameters
        ----------
        idx : int
            Index of the data point to extract.
        """
        base_data_dict = self.retrieve_base_data(idx)

        if self.message_handler is not None:
            for cav_id, selected_cav_base in base_data_dict.items():
                selected_cav_processed = self.get_item_single_car(selected_cav_base)

                with self.message_handler.handle_opencda_message(cav_id, self.module_name) as msg:
                    msg["object_ids"] = {
                        "name": "object_ids",
                        "label": "LABEL_REPEATED",
                        "type": "int64",
                        "data": selected_cav_processed["object_ids"],
                    }

                    msg["lidar_pose"] = {
                        "name": "lidar_pose",
                        "label": "LABEL_REPEATED",
                        "type": "float",
                        "data": selected_cav_base["params"]["lidar_pose"],
                    }

                    msg["object_bbx_center"] = {
                        "name": "object_bbx_center",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["object_bbx_center"]),
                    }

                    msg["object_bbx_mask"] = {
                        "name": "object_bbx_mask",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["object_bbx_mask"]),
                    }

                    msg["anchor_box"] = {
                        "name": "anchor_box",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["anchor_box"]),
                    }

                    msg["pos_equal_one"] = {
                        "name": "pos_equal_one",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["label_dict"]["pos_equal_one"]),
                    }

                    msg["neg_equal_one"] = {
                        "name": "neg_equal_one",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["label_dict"]["neg_equal_one"]),
                    }

                    msg["targets"] = {
                        "name": "targets",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["label_dict"]["targets"]),
                    }

                    msg["voxel_features"] = {
                        "name": "voxel_features",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["processed_lidar"]["voxel_features"]),
                    }

                    msg["voxel_coords"] = {
                        "name": "voxel_coords",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["processed_lidar"]["voxel_coords"]),
                    }

                    msg["voxel_num_points"] = {
                        "name": "voxel_num_points",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["processed_lidar"]["voxel_num_points"]),
                    }

                    msg["origin_lidar"] = {
                        "name": "origin_lidar",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["origin_lidar"]),
                    }

    def __find_ego_vehicle(self, base_data_dict: Dict[str, Any]) -> Tuple[str, List[float]]:
        """
        Find the ego vehicle in the base data dictionary.

        Parameters
        ----------
        base_data_dict : Dict[str, Any]
            Dictionary containing data from all CAVs.

        Returns
        -------
        ego_id : str
            ID of the ego vehicle.
        ego_lidar_pose : List[float]
            Pose of the ego vehicle's LiDAR.

        Raises
        ------
        AssertionError
            If ego vehicle is not found or not the first element.
        """
        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                ego_id = cav_id  # NOTE error Incompatible types in assignment (expression has type "str", variable has type "int")
                ego_lidar_pose = cav_content["params"]["lidar_pose"]
                break

        assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1

        return ego_id, ego_lidar_pose  # NOTE the same problem with "str"/"int"

    def __process_with_messages(self, ego_id: str, ego_lidar_pose: List[float], base_data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using message passing between connected vehicles.

        Parameters
        ----------
        ego_id : str
            ID of the ego vehicle.
        ego_lidar_pose : List[float]
            LiDAR pose of the ego vehicle.
        base_data_dict : Dict[str, Any]
            Dictionary containing base data for all vehicles.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed data from all connected vehicles.
        """
        processed_data_dict = OrderedDict()

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        anchor_box = []
        pos_equal_one = []
        neg_equal_one = []
        targets = []
        voxel_features = []
        voxel_coords = []
        voxel_num_points = []
        transformation_matrix = []
        origin_lidar: Optional[List[Any]] = [] if self.visualize else None

        ego_cav_base = base_data_dict.get(ego_id)
        ego_cav_processed = self.get_item_single_car(ego_cav_base)  # NOTE None-check is required

        object_bbx_center.append(ego_cav_processed["object_bbx_center"])
        object_bbx_mask.append(ego_cav_processed["object_bbx_mask"])
        object_ids += ego_cav_processed["object_ids"]
        anchor_box.append(ego_cav_processed["anchor_box"])
        pos_equal_one.append(ego_cav_processed["label_dict"]["pos_equal_one"])
        neg_equal_one.append(ego_cav_processed["label_dict"]["neg_equal_one"])
        targets.append(ego_cav_processed["label_dict"]["targets"])
        voxel_features.append(ego_cav_processed["processed_lidar"]["voxel_features"])
        voxel_coords.append(ego_cav_processed["processed_lidar"]["voxel_coords"])
        voxel_num_points.append(ego_cav_processed["processed_lidar"]["voxel_num_points"])

        transformation_matrix_info = x1_to_x2(ego_lidar_pose, ego_lidar_pose)
        ego_cav_processed["transformation_matrix"] = transformation_matrix_info
        transformation_matrix.append(transformation_matrix_info)

        if self.visualize:
            origin_lidar.append(ego_cav_processed["origin_lidar"])

        processed_data_dict.update({"ego": ego_cav_processed})

        if ego_id in self.message_handler.current_message_artery:  # NOTE None-check is required
            for cav_id, _ in base_data_dict.items():
                if cav_id in self.message_handler.current_message_artery[ego_id]:  # NOTE None-check is required
                    with self.message_handler.handle_artery_message(ego_id, cav_id, self.module_name) as msg:  # NOTE None-check is required
                        object_ids += msg["object_ids"]
                        cav_lidar_pose = msg["lidar_pose"]

                        def unpack(msg_key: str) -> npt.NDArray[Any]:
                            array = np.frombuffer(msg[msg_key]["data"], np.dtype(msg[msg_key]["dtype"]))
                            return array.reshape(msg[msg_key]["shape"])

                        object_bbx_center.append(unpack("object_bbx_center"))
                        object_bbx_mask.append(unpack("object_bbx_mask"))
                        anchor_box.append(unpack("anchor_box"))
                        pos_equal_one.append(unpack("pos_equal_one"))
                        neg_equal_one.append(unpack("neg_equal_one"))
                        targets.append(unpack("targets"))
                        voxel_features.append(unpack("voxel_features"))
                        voxel_coords.append(unpack("voxel_coords"))
                        voxel_num_points.append(unpack("voxel_num_points"))

                        transformation_matrix_info = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
                        transformation_matrix.append(transformation_matrix_info)

                        if self.visualize:
                            origin_lidar.append(unpack("origin_lidar"))

                    update_cav = "ego" if cav_id == ego_id else cav_id

                    selected_cav_processed = {
                        "object_bbx_center": object_bbx_center,
                        "object_bbx_mask": object_bbx_mask,
                        "object_ids": object_ids,
                        "anchor_box": anchor_box,
                        "pos_equal_one": pos_equal_one,
                        "neg_equal_one": neg_equal_one,
                        "targets": targets,
                        "voxel_features": voxel_features,
                        "voxel_coords": voxel_coords,
                        "voxel_num_points": voxel_num_points,
                        "transformation_matrix": transformation_matrix,
                        "origin_lidar": origin_lidar or [],
                    }

                    processed_data_dict.update({update_cav: selected_cav_processed})

        return processed_data_dict

    def __process_without_messages(self, ego_id: str, ego_lidar_pose: List[float], base_data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data without using message passing.

        Parameters
        ----------
        ego_id : str
            ID of the ego vehicle.
        ego_lidar_pose : List[float]
            LiDAR pose of the ego vehicle.
        base_data_dict : Dict[str, Any]
            Dictionary containing base data for all vehicles.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed data from all vehicles within communication range.
        """
        processed_data_dict = OrderedDict()

        for cav_id, selected_cav_base in base_data_dict.items():
            dx = selected_cav_base["params"]["lidar_pose"][0] - ego_lidar_pose[0]
            dy = selected_cav_base["params"]["lidar_pose"][1] - ego_lidar_pose[1]
            distance = math.hypot(dx, dy)

            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            # find the transformation matrix from current cav to ego.
            cav_lidar_pose = selected_cav_base["params"]["lidar_pose"]
            transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)

            selected_cav_processed = self.get_item_single_car(selected_cav_base)
            selected_cav_processed.update({"transformation_matrix": transformation_matrix})
            update_cav = "ego" if cav_id == ego_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        selected_cav_base : Dict[str, Any]
            Dictionary containing a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
        Processed data in CAV's local frame:
            - processed_lidar : dict
                Preprocessed LiDAR (voxels/BEV).
            - anchor_box : NDArray
                Generated anchor boxes.
            - object_bbx_center : NDArray
                Object bounding boxes (max_num, 7).
            - object_bbx_mask : NDArray
                Valid object mask (max_num,).
            - object_ids : list of str
                Object IDs.
            - label_dict : dict
                Ground truth labels for training.
            - origin_lidar : NDArray, optional
                Raw point cloud (if visualize=True).
        """
        selected_cav_processed = {}

        # filter lidar
        lidar_np = selected_cav_base["lidar_np"]
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np, self.params["preprocess"]["cav_lidar_range"])
        # remove points that hit ego vehicle
        lidar_np = mask_ego_points(lidar_np)

        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center(  # NOTE None-check is required
            [selected_cav_base], selected_cav_base["params"]["lidar_pose"]
        )
        # data augmentation
        lidar_np, object_bbx_center, object_bbx_mask = self.augment(lidar_np, object_bbx_center, object_bbx_mask)

        if self.visualize:
            selected_cav_processed.update({"origin_lidar": lidar_np})

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(lidar_np)  # NOTE None-check is required
        selected_cav_processed.update({"processed_lidar": lidar_dict})

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()  # NOTE None-check is required
        selected_cav_processed.update({"anchor_box": anchor_box})

        selected_cav_processed.update({"object_bbx_center": object_bbx_center, "object_bbx_mask": object_bbx_mask, "object_ids": object_ids})

        # generate targets label
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center, anchors=anchor_box, mask=object_bbx_mask
        )  # NOTE None-check is required
        selected_cav_processed.update({"label_dict": label_dict})

        return selected_cav_processed

    def get_item_train(self, base_data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process training data for a single sample.

        Parameters
        ----------
        base_data_dict : Dict[str, Any]
            Dictionary containing base data for all vehicles.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed training data for a single vehicle.
            During training, returns a random vehicle's data unless in visualization mode,
            in which case it returns the ego vehicle's data.
        """
        processed_data_dict = OrderedDict()

        # during training, we return a random cav's data
        if not self.visualize:
            _, selected_cav_base = random.choice(list(base_data_dict.items()))
        else:
            _, selected_cav_base = list(base_data_dict.items())[0]

        selected_cav_processed = self.get_item_single_car(selected_cav_base)
        processed_data_dict.update({"ego": selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process test data for a single sample.

        Parameters
        ----------
        base_data_dict : Dict[str, Any]
            Dictionary containing base data for all vehicles.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed test data for all vehicles.
            If message handler is available, uses message passing; otherwise,
            processes data directly from base data.
        """
        ego_id = -1
        ego_lidar_pose: List[float] = []

        ego_id, ego_lidar_pose = self.__find_ego_vehicle(
            base_data_dict
        )  # NOTE Incompatible types in assignment (expression has type "str", variable has type "int")

        if self.message_handler is not None:
            processed_data_dict = self.__process_with_messages(
                ego_id, ego_lidar_pose, base_data_dict
            )  # NOTE Incompatible types in assignment (expression has type "str", variable has type "int")
        else:
            processed_data_dict = self.__process_without_messages(
                ego_id, ego_lidar_pose, base_data_dict
            )  # NOTE Incompatible types in assignment (expression has type "str", variable has type "int")

        return processed_data_dict

    def collate_batch_test(self, batch: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Collate function for test data loader.

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            List of data samples to collate (batch size must be 1).

        Returns
        -------
        output_dict : dict of {str: dict}
        Dictionary with data for each CAV:
            - {cav_id} : dict
                - object_bbx_center : torch.Tensor
                - object_bbx_mask : torch.Tensor
                - anchor_box : torch.Tensor, optional
                - processed_lidar : dict of tensors
                - label_dict : dict of tensors
                - object_ids : list of str
                - transformation_matrix : torch.Tensor
                - origin_lidar : torch.Tensor, optional (if visualize=True)
            - ego : dict, optional (if visualize=True)
                - origin_lidar : torch.Tensor
                    Fused LiDAR from all CAVs projected to ego frame.

        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]  # NOTE This line assumes `batch` is a list and unwraps the first element

        output_dict: Dict[str, Dict[str, Any]] = {}

        # for late fusion, we also need to stack the lidar for better
        # visualization
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})  # NOTE Incompatible types
            # shape: (1, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array([cav_content["object_bbx_center"]]))
            object_bbx_mask = torch.from_numpy(np.array([cav_content["object_bbx_mask"]]))
            object_ids = cav_content["object_ids"]

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content["anchor_box"] is not None:
                output_dict[cav_id].update({"anchor_box": torch.from_numpy(np.array(cav_content["anchor_box"]))})  # NOTE Incompatible types
            if self.visualize:
                transformation_matrix = cav_content["transformation_matrix"]
                origin_lidar = [cav_content["origin_lidar"]]

                projected_lidar = cav_content["origin_lidar"]
                projected_lidar[:, :3] = box_utils.project_points_by_matrix_torch(projected_lidar[:, :3], transformation_matrix)
                projected_lidar_list.append(projected_lidar)

            # processed lidar dictionary
            processed_lidar_torch_dict = self.pre_processor.collate_batch([cav_content["processed_lidar"]])  # NOTE None-check is required
            # label dictionary
            label_torch_dict = self.post_processor.collate_batch([cav_content["label_dict"]])  # NOTE None-check is required

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = torch.from_numpy(np.array(cav_content["transformation_matrix"])).float()

            output_dict[cav_id].update(  # NOTE Incompatible types
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
                origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({"origin_lidar": origin_lidar})  # NOTE Incompatible types

        if self.visualize:
            projected_lidar_stack = torch.from_numpy(np.vstack(projected_lidar_list))
            output_dict["ego"].update({"origin_lidar": projected_lidar_stack})

        return output_dict

    def post_process(self, data_dict: Dict[str, Any], output_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing the origin input data of model.
        output_dict : Dict[str, Any]
            Dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            Tensor of prediction bounding boxes after NMS.
        pred_score : torch.Tensor
            Tensor of confidence scores for predicted boxes.
        gt_box_tensor : torch.Tensor
            Tensor of ground truth bounding boxes.
        """
        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)  # NOTE None-check is required
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)  # NOTE None-check is required

        return pred_box_tensor, pred_score, gt_box_tensor
