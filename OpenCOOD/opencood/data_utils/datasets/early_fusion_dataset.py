"""
Dataset class for early fusion
"""

import math
import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

import opencood.data_utils.datasets
from opencood.utils import box_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.models.communication_adapters import PoseFrameMetadata
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2

logger = logging.getLogger("cavise.opencda.OpenCOOD.opencood.data_utils.datasets.early_fusion_dataset")


class EarlyFusionDataset(basedataset.BaseDataset):
    """
    This dataset is used for early fusion, where each CAV transmit the raw
    point cloud to the ego vehicle.
    """

    def __init__(self, params, visualize, train=True, payload_handler=None):
        super(EarlyFusionDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)

        self.payload_handler = payload_handler
        self.module_name = "OpenCOOD.EarlyFusionDataset"

    def __find_ego_vehicle(self, base_data_dict):
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

    def extract_data(
        self,
        idx: int,
        agent_payload_builder: Callable[[npt.NDArray[Any], PoseFrameMetadata], object],
    ) -> None:
        """
        Publish sender-local observations through the communication adapter.

        Parameters
        ----------
        idx : int
            Dataset frame to publish.
        agent_payload_builder : Callable
            Adapter callback converting one local observation into its wire
            representation.
        """
        base_data_dict = self.retrieve_base_data(idx)

        if self.payload_handler is not None:
            for cav_id, selected_cav_base in base_data_dict.items():
                lidar_points, _ = self.__prepare_local_lidar(selected_cav_base)
                payload = agent_payload_builder(
                    lidar_points,
                    PoseFrameMetadata(
                        lidar_pose=selected_cav_base["params"]["lidar_pose"],
                        capture_frame=idx - int(selected_cav_base["time_delay"]),
                    ),
                )
                self.payload_handler.set_opencda_payload(cav_id, self.module_name, payload)

    def __process_with_messages(self, ego_id, ego_lidar_pose, base_data_dict):
        projected_lidar_stack = []

        ego_cav_base = base_data_dict.get(ego_id)
        ego_cav_processed = self.get_item_single_car(ego_cav_base, ego_lidar_pose)

        projected_lidar_stack.append(ego_cav_processed["projected_lidar"])

        if ego_id in self.payload_handler.current_artery_payload:
            for cav_id, _ in base_data_dict.items():
                if cav_id == ego_id:
                    continue
                raw_payload = self.payload_handler.get_artery_payload(ego_id, cav_id, self.module_name)
                if raw_payload is None:
                    continue
                if self.communication_adapter is None:
                    raise RuntimeError("Early-fusion payload decoding requires a communication adapter")
                decoded_payload = self.communication_adapter.decode_received_payload(
                    raw_payload,
                    ego_lidar_pose,
                )
                projected_lidar_stack.append(decoded_payload["projected_lidar"])

        return {"projected_lidar_stack": projected_lidar_stack}

    def __process_without_messages(self, ego_lidar_pose, base_data_dict):
        projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            dx = selected_cav_base["params"]["lidar_pose"][0] - ego_lidar_pose[0]
            dy = selected_cav_base["params"]["lidar_pose"][1] - ego_lidar_pose[1]
            distance = math.hypot(dx, dy)

            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)
            projected_lidar_stack.append(selected_cav_processed["projected_lidar"])

        return {"projected_lidar_stack": projected_lidar_stack}

    def build_local_supervision(self, base_data_dict: dict[str, Any], ego_lidar_pose: list[float]) -> dict[str, Any]:
        """
        Build ego-frame ground truth from the local scene snapshot.

        Parameters
        ----------
        base_data_dict : dict[str, Any]
            Complete local dataset snapshot for the current scene.
        ego_lidar_pose : list[float]
            Current ego LiDAR pose in world coordinates.

        Returns
        -------
        dict[str, Any]
            Ground-truth boxes, mask, and object identifiers.
        """
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center(list(base_data_dict.values()), ego_lidar_pose)
        return {
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
            "object_ids": object_ids,
        }

    def build_visualization_context(
        self,
        ego_id: str,
        ego_lidar_pose: list[float],
        base_data_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build point-cloud visualization data from the local scene.

        Parameters
        ----------
        ego_id : str
            Identifier of the receiving ego agent.
        ego_lidar_pose : list[float]
            Current ego LiDAR pose in world coordinates.
        base_data_dict : dict[str, Any]
            Local dataset snapshot, independent of delivered messages.

        Returns
        -------
        dict[str, Any]
            Per-agent point clouds, identities, roles, and spoofing masks.
        """
        if not self.visualize:
            return {}

        projected_lidar_stack = []
        projected_lidar_roles = []
        projected_lidar_agent_ids = []
        projected_lidar_spoofing_masks = []
        lidar_range = self.params["preprocess"]["cav_lidar_range"]

        for cav_id, cav_base in base_data_dict.items():
            cav_processed = self.get_item_single_car(cav_base, ego_lidar_pose)
            projected_lidar = cav_processed["projected_lidar"]
            range_mask = (
                (projected_lidar[:, 0] > lidar_range[0])
                & (projected_lidar[:, 0] < lidar_range[3])
                & (projected_lidar[:, 1] > lidar_range[1])
                & (projected_lidar[:, 1] < lidar_range[4])
                & (projected_lidar[:, 2] > lidar_range[2])
                & (projected_lidar[:, 2] < lidar_range[5])
            )
            projected_lidar_stack.append(projected_lidar[range_mask])
            projected_lidar_roles.append("ego" if cav_id == ego_id else "other")
            projected_lidar_agent_ids.append(cav_id)
            projected_lidar_spoofing_masks.append(cav_processed["projected_lidar_spoofing_mask"][range_mask])

        return {
            "origin_lidar": np.vstack(projected_lidar_stack),
            "origin_lidar_by_agent": projected_lidar_stack,
            "origin_lidar_roles": projected_lidar_roles,
            "origin_lidar_agent_ids": projected_lidar_agent_ids,
            "origin_lidar_spoofing_masks": projected_lidar_spoofing_masks,
        }

    def assemble_inference_sample(
        self,
        inference_input: dict[str, Any],
        local_supervision: dict[str, Any],
        visualization_context: dict[str, Any],
    ) -> OrderedDict[str, dict[str, Any]]:
        """
        Combine delivered LiDAR with local supervision and visualization.

        Parameters
        ----------
        inference_input : dict[str, Any]
            Ego LiDAR plus point clouds from messages that were delivered.
        local_supervision : dict[str, Any]
            Ground truth produced solely from the local dataset snapshot.
        visualization_context : dict[str, Any]
            Visualization data produced solely from the local snapshot.

        Returns
        -------
        collections.OrderedDict
            Early-fusion sample ready for collation.
        """
        projected_lidar_stack = np.vstack(inference_input["projected_lidar_stack"])
        object_bbx_center = local_supervision["object_bbx_center"]
        object_bbx_mask = local_supervision["object_bbx_mask"]

        projected_lidar_stack, object_bbx_center, object_bbx_mask = self.augment(
            projected_lidar_stack,
            object_bbx_center,
            object_bbx_mask,
        )

        lidar_range = self.params["preprocess"]["cav_lidar_range"]
        lidar_mask = (
            (projected_lidar_stack[:, 0] > lidar_range[0])
            & (projected_lidar_stack[:, 0] < lidar_range[3])
            & (projected_lidar_stack[:, 1] > lidar_range[1])
            & (projected_lidar_stack[:, 1] < lidar_range[4])
            & (projected_lidar_stack[:, 2] > lidar_range[2])
            & (projected_lidar_stack[:, 2] < lidar_range[5])
        )
        projected_lidar_stack = projected_lidar_stack[lidar_mask]

        valid_indices = np.flatnonzero(object_bbx_mask == 1)
        valid_object_ids = [local_supervision["object_ids"][index] for index in valid_indices]
        valid_boxes, range_mask = box_utils.mask_boxes_outside_range_numpy(
            object_bbx_center[valid_indices],
            lidar_range,
            self.params["postprocess"]["order"],
            return_mask=True,
        )
        object_bbx_center.fill(0)
        object_bbx_mask.fill(0)
        object_bbx_center[: valid_boxes.shape[0]] = valid_boxes
        object_bbx_mask[: valid_boxes.shape[0]] = 1

        anchor_box = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center,
            anchors=anchor_box,
            mask=object_bbx_mask,
        )
        ego_sample = {
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
            "object_ids": [object_id for object_id, keep in zip(valid_object_ids, range_mask) if keep],
            "anchor_box": anchor_box,
            "processed_lidar": self.pre_processor.preprocess(projected_lidar_stack),
            "label_dict": label_dict,
            **visualization_context,
        }
        return OrderedDict({"ego": ego_sample})

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        ego_id, ego_lidar_pose = self.__find_ego_vehicle(base_data_dict)

        if self.payload_handler is not None:
            inference_input = self.__process_with_messages(ego_id, ego_lidar_pose, base_data_dict)
        else:
            inference_input = self.__process_without_messages(ego_lidar_pose, base_data_dict)

        local_supervision = self.build_local_supervision(base_data_dict, ego_lidar_pose)
        visualization_context = self.build_visualization_context(ego_id, ego_lidar_pose, base_data_dict)
        return self.assemble_inference_sample(inference_input, local_supervision, visualization_context)

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project one agent's LiDAR into ego coordinates.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        # calculate the transformation matrix
        transformation_matrix = x1_to_x2(selected_cav_base["params"]["lidar_pose"], ego_pose)

        lidar_np, spoofing_mask = self.__prepare_local_lidar(selected_cav_base)
        # project the lidar to ego space
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)

        return {
            "projected_lidar": lidar_np,
            "projected_lidar_spoofing_mask": spoofing_mask,
        }

    @staticmethod
    def __prepare_local_lidar(
        selected_cav_base: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Shuffle and remove ego returns without changing coordinate frames.

        Parameters
        ----------
        selected_cav_base : dict[str, Any]
            Raw dataset entry for one sending agent.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Sender-local LiDAR points and their aligned spoofing mask.
        """
        lidar_np = selected_cav_base["lidar_np"]
        spoofing_mask = np.asarray(
            selected_cav_base.get(
                "spoofing_mask",
                np.zeros((lidar_np.shape[0],), dtype=np.bool_),
            ),
            dtype=np.bool_,
        )

        shuffle_idx = np.random.permutation(lidar_np.shape[0])
        lidar_np = np.array(lidar_np[shuffle_idx], copy=True)
        spoofing_mask = np.array(spoofing_mask[shuffle_idx], copy=True)

        ego_mask = (lidar_np[:, 0] >= -1.95) & (lidar_np[:, 0] <= 2.95) & (lidar_np[:, 1] >= -1.1) & (lidar_np[:, 1] <= 1.1)
        keep_mask = np.logical_not(ego_mask)
        return lidar_np[keep_mask], spoofing_mask[keep_mask]

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        for cav_id, cav_content in batch.items():
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
            processed_lidar_torch_dict = self.pre_processor.collate_batch([cav_content["processed_lidar"]])
            # label dictionary
            label_torch_dict = self.post_processor.collate_batch([cav_content["label_dict"]])

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
                origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({"origin_lidar": origin_lidar})
                if "origin_lidar_by_agent" in cav_content:
                    output_dict[cav_id].update(
                        {
                            "origin_lidar_by_agent": [torch.from_numpy(np.array(points)) for points in cav_content["origin_lidar_by_agent"]],
                            "origin_lidar_roles": list(cav_content["origin_lidar_roles"]),
                            "origin_lidar_agent_ids": list(cav_content["origin_lidar_agent_ids"]),
                            "origin_lidar_spoofing_masks": [
                                torch.from_numpy(np.asarray(mask, dtype=np.bool_)) for mask in cav_content["origin_lidar_spoofing_masks"]
                            ],
                        }
                    )

        return output_dict

    def post_process(self, data_dict, output_dict):
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
        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor
