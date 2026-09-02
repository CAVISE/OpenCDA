"""
Dataset class for late fusion
"""

import math
import random
import logging
from collections import OrderedDict
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

import opencood.data_utils.datasets
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.models.communication_adapters import (
    LateFusionWirePayload,
    PoseFrameMetadata,
    build_inference_input,
    inference_input_to_dict,
)
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils
from opencood.utils.pcd_utils import pcd_to_np, mask_points_by_range, mask_ego_points, shuffle_points, downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2

logger = logging.getLogger("cavise.opencda.OpenCOOD.opencood.data_utils.datasets.late_fusion_dataset")


class LateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """

    def __init__(self, params, visualize, train=True, payload_handler=None):
        super(LateFusionDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)

        self.payload_handler = payload_handler
        self.module_name = "OpenCOOD.LateFusionDataset"

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(idx, base_data_dict)

        return reformat_data_dict

    def extract_data(self, idx):
        base_data_dict = self.retrieve_base_data(idx)

        if self.payload_handler is not None:
            for cav_id, selected_cav_base in base_data_dict.items():
                selected_cav_processed = self.__build_model_input(selected_cav_base)
                payload = self.build_wire_payload(selected_cav_base, selected_cav_processed, idx)
                self.payload_handler.set_opencda_payload(cav_id, self.module_name, payload)

    @staticmethod
    def build_wire_payload(
        selected_cav_base: dict[str, Any],
        selected_cav_processed: dict[str, Any],
        receive_frame: int,
    ) -> LateFusionWirePayload:
        """Build the late-fusion payload sent through Artery.

        Parameters
        ----------
        selected_cav_base : dict[str, Any]
            Raw dataset entry for one sending agent.
        selected_cav_processed : dict[str, Any]
            Processed data for the same sending agent.
        receive_frame : int
            Dataset frame in which the payload is sent.

        Returns
        -------
        LateFusionWirePayload
            Typed payload containing only remote model-input data.
        """
        return LateFusionWirePayload(
            inference_input=selected_cav_processed["inference_input"],
            metadata=PoseFrameMetadata(
                lidar_pose=selected_cav_base["params"]["lidar_pose"],
                capture_frame=receive_frame - int(selected_cav_base["time_delay"]),
            ),
        )

    @staticmethod
    def decode_wire_payload(payload: object) -> LateFusionWirePayload:
        """Validate and decode a late-fusion payload received from Artery.

        Parameters
        ----------
        payload : object
            Deserialized module payload received from Artery.

        Returns
        -------
        LateFusionWirePayload
            Validated late-fusion payload.

        Raises
        ------
        TypeError
            If the received payload has an unexpected type.
        """
        if not isinstance(payload, LateFusionWirePayload):
            raise TypeError(f"Expected LateFusionWirePayload, got {type(payload).__name__}")
        return payload

    def __find_ego_vehicle(self, base_data_dict):
        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                ego_id = cav_id
                ego_lidar_pose = cav_content["params"]["lidar_pose"]
                break

        assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1

        return ego_id, ego_lidar_pose

    def __retrieve_visualization_base_data(self, idx):
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        for cav_id, cav_content in scenario_database.items():
            cav_snapshot = cav_content[timestamp_key]
            data[cav_id] = OrderedDict()
            data[cav_id]["ego"] = cav_content["ego"]
            data[cav_id]["time_delay"] = 0
            data[cav_id]["params"] = self.reform_param(
                cav_content,
                ego_cav_content,
                timestamp_key,
                timestamp_key,
                cur_ego_pose_flag=True,
            )
            data[cav_id]["lidar_np"] = cav_snapshot["lidar_np"] if "lidar_np" in cav_snapshot else pcd_to_np(cav_snapshot["lidar"])
            if "spoofing_mask" in cav_snapshot:
                data[cav_id]["spoofing_mask"] = cav_snapshot["spoofing_mask"]
        return data

    def __project_lidar_for_visualization(self, cav_base, ego_lidar_pose):
        transformation_matrix = x1_to_x2(cav_base["params"]["lidar_pose"], ego_lidar_pose)
        lidar_np = np.array(cav_base["lidar_np"], copy=True)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        return mask_points_by_range(lidar_np, self.params["preprocess"]["cav_lidar_range"])

    def build_visualization_context(
        self,
        ego_id: str,
        ego_lidar_pose: list[float],
        base_data_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Build visualization data from the local scene snapshot.

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
            Per-agent projected point clouds and their local identities.
        """
        if not self.visualize:
            return {}

        origin_lidar_by_agent = []
        origin_lidar_roles = []
        origin_lidar_agent_ids = []

        for cav_id in base_data_dict:
            origin_lidar_by_agent.append(self.__project_lidar_for_visualization(base_data_dict[cav_id], ego_lidar_pose))
            origin_lidar_roles.append("ego" if cav_id == ego_id else "other")
            origin_lidar_agent_ids.append(cav_id)

        return {
            "origin_lidar_by_agent": origin_lidar_by_agent,
            "origin_lidar_roles": origin_lidar_roles,
            "origin_lidar_agent_ids": origin_lidar_agent_ids,
        }

    def __process_with_messages(self, ego_id, ego_lidar_pose, base_data_dict):
        processed_data_dict = OrderedDict()

        ego_cav_base = base_data_dict.get(ego_id)
        ego_cav_processed = self.__build_model_input(ego_cav_base)

        transformation_matrix_info = x1_to_x2(ego_lidar_pose, ego_lidar_pose)
        processed_data_dict["ego"] = {
            "inference_input": ego_cav_processed["inference_input"],
            "transformation_matrix": transformation_matrix_info,
        }

        if ego_id in self.payload_handler.current_artery_payload:
            for cav_id, _ in base_data_dict.items():
                raw_payload = self.payload_handler.get_artery_payload(ego_id, cav_id, self.module_name)
                if raw_payload is None:
                    continue
                payload = self.decode_wire_payload(raw_payload)
                transformation_matrix_info = x1_to_x2(payload.metadata.lidar_pose, ego_lidar_pose)

                selected_cav_processed = {
                    "inference_input": payload.inference_input,
                    "transformation_matrix": transformation_matrix_info,
                }

                processed_data_dict.update({cav_id: selected_cav_processed})

        return processed_data_dict

    def __process_without_messages(self, ego_id, ego_lidar_pose, base_data_dict):
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

            selected_cav_processed = self.__build_model_input(selected_cav_base)
            update_cav = "ego" if cav_id == ego_id else cav_id
            processed_data_dict[update_cav] = {
                "inference_input": selected_cav_processed["inference_input"],
                "transformation_matrix": transformation_matrix,
            }

        return processed_data_dict

    def build_local_supervision(self, base_data_dict: dict[str, Any], ego_lidar_pose: list[float]) -> dict[str, Any]:
        """Build ego-frame ground truth and labels from the local scene.

        Parameters
        ----------
        base_data_dict : dict[str, Any]
            Complete local dataset snapshot for the current scene.
        ego_lidar_pose : list[float]
            Current ego LiDAR pose in world coordinates.

        Returns
        -------
        dict[str, Any]
            Ego-frame ground truth, anchors, and target labels.
        """
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center(list(base_data_dict.values()), ego_lidar_pose)
        anchor_box = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center,
            anchors=anchor_box,
            mask=object_bbx_mask,
        )
        return {
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
            "object_ids": object_ids,
            "anchor_box": anchor_box,
            "label_dict": label_dict,
        }

    def assemble_inference_sample(
        self,
        inference_input: OrderedDict[str, dict[str, Any]],
        local_supervision: dict[str, Any],
        visualization_context: dict[str, Any],
    ) -> OrderedDict[str, dict[str, Any]]:
        """Combine delivered model inputs with local-only dataset data.

        Parameters
        ----------
        inference_input : collections.OrderedDict
            Ego model input plus model inputs from delivered messages.
        local_supervision : dict[str, Any]
            Ground truth and labels produced from the local scene.
        visualization_context : dict[str, Any]
            Visualization data produced from the local scene.

        Returns
        -------
        collections.OrderedDict
            Late-fusion sample ready for collation.
        """
        anchor_box = local_supervision["anchor_box"]
        for cav_content in inference_input.values():
            cav_content["anchor_box"] = anchor_box

        inference_input["ego"].update(local_supervision)
        inference_input["ego"].update(visualization_context)
        return inference_input

    def __prepare_lidar(self, selected_cav_base: dict[str, Any]) -> npt.NDArray[Any]:
        """Filter one agent's local LiDAR point cloud.

        Parameters
        ----------
        selected_cav_base : dict[str, Any]
            Raw dataset entry for one agent.

        Returns
        -------
        numpy.ndarray
            Shuffled and range-filtered LiDAR points.
        """
        lidar_np = shuffle_points(selected_cav_base["lidar_np"])
        lidar_np = mask_points_by_range(lidar_np, self.params["preprocess"]["cav_lidar_range"])
        return mask_ego_points(lidar_np)

    def __build_model_input(self, selected_cav_base: dict[str, Any]) -> dict[str, Any]:
        """Build model input without constructing local supervision.

        Parameters
        ----------
        selected_cav_base : dict[str, Any]
            Raw dataset entry for one agent.

        Returns
        -------
        dict[str, Any]
            Preprocessed LiDAR features used by the detector.
        """
        lidar_np = self.__prepare_lidar(selected_cav_base)
        return {"inference_input": build_inference_input(self.pre_processor.preprocess(lidar_np))}

    def get_item_single_car(self, selected_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        lidar_np = self.__prepare_lidar(selected_cav_base)

        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center(
            [selected_cav_base], selected_cav_base["params"]["lidar_pose"]
        )
        # data augmentation
        lidar_np, object_bbx_center, object_bbx_mask = self.augment(lidar_np, object_bbx_center, object_bbx_mask)

        if self.visualize:
            selected_cav_processed.update({"origin_lidar": lidar_np})

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update({"processed_lidar": lidar_dict})

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({"anchor_box": anchor_box})

        selected_cav_processed.update({"object_bbx_center": object_bbx_center, "object_bbx_mask": object_bbx_mask, "object_ids": object_ids})

        # generate targets label
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=anchor_box, mask=object_bbx_mask)
        selected_cav_processed.update({"label_dict": label_dict})

        return selected_cav_processed

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()

        # during training, we return a random cav's data
        if not self.visualize:
            _, selected_cav_base = random.choice(list(base_data_dict.items()))
        else:
            _, selected_cav_base = list(base_data_dict.items())[0]

        selected_cav_processed = self.get_item_single_car(selected_cav_base)
        processed_data_dict.update({"ego": selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, idx, base_data_dict):
        ego_id = -1
        ego_lidar_pose = []

        ego_id, ego_lidar_pose = self.__find_ego_vehicle(base_data_dict)
        visualization_base_data_dict = self.__retrieve_visualization_base_data(idx) if self.visualize else base_data_dict
        _, visualization_ego_lidar_pose = self.__find_ego_vehicle(visualization_base_data_dict)

        if self.payload_handler is not None:
            inference_input = self.__process_with_messages(ego_id, ego_lidar_pose, base_data_dict)
        else:
            inference_input = self.__process_without_messages(ego_id, ego_lidar_pose, base_data_dict)

        local_supervision = self.build_local_supervision(base_data_dict, ego_lidar_pose)
        visualization_context = self.build_visualization_context(
            ego_id,
            visualization_ego_lidar_pose,
            visualization_base_data_dict,
        )
        return self.assemble_inference_sample(inference_input, local_supervision, visualization_context)

    def __collate_processed_lidar(self, cav_content):
        if "inference_input" in cav_content:
            return self.pre_processor.collate_batch([inference_input_to_dict(cav_content["inference_input"])])
        return self.pre_processor.collate_batch([cav_content["processed_lidar"]])

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

        # for late fusion, we also need to stack the lidar for better
        # visualization
        if self.visualize:
            projected_lidar_list = []
            origin_lidar_by_agent = batch.get("ego", {}).get("origin_lidar_by_agent")
            origin_lidar_roles = batch.get("ego", {}).get("origin_lidar_roles", [])
            origin_lidar_agent_ids = batch.get("ego", {}).get("origin_lidar_agent_ids", [])

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content["anchor_box"] is not None:
                output_dict[cav_id].update({"anchor_box": torch.from_numpy(np.array(cav_content["anchor_box"]))})
            if self.visualize:
                if "origin_lidar" in cav_content:
                    transformation_matrix = cav_content["transformation_matrix"]
                    local_lidar = np.array(cav_content["origin_lidar"], copy=True)
                    projected_lidar = np.array(cav_content["origin_lidar"], copy=True)
                    projected_lidar[:, :3] = box_utils.project_points_by_matrix_torch(projected_lidar[:, :3], transformation_matrix)
                    projected_lidar_list.append(projected_lidar)

            # processed lidar dictionary
            processed_lidar_torch_dict = self.__collate_processed_lidar(cav_content)

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = torch.from_numpy(np.array(cav_content["transformation_matrix"])).float()

            output_dict[cav_id].update(
                {
                    "processed_lidar": processed_lidar_torch_dict,
                    "transformation_matrix": transformation_matrix_torch,
                }
            )

            if "object_bbx_center" in cav_content:
                output_dict[cav_id].update(
                    {
                        "object_bbx_center": torch.from_numpy(np.array([cav_content["object_bbx_center"]])),
                        "object_bbx_mask": torch.from_numpy(np.array([cav_content["object_bbx_mask"]])),
                        "label_dict": self.post_processor.collate_batch([cav_content["label_dict"]]),
                        "object_ids": cav_content["object_ids"],
                    }
                )

            if self.visualize:
                if "origin_lidar" in cav_content:
                    origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=[local_lidar]))
                    origin_lidar = torch.from_numpy(origin_lidar)
                    output_dict[cav_id].update({"origin_lidar": origin_lidar, "origin_lidar_local": origin_lidar.clone(), "agent_id": cav_id})

        if self.visualize:
            if origin_lidar_by_agent is not None:
                output_dict["ego"].update(
                    {
                        "origin_lidar": torch.from_numpy(np.vstack(origin_lidar_by_agent)),
                        "origin_lidar_by_agent": [torch.from_numpy(np.array(points)) for points in origin_lidar_by_agent],
                        "origin_lidar_roles": list(origin_lidar_roles),
                        "origin_lidar_agent_ids": list(origin_lidar_agent_ids),
                    }
                )
            elif projected_lidar_list:
                projected_lidar_stack = torch.from_numpy(np.vstack(projected_lidar_list))
                output_dict["ego"].update({"origin_lidar": projected_lidar_stack})

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
        gt_box_tensor = self.post_processor.generate_gt_bbx({"ego": data_dict["ego"]})

        return pred_box_tensor, pred_score, gt_box_tensor
