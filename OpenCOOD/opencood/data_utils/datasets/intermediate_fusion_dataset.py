"""
Dataset class for intermediate fusion
"""

import math
import logging
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from opencood.data_utils.datasets import basedataset
from opencood.models.communication_adapters import (
    InferenceInput,
    IntermediateFusionWirePayload,
    IntermediateMetadata,
    PoseFrameMetadata,
    V2XViTMetadata,
    build_inference_input,
    merge_inference_inputs,
)
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import pcd_to_np, mask_points_by_range, mask_ego_points, shuffle_points, downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2

logger = logging.getLogger("cavise.opencda.OpenCOOD.opencood.data_utils.datasets.intermediate_fusion_dataset")

PAIRWISE_METADATA_MODELS = frozenset(
    {
        "point_pillar_coalign",
        "point_pillar_v2vnet",
        "point_pillar_where2comm",
    }
)
V2X_VIT_METADATA_MODELS = frozenset({"point_pillar_transformer"})


class IntermediateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """

    def __init__(self, params, visualize, train=True, payload_handler=None):
        super(IntermediateFusionDataset, self).__init__(params, visualize, train)

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first = True
        if "proj_first" in params["fusion"]["args"] and not params["fusion"]["args"]["proj_first"]:
            self.proj_first = False

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        if "cur_ego_pose_flag" in params["fusion"]["args"]:
            self.cur_ego_pose_flag = params["fusion"]["args"]["cur_ego_pose_flag"]
        else:
            self.cur_ego_pose_flag = True

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = post_processor.build_postprocessor(params["postprocess"], train)

        self.payload_handler = payload_handler
        self.module_name = "OpenCOOD.IntermediateFusionDataset"
        self.model_name = params["model"]["core_method"]

    def extract_data(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)
        _, ego_lidar_pose = self.__find_ego_vehicle(base_data_dict)

        if self.payload_handler is not None:
            for cav_id, selected_cav_base in base_data_dict.items():
                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)
                payload = self.build_wire_payload(cav_id, selected_cav_base, selected_cav_processed, idx)
                self.payload_handler.set_opencda_payload(cav_id, self.module_name, payload)

    def build_wire_payload(
        self,
        cav_id: str,
        selected_cav_base: dict[str, Any],
        selected_cav_processed: dict[str, Any],
        receive_frame: int,
    ) -> IntermediateFusionWirePayload:
        """
        Build the intermediate-fusion payload sent over V2X.

        Parameters
        ----------
        cav_id : str
            Identifier of the sending CAV or RSU.
        selected_cav_base : dict[str, Any]
            Raw dataset entry for the sending agent.
        selected_cav_processed : dict[str, Any]
            Processed data for the same sending agent.
        receive_frame : int
            Dataset frame in which the payload is sent.

        Returns
        -------
        IntermediateFusionWirePayload
            Typed payload containing only remote model-input data.
        """
        return IntermediateFusionWirePayload(
            inference_input=selected_cav_processed["inference_input"],
            metadata=self.build_model_metadata(cav_id, selected_cav_base, receive_frame),
        )

    def build_model_metadata(
        self,
        cav_id: str,
        selected_cav_base: dict[str, Any],
        receive_frame: int,
    ) -> IntermediateMetadata | None:
        """
        Build only the metadata consumed by the configured model.

        Parameters
        ----------
        cav_id : str
            Identifier of the sending CAV or RSU.
        selected_cav_base : dict[str, Any]
            Local dataset entry for the sending agent.
        receive_frame : int
            Dataset frame in which the model input is received.

        Returns
        -------
        IntermediateMetadata | None
            Model-specific metadata, or ``None`` for models that consume only
            the preprocessed inference input.
        """
        capture_frame = receive_frame - int(selected_cav_base["time_delay"])
        if self.model_name in PAIRWISE_METADATA_MODELS:
            return PoseFrameMetadata(
                lidar_pose=selected_cav_base["params"]["lidar_pose"],
                capture_frame=capture_frame,
            )
        if self.model_name in V2X_VIT_METADATA_MODELS:
            return V2XViTMetadata(
                velocity=float(selected_cav_base["params"]["ego_speed"] / 30),
                agent_type=1 if "rsu" in cav_id else 0,
                capture_frame=capture_frame,
                spatial_correction_matrix=selected_cav_base["params"]["spatial_correction_matrix"],
            )
        return None

    @staticmethod
    def decode_wire_payload(payload: object) -> IntermediateFusionWirePayload:
        """
        Validate an intermediate-fusion payload received over V2X.

        Parameters
        ----------
        payload : object
            Deserialized module payload received over V2X.

        Returns
        -------
        IntermediateFusionWirePayload
            Validated intermediate-fusion payload.

        Raises
        ------
        TypeError
            If the received payload has an unexpected type.
        """
        if not isinstance(payload, IntermediateFusionWirePayload):
            raise TypeError(f"Expected IntermediateFusionWirePayload, got {type(payload).__name__}")
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
        assert len(ego_lidar_pose) > 0

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

    def __pad_spatial_matrix(self, matrix_list):
        matrix_list = np.stack(matrix_list)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(matrix_list), 1, 1))
        return np.concatenate([matrix_list, padding_eye], axis=0)

    def __pad_to_max(self, lst, pad_value):
        return lst + (self.max_cav - len(lst)) * [pad_value]

    def __build_pairwise_transformation(self, metadata: list[PoseFrameMetadata]) -> np.ndarray:
        """
        Build pairwise transformations for the received agents.

        Parameters
        ----------
        metadata : list[PoseFrameMetadata]
            Pose metadata ordered exactly like the received model inputs.

        Returns
        -------
        numpy.ndarray
            Pairwise transformation matrix padded to ``max_cav``.
        """
        pairwise_t_matrix = np.zeros((self.max_cav, self.max_cav, 4, 4))
        if self.proj_first:
            pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix

        for source_index, source_metadata in enumerate(metadata):
            for target_index, target_metadata in enumerate(metadata):
                pairwise_t_matrix[source_index, target_index] = x1_to_x2(
                    source_metadata.lidar_pose,
                    target_metadata.lidar_pose,
                )
        return pairwise_t_matrix

    def __assemble_model_metadata(
        self,
        metadata: list[IntermediateMetadata | None],
        receive_frame: int,
    ) -> dict[str, Any]:
        """
        Derive model tensors from received metadata.

        Parameters
        ----------
        metadata : list[IntermediateMetadata | None]
            Metadata ordered exactly like the available inference inputs.
        receive_frame : int
            Current receiver dataset frame.

        Returns
        -------
        dict[str, Any]
            Model fields derived from metadata.

        Raises
        ------
        TypeError
            If the received metadata does not match the configured model.
        """
        if self.model_name in PAIRWISE_METADATA_MODELS:
            if not all(isinstance(agent_metadata, PoseFrameMetadata) for agent_metadata in metadata):
                raise TypeError(f"{self.model_name} requires PoseFrameMetadata for every agent")
            pose_metadata = [agent_metadata for agent_metadata in metadata if isinstance(agent_metadata, PoseFrameMetadata)]
            return {"pairwise_t_matrix": self.__build_pairwise_transformation(pose_metadata)}

        if self.model_name in V2X_VIT_METADATA_MODELS:
            if not all(isinstance(agent_metadata, V2XViTMetadata) for agent_metadata in metadata):
                raise TypeError(f"{self.model_name} requires V2XViTMetadata for every agent")
            v2x_metadata = [agent_metadata for agent_metadata in metadata if isinstance(agent_metadata, V2XViTMetadata)]
            return {
                "velocity": self.__pad_to_max([agent_metadata.velocity for agent_metadata in v2x_metadata], 0.0),
                "time_delay": self.__pad_to_max(
                    [float(receive_frame - agent_metadata.capture_frame) for agent_metadata in v2x_metadata],
                    0.0,
                ),
                "infra": self.__pad_to_max([agent_metadata.agent_type for agent_metadata in v2x_metadata], 0.0),
                "spatial_correction_matrix": self.__pad_spatial_matrix([agent_metadata.spatial_correction_matrix for agent_metadata in v2x_metadata]),
            }

        if any(agent_metadata is not None for agent_metadata in metadata):
            raise TypeError(f"{self.model_name} does not consume wire metadata")
        return {}

    @staticmethod
    def __build_model_data():
        return {
            "inference_inputs": [],
            "metadata": [],
        }

    def __append_processed_model_data(self, data, cav_id, cav_base, cav_processed, receive_frame):
        data["inference_inputs"].append(cav_processed["inference_input"])
        data["metadata"].append(self.build_model_metadata(cav_id, cav_base, receive_frame))

    @staticmethod
    def __append_message_model_data(data: dict[str, Any], payload: IntermediateFusionWirePayload) -> None:
        data["inference_inputs"].append(payload.inference_input)
        data["metadata"].append(payload.metadata)

    def __agent_in_communication_range(self, cav_base, ego_lidar_pose):
        dx = cav_base["params"]["lidar_pose"][0] - ego_lidar_pose[0]
        dy = cav_base["params"]["lidar_pose"][1] - ego_lidar_pose[1]
        return math.hypot(dx, dy) <= opencood.data_utils.datasets.COM_RANGE

    def __project_lidar_for_visualization(self, cav_base):
        transformation_matrix = cav_base["params"]["transformation_matrix"]
        lidar_np = np.array(cav_base["lidar_np"], copy=True)
        lidar_np = mask_ego_points(lidar_np)
        if self.proj_first:
            lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        return mask_points_by_range(lidar_np, self.params["preprocess"]["cav_lidar_range"])

    def build_visualization_context(self, ego_id: str, base_data_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Build visualization data from the local scene snapshot.

        Parameters
        ----------
        ego_id : str
            Identifier of the receiving ego agent.
        base_data_dict : dict[str, Any]
            Local dataset snapshot, independent of delivered messages.

        Returns
        -------
        dict[str, Any]
            Per-agent projected point clouds and their local identities.
        """
        if not self.visualize:
            return {
                "projected_lidar_stack": [],
                "projected_lidar_roles": [],
                "projected_lidar_agent_ids": [],
            }

        projected_lidar_stack = []
        projected_lidar_roles = []
        projected_lidar_agent_ids = []

        for cav_id in base_data_dict:
            projected_lidar_stack.append(self.__project_lidar_for_visualization(base_data_dict[cav_id]))
            projected_lidar_roles.append("ego" if cav_id == ego_id else "other")
            projected_lidar_agent_ids.append(cav_id)

        return {
            "projected_lidar_stack": projected_lidar_stack,
            "projected_lidar_roles": projected_lidar_roles,
            "projected_lidar_agent_ids": projected_lidar_agent_ids,
        }

    def __process_with_messages(self, ego_id, ego_lidar_pose, base_data_dict, receive_frame):
        data = self.__build_model_data()
        ego_cav_base = base_data_dict.get(ego_id)
        ego_cav_processed = self.get_item_single_car(ego_cav_base, ego_lidar_pose)
        self.__append_processed_model_data(data, ego_id, ego_cav_base, ego_cav_processed, receive_frame)

        if ego_id in self.payload_handler.current_artery_payload:
            for cav_id, _ in base_data_dict.items():
                raw_payload = self.payload_handler.get_artery_payload(ego_id, cav_id, self.module_name)
                if raw_payload is None:
                    continue
                payload = self.decode_wire_payload(raw_payload)
                self.__append_message_model_data(data, payload)

        return data

    def __process_without_messages(self, ego_id, ego_lidar_pose, base_data_dict, receive_frame):
        data = self.__build_model_data()
        for cav_id, selected_cav_base in base_data_dict.items():
            if not self.__agent_in_communication_range(selected_cav_base, ego_lidar_pose):
                continue

            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)
            self.__append_processed_model_data(data, cav_id, selected_cav_base, selected_cav_processed, receive_frame)

        return data

    def build_local_supervision(self, base_data_dict: dict[str, Any], ego_lidar_pose: list[float]) -> dict[str, Any]:
        """
        Build ground truth and training targets from the local scene.

        Parameters
        ----------
        base_data_dict : dict[str, Any]
            Complete local dataset snapshot for the current scene.
        ego_lidar_pose : list[float]
            Current ego LiDAR pose in world coordinates.

        Returns
        -------
        dict[str, Any]
            Ground-truth boxes, object identifiers, anchors, and labels.
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
        inference_input: dict[str, Any],
        local_supervision: dict[str, Any],
        visualization_context: dict[str, Any],
        receive_frame: int,
    ) -> OrderedDict[str, dict[str, Any]]:
        """
        Assemble model input with local supervision and visualization.

        Parameters
        ----------
        inference_input : dict[str, Any]
            Ego features plus features from messages that were delivered.
        local_supervision : dict[str, Any]
            Ground truth produced solely from the local dataset snapshot.
        visualization_context : dict[str, Any]
            Visualization data produced solely from the local snapshot.
        receive_frame : int
            Current receiver dataset frame.

        Returns
        -------
        collections.OrderedDict
            Dataset sample consumed by the collate and inference pipeline.

        Raises
        ------
        ValueError
            If model inputs and metadata entries are not aligned.
        """
        typed_inputs: list[InferenceInput] = inference_input["inference_inputs"]
        merged_feature_dict = merge_inference_inputs(typed_inputs)
        metadata: list[IntermediateMetadata | None] = inference_input["metadata"]
        if len(metadata) != len(typed_inputs):
            raise ValueError("Every inference input must have a matching metadata entry")

        ego_sample = {
            **local_supervision,
            "processed_lidar": merged_feature_dict,
            "cav_num": len(typed_inputs),
            **self.__assemble_model_metadata(metadata, receive_frame),
        }
        if self.visualize:
            ego_sample.update(
                {
                    "origin_lidar": np.vstack(visualization_context["projected_lidar_stack"]),
                    "origin_lidar_by_agent": visualization_context["projected_lidar_stack"],
                    "origin_lidar_roles": visualization_context["projected_lidar_roles"],
                    "origin_lidar_agent_ids": visualization_context["projected_lidar_agent_ids"],
                }
            )

        return OrderedDict({"ego": ego_sample})

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)
        ego_id, ego_lidar_pose = self.__find_ego_vehicle(base_data_dict)
        visualization_base_data_dict = self.__retrieve_visualization_base_data(idx) if self.visualize else base_data_dict

        if self.payload_handler is not None:
            inference_input = self.__process_with_messages(ego_id, ego_lidar_pose, base_data_dict, idx)
        else:
            inference_input = self.__process_without_messages(ego_id, ego_lidar_pose, base_data_dict, idx)

        local_supervision = self.build_local_supervision(base_data_dict, ego_lidar_pose)
        visualization_context = self.build_visualization_context(ego_id, visualization_base_data_dict)
        return self.assemble_inference_sample(
            inference_input,
            local_supervision,
            visualization_context,
            idx,
        )

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project and preprocess one agent's LiDAR input.

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
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = selected_cav_base["params"]["transformation_matrix"]

        # filter lidar
        lidar_np = selected_cav_base["lidar_np"]
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np, self.params["preprocess"]["cav_lidar_range"])
        inference_input = build_inference_input(self.pre_processor.preprocess(lidar_np))

        selected_cav_processed.update(
            {
                "projected_lidar": lidar_np,
                "inference_input": inference_input,
            }
        )

        return selected_cav_processed

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {"ego": {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []

        # used for PriorEncoding for models
        velocity = []
        time_delay = []
        infra = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            object_ids.append(ego_dict["object_ids"])

            processed_lidar_list.append(ego_dict["processed_lidar"])
            record_len.append(ego_dict["cav_num"])
            label_dict_list.append(ego_dict["label_dict"])

            if self.model_name in PAIRWISE_METADATA_MODELS:
                pairwise_t_matrix_list.append(ego_dict["pairwise_t_matrix"])
            elif self.model_name in V2X_VIT_METADATA_MODELS:
                velocity.append(ego_dict["velocity"])
                time_delay.append(ego_dict["time_delay"])
                infra.append(ego_dict["infra"])
                spatial_correction_matrix_list.append(ego_dict["spatial_correction_matrix"])

            if self.visualize:
                origin_lidar.append(ego_dict["origin_lidar"])
        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # Merge per-sample agent inputs before preprocessor-specific collation.
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "processed_lidar": processed_lidar_torch_dict,
                "record_len": record_len,
                "label_dict": label_torch_dict,
                "object_ids": object_ids[0],
            }
        )

        if self.model_name in PAIRWISE_METADATA_MODELS:
            output_dict["ego"]["pairwise_t_matrix"] = torch.from_numpy(np.array(pairwise_t_matrix_list))
        elif self.model_name in V2X_VIT_METADATA_MODELS:
            velocity_tensor = torch.from_numpy(np.array(velocity))
            time_delay_tensor = torch.from_numpy(np.array(time_delay))
            infra_tensor = torch.from_numpy(np.array(infra))
            output_dict["ego"].update(
                {
                    "prior_encoding": torch.stack([velocity_tensor, time_delay_tensor, infra_tensor], dim=-1).float(),
                    "spatial_correction_matrix": torch.from_numpy(np.array(spatial_correction_matrix_list)),
                }
            )

        if self.visualize:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict["ego"].update({"origin_lidar": origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]["ego"]["anchor_box"] is not None:
            output_dict["ego"].update({"anchor_box": torch.from_numpy(np.array(batch[0]["ego"]["anchor_box"]))})

        if self.visualize and "origin_lidar_by_agent" in batch[0]["ego"]:
            output_dict["ego"].update(
                {
                    "origin_lidar_by_agent": [torch.from_numpy(np.array(points)) for points in batch[0]["ego"]["origin_lidar_by_agent"]],
                    "origin_lidar_roles": list(batch[0]["ego"]["origin_lidar_roles"]),
                    "origin_lidar_agent_ids": list(batch[0]["ego"]["origin_lidar_agent_ids"]),
                }
            )

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
        output_dict["ego"].update({"transformation_matrix": transformation_matrix_torch})

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
