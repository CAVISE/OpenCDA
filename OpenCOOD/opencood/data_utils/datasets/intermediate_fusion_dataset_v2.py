"""
Dataset class for 2-stage backbone intermediate fusion
"""

import math
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from opencood.data_utils.datasets import basedataset
from opencood.models.communication_adapters import PoseFrameMetadata
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import pcd_to_np, mask_points_by_range, mask_ego_points, shuffle_points
from opencood.utils.transformation_utils import x1_to_x2
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu


# TODO: The fpvrcnn_intermediate_fusion model has an issue with weights on this dataset
# TODO: Verify the model behavior with this dataset
# The weights appear to be incorrectly arranged
# size mismatch for spconv_block.conv_out.0.weight:
# copying a param with shape torch.Size([3, 1, 1, 64, 64]) from checkpoint,
# while the current model expects torch.Size([64, 3, 1, 1, 64])
# The model will need to be retrained and the code updated
class IntermediateFusionDatasetV2(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """

    def __init__(self, params, visualize, train=True, payload_handler=None):
        super(IntermediateFusionDatasetV2, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = post_processor.build_postprocessor(params["postprocess"], train)

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        if "cur_ego_pose_flag" in params["fusion"]["args"]:
            self.cur_ego_pose_flag = params["fusion"]["args"]["cur_ego_pose_flag"]
        else:
            self.cur_ego_pose_flag = True

        self.payload_handler = payload_handler
        self.module_name = "OpenCOOD.IntermediateFusionDatasetV2"

    def extract_data(
        self,
        idx: int,
        agent_payload_builder: Callable[[dict[str, Any], PoseFrameMetadata], object] | None = None,
    ) -> None:
        """Build model-specific payloads from local agent observations.

        Parameters
        ----------
        idx : int
            Dataset frame to publish.
        agent_payload_builder : Callable[[dict[str, Any], PoseFrameMetadata], object] | None
            Model adapter callback that converts one local agent observation
            and its metadata into a typed wire payload.

        Raises
        ------
        NotImplementedError
            If the configured model has no communication adapter for this
            dataset contract.
        """
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)

        if self.payload_handler is not None:
            if agent_payload_builder is None:
                raise NotImplementedError("IntermediateFusionDatasetV2 requires a model-specific communication adapter")
            for cav_id, selected_cav_base in base_data_dict.items():
                sender_pose = selected_cav_base["params"]["lidar_pose"]
                selected_cav_processed = self.build_agent_model_input(selected_cav_base, sender_pose)
                metadata = PoseFrameMetadata(
                    lidar_pose=tuple(float(value) for value in sender_pose),
                    capture_frame=int(idx - selected_cav_base["time_delay"]),
                )
                payload = agent_payload_builder(selected_cav_processed, metadata)
                self.payload_handler.set_opencda_payload(cav_id, self.module_name, payload)

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

    def __project_lidar_for_visualization(self, cav_base, ego_lidar_pose):
        transformation_matrix = x1_to_x2(cav_base["params"]["lidar_pose"], ego_lidar_pose)
        lidar_np = np.array(cav_base["lidar_np"], copy=True)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        return mask_points_by_range(lidar_np, self.params["preprocess"]["cav_lidar_range"])

    def build_visualization_context(self, ego_id, ego_lidar_pose, base_data_dict):
        """Build local visualization data independently of wire delivery.

        Parameters
        ----------
        ego_id : str
            Identifier of the receiving agent.
        ego_lidar_pose : list[float]
            Receiver pose used to project visualization point clouds.
        base_data_dict : dict[str, Any]
            Complete local scene snapshot.

        Returns
        -------
        dict[str, Any]
            Per-agent point clouds, roles, and identifiers used only by
            visualization code.
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
            projected_lidar_stack.append(self.__project_lidar_for_visualization(base_data_dict[cav_id], ego_lidar_pose))
            projected_lidar_roles.append("ego" if cav_id == ego_id else "other")
            projected_lidar_agent_ids.append(cav_id)

        return {
            "projected_lidar_stack": projected_lidar_stack,
            "projected_lidar_roles": projected_lidar_roles,
            "projected_lidar_agent_ids": projected_lidar_agent_ids,
        }

    def __process_with_messages(self, ego_id, ego_lidar_pose, base_data_dict, visualization_ego_lidar_pose, visualization_base_data_dict):
        processed_features = []
        model_lidar_stack = []
        remote_agent_outputs = []

        ego_cav_base = base_data_dict.get(ego_id)
        ego_cav_processed = self.build_agent_model_input(ego_cav_base, ego_lidar_pose)

        processed_features.append(ego_cav_processed["processed_features"])
        model_lidar_stack.append(ego_cav_processed["projected_lidar"])

        if self.communication_adapter is None:
            raise RuntimeError("IntermediateFusionDatasetV2 requires an attached communication adapter")

        if ego_id in self.payload_handler.current_artery_payload:
            for cav_id, _ in base_data_dict.items():
                if cav_id == ego_id:
                    continue
                raw_payload = self.payload_handler.get_artery_payload(ego_id, cav_id, self.module_name)
                if raw_payload is None:
                    continue
                remote_agent_outputs.append(self.communication_adapter.decode_received_payload(raw_payload, ego_lidar_pose))

        return {
            "processed_features": processed_features,
            "model_lidar_stack": model_lidar_stack,
            "remote_agent_outputs": remote_agent_outputs,
            **self.build_visualization_context(ego_id, visualization_ego_lidar_pose, visualization_base_data_dict),
        }

    def build_local_supervision(self, base_data_dict: dict[str, Any], ego_lidar_pose: list[float]) -> dict[str, Any]:
        """Build two-stage ground truth from the complete local scene.

        Parameters
        ----------
        base_data_dict : dict[str, Any]
            Complete local dataset snapshot, independent of packet delivery.
        ego_lidar_pose : list[float]
            Current receiver LiDAR pose.

        Returns
        -------
        dict[str, Any]
            Ground-truth boxes, masks, identifiers, and anchors, plus
            stage-two targets when building a training sample.
        """
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center(
            list(base_data_dict.values()),
            ego_lidar_pose,
        )
        anchor_box = self.post_processor.generate_anchor_box()
        supervision: dict[str, Any] = {
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
            "object_ids": object_ids,
            "anchor_box": anchor_box,
        }
        if self.payload_handler is None:
            supervision["stage2_label"] = self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=object_bbx_mask,
            )
        return supervision

    def __process_without_messages(self, ego_id, ego_lidar_pose, base_data_dict, visualization_ego_lidar_pose, visualization_base_data_dict):
        processed_features = []
        object_stack = []
        model_lidar_stack = []
        agent_to_ego = []

        for cav_id, selected_cav_base in base_data_dict.items():
            dx = selected_cav_base["params"]["lidar_pose"][0] - ego_lidar_pose[0]
            dy = selected_cav_base["params"]["lidar_pose"][1] - ego_lidar_pose[1]
            distance = math.hypot(dx, dy)

            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            agent_pose = selected_cav_base["params"]["lidar_pose"]
            selected_cav_processed = self.get_item_single_car(selected_cav_base, agent_pose)

            if len(selected_cav_processed["projected_lidar"]) > 10:
                object_stack.append(selected_cav_processed["object_bbx_center"])
                processed_features.append(selected_cav_processed["processed_features"])
                model_lidar_stack.append(selected_cav_processed["projected_lidar"])
                agent_to_ego.append(x1_to_x2(agent_pose, ego_lidar_pose))

        return {
            "processed_features": processed_features,
            "object_stack": object_stack,
            "model_lidar_stack": model_lidar_stack,
            "agent_to_ego": agent_to_ego,
            **self.build_visualization_context(ego_id, visualization_ego_lidar_pose, visualization_base_data_dict),
        }

    def __getitem__(self, idx):
        # put here to avoid initialization error
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)

        processed_data_dict = OrderedDict()
        processed_data_dict["ego"] = {}

        ego_id, ego_lidar_pose = self.__find_ego_vehicle(base_data_dict)
        visualization_base_data_dict = self.__retrieve_visualization_base_data(idx) if self.visualize else base_data_dict
        _, visualization_ego_lidar_pose = self.__find_ego_vehicle(visualization_base_data_dict)

        if self.payload_handler is not None:
            data = self.__process_with_messages(
                ego_id,
                ego_lidar_pose,
                base_data_dict,
                visualization_ego_lidar_pose,
                visualization_base_data_dict,
            )
        else:
            data = self.__process_without_messages(
                ego_id,
                ego_lidar_pose,
                base_data_dict,
                visualization_ego_lidar_pose,
                visualization_base_data_dict,
            )

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(data["processed_features"])
        merged_feature_dict = self.merge_features_to_dict(data["processed_features"])

        local_supervision = self.build_local_supervision(base_data_dict, ego_lidar_pose)
        anchor_box = local_supervision["anchor_box"]

        processed_data_dict["ego"].update(
            {
                "object_bbx_center": local_supervision["object_bbx_center"],
                "object_bbx_mask": local_supervision["object_bbx_mask"],
                "object_ids": local_supervision["object_ids"],
                "anchor_box": anchor_box,
                "processed_lidar": merged_feature_dict,
                "cav_num": cav_num,
            }
        )

        if "stage2_label" in local_supervision:
            label_dict_no_coop = []
            for boxes, points in zip(data["object_stack"], data["model_lidar_stack"]):
                point_indices = points_in_boxes_cpu(points[:, :3], boxes[:, [0, 1, 2, 5, 4, 3, 6]])
                cur_mask = point_indices.sum(axis=1) > 0
                if cur_mask.sum() == 0:
                    label_dict_no_coop.append(
                        {
                            "pos_equal_one": np.zeros((*anchor_box.shape[:2], self.post_processor.anchor_num)),
                            "neg_equal_one": np.ones((*anchor_box.shape[:2], self.post_processor.anchor_num)),
                            "targets": np.zeros((*anchor_box.shape[:2], self.post_processor.anchor_num * 7)),
                        }
                    )
                    continue
                bbx_center = np.zeros((self.params["postprocess"]["max_num"], 7))
                bbx_mask = np.zeros(self.params["postprocess"]["max_num"])
                bbx_center[: boxes[cur_mask].shape[0], :] = boxes[cur_mask]
                bbx_mask[: boxes[cur_mask].shape[0]] = 1
                label_dict_no_coop.append(
                    self.post_processor.generate_label(
                        gt_box_center=bbx_center,
                        anchors=anchor_box,
                        mask=bbx_mask,
                    )
                )
            processed_data_dict["ego"]["label_dict"] = {
                "stage1": label_dict_no_coop,
                "stage2": local_supervision["stage2_label"],
            }

        processed_data_dict["ego"].update(
            {
                "origin_lidar": data["model_lidar_stack"],
                "origin_lidar_by_agent": data["projected_lidar_stack"],
                "origin_lidar_roles": data["projected_lidar_roles"],
                "origin_lidar_agent_ids": data["projected_lidar_agent_ids"],
            }
        )
        if "remote_agent_outputs" in data:
            processed_data_dict["ego"]["remote_agent_outputs"] = data["remote_agent_outputs"]
        if "agent_to_ego" in data:
            processed_data_dict["ego"]["agent_to_ego"] = data["agent_to_ego"]
        return processed_data_dict

    def build_agent_model_input(self, selected_cav_base, target_pose):
        """Build one agent's local two-stage input without supervision data.

        Parameters
        ----------
        selected_cav_base : dict
            Raw and metadata fields for one agent.
        target_pose : list
            Pose of the coordinate frame used for this model input.

        Returns
        -------
        dict
            Projected raw points and their voxelized representation.
        """
        transformation_matrix = x1_to_x2(selected_cav_base["params"]["lidar_pose"], target_pose)

        lidar_np = selected_cav_base["lidar_np"]
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np, self.params["preprocess"]["cav_lidar_range"])
        return {
            "projected_lidar": lidar_np,
            "processed_features": self.pre_processor.preprocess(lidar_np),
        }

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """Build one local training agent with its stage-one ground truth.

        Parameters
        ----------
        selected_cav_base : dict
            Raw and metadata fields for one agent.
        ego_pose : list
            Pose of the coordinate frame used for model input and targets.

        Returns
        -------
        dict
            Agent model input plus local stage-one ground-truth boxes.
        """
        selected_cav_processed = self.build_agent_model_input(selected_cav_base, ego_pose)
        object_bbx_center, object_bbx_mask, _ = self.post_processor.generate_object_center([selected_cav_base], ego_pose)
        selected_cav_processed["object_bbx_center"] = object_bbx_center[object_bbx_mask == 1]
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
        label_dict_no_coop_list = []
        has_labels = "label_dict" in batch[0]["ego"]

        origin_lidar = []
        agent_to_ego = []

        # added by yys, fpvrcnn needs anchors for
        # first stage proposal generation
        if batch[0]["ego"]["anchor_box"] is not None:
            output_dict["ego"].update({"anchor_box": torch.from_numpy(np.array(batch[0]["ego"]["anchor_box"]))})

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            object_ids.append(ego_dict["object_ids"])

            processed_lidar_list.append(ego_dict["processed_lidar"])
            record_len.append(ego_dict["cav_num"])
            if has_labels:
                label_dict_no_coop_list.append(ego_dict["label_dict"]["stage1"])
                label_dict_list.append(ego_dict["label_dict"]["stage2"])

            origin_lidar.append(ego_dict["origin_lidar"])
            if "agent_to_ego" in ego_dict:
                agent_to_ego.extend(ego_dict["agent_to_ego"])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= 5
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "processed_lidar": processed_lidar_torch_dict,
                "record_len": record_len,
                "object_ids": object_ids[0],
            }
        )
        if has_labels:
            label_torch_dict = self.post_processor.collate_batch(label_dict_list)
            label_dict_no_coop = [label_dict for label_list in label_dict_no_coop_list for label_dict in label_list]
            output_dict["ego"]["label_dict"] = {
                "stage1": self.post_processor.collate_batch(label_dict_no_coop),
                "stage2": label_torch_dict,
            }

        coords = []
        idx = 0
        for b in range(len(batch)):
            for points in origin_lidar[b]:
                assert len(points) != 0
                coor_pad = np.pad(points, ((0, 0), (1, 0)), mode="constant", constant_values=idx)
                coords.append(coor_pad)
                idx += 1
        origin_lidar = np.concatenate(coords, axis=0)

        origin_lidar = torch.from_numpy(origin_lidar)
        output_dict["ego"].update({"origin_lidar": origin_lidar})

        if agent_to_ego:
            output_dict["ego"]["agent_to_ego"] = torch.from_numpy(np.asarray(agent_to_ego)).float()

        if "remote_agent_outputs" in batch[0]["ego"]:
            if len(batch) != 1:
                raise ValueError("Distributed two-stage inference requires batch size one")
            output_dict["ego"]["remote_agent_outputs"] = [
                {field: torch.from_numpy(np.asarray(value)) for field, value in agent_output.items()}
                for agent_output in batch[0]["ego"]["remote_agent_outputs"]
            ]

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]["ego"]["anchor_box"] is not None:
            output_dict["ego"].update({"anchor_box": torch.from_numpy(np.array(batch[0]["ego"]["anchor_box"]))})

        if "origin_lidar_by_agent" in batch[0]["ego"]:
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

    def visualize_result(self, pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        # we need to convert the pcd from [n, 5] -> [n, 4]
        pcd = pcd[:, 1:]
        # visualize the model output
        self.post_processor.visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=dataset)
