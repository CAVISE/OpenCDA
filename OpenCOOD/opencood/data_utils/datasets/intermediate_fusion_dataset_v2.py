"""
Dataset class for 2-stage backbone intermediate fusion
"""

import math
from collections import OrderedDict

import numpy as np
import torch

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, shuffle_points
from opencood.utils.transformation_utils import x1_to_x2
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu


# TODO: У модели fpvrcnn_intermediate_fusion в этом датасете возникает проблема с весами
# TODO: Проверить работу моделей с такми датасетом
# они не правильно расположены
# size mismatch for spconv_block.conv_out.0.weight: copying a param with shape torch.Size([3, 1, 1, 64, 64]) from checkpoint, the shape in current model is torch.Size([64, 3, 1, 1, 64])
# Надо будет переобучить модель и обновить код
class IntermediateFusionDatasetV2(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """

    def __init__(self, params, visualize, train=True, message_handler=None):
        super(IntermediateFusionDatasetV2, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = post_processor.build_postprocessor(params["postprocess"], train)

        self.message_handler = message_handler
        self.module_name = "OpenCOOD.IntermediateFusionDatasetV2"

    @staticmethod
    def __wrap_ndarray(ndarray):
        return {"data": ndarray.tobytes(), "shape": ndarray.shape, "dtype": str(ndarray.dtype)}

    def extract_data(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)
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

                    msg["voxel_num_points"] = {
                        "name": "voxel_num_points",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["processed_features"]["voxel_num_points"]),
                    }

                    msg["voxel_features"] = {
                        "name": "voxel_features",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["processed_features"]["voxel_features"]),
                    }

                    msg["voxel_coords"] = {
                        "name": "voxel_coords",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["processed_features"]["voxel_coords"]),
                    }

                    msg["projected_lidar"] = {
                        "name": "projected_lidar",
                        "label": "LABEL_OPTIONAL",
                        "type": "NDArray",
                        "data": self.__wrap_ndarray(selected_cav_processed["projected_lidar"]),
                    }

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

    def __process_with_messages(self, ego_id, ego_lidar_pose, base_data_dict):
        processed_features = []
        object_stack = []
        object_id_stack = []
        projected_lidar_stack = []

        ego_cav_base = base_data_dict.get(ego_id)
        ego_cav_processed = self.get_item_single_car(ego_cav_base, ego_lidar_pose)

        object_id_stack += ego_cav_processed["object_ids"]
        object_stack.append(ego_cav_processed["object_bbx_center"])
        processed_features.append(ego_cav_processed["processed_features"])
        projected_lidar_stack.append(ego_cav_processed["projected_lidar"])

        if ego_id in self.message_handler.current_message_artery:
            for cav_id, _ in base_data_dict.items():
                if cav_id in self.message_handler.current_message_artery[ego_id]:
                    with self.message_handler.handle_artery_message(ego_id, cav_id, self.module_name) as msg:
                        projected = np.frombuffer(msg["projected_lidar"]["data"], np.dtype(msg["projected_lidar"]["dtype"]))
                        projected = projected.reshape(msg["projected_lidar"]["shape"])

                        if len(projected) > 10:
                            projected_lidar_stack.append(projected)

                            object_id_stack += msg["object_ids"]

                            bbx = np.frombuffer(msg["object_bbx_center"]["data"], np.dtype(msg["object_bbx_center"]["dtype"]))
                            bbx = bbx.reshape(msg["object_bbx_center"]["shape"])
                            object_stack.append(bbx)

                            voxel_num_points = np.frombuffer(msg["voxel_num_points"]["data"], np.dtype(msg["voxel_num_points"]["dtype"]))
                            voxel_num_points = voxel_num_points.reshape(msg["voxel_num_points"]["shape"])

                            voxel_features = np.frombuffer(msg["voxel_features"]["data"], np.dtype(msg["voxel_features"]["dtype"]))
                            voxel_features = voxel_features.reshape(msg["voxel_features"]["shape"])

                            voxel_coords = np.frombuffer(msg["voxel_coords"]["data"], np.dtype(msg["voxel_coords"]["dtype"]))
                            voxel_coords = voxel_coords.reshape(msg["voxel_coords"]["shape"])

                            processed_features.append(
                                {"voxel_num_points": voxel_num_points, "voxel_features": voxel_features, "voxel_coords": voxel_coords}
                            )

        return {
            "processed_features": processed_features,
            "object_stack": object_stack,
            "object_id_stack": object_id_stack,
            "projected_lidar_stack": projected_lidar_stack,
        }

    def __process_without_messages(self, ego_lidar_pose, base_data_dict):
        processed_features = []
        object_stack = []
        object_id_stack = []
        projected_lidar_stack = []

        for cav_id, selected_cav_base in base_data_dict.items():
            dx = selected_cav_base["params"]["lidar_pose"][0] - ego_lidar_pose[0]
            dy = selected_cav_base["params"]["lidar_pose"][1] - ego_lidar_pose[1]
            distance = math.hypot(dx, dy)

            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)

            if len(selected_cav_processed["projected_lidar"]) > 10:
                object_stack.append(selected_cav_processed["object_bbx_center"])
                object_id_stack += selected_cav_processed["object_ids"]
                processed_features.append(selected_cav_processed["processed_features"])

                projected_lidar_stack.append(selected_cav_processed["projected_lidar"])

        return {
            "processed_features": processed_features,
            "object_stack": object_stack,
            "object_id_stack": object_id_stack,
            "projected_lidar_stack": projected_lidar_stack,
        }

    def __getitem__(self, idx):
        # put here to avoid initialization error
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
        object_stack_all = np.vstack(data["object_id_stack"])
        object_stack_all = object_stack_all[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = np.zeros((self.params["postprocess"]["max_num"], 7))
        mask = np.zeros(self.params["postprocess"]["max_num"])
        object_bbx_center[: object_stack_all.shape[0], :] = object_stack_all
        mask[: object_stack_all.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(data["processed_features"])
        merged_feature_dict = self.merge_features_to_dict(data["processed_features"])

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center,  # hwl
            anchors=anchor_box,
            mask=mask,
        )

        # Filter empty boxes
        object_stack_filtered = []
        label_dict_no_coop = []
        for boxes, points in zip(data["object_stack"], data["projected_lidar_stack"]):
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
            object_stack_filtered.append(boxes[cur_mask])
            bbx_center = np.zeros((self.params["postprocess"]["max_num"], 7))
            bbx_mask = np.zeros(self.params["postprocess"]["max_num"])
            bbx_center[: boxes[cur_mask].shape[0], :] = boxes[cur_mask]
            bbx_mask[: boxes[cur_mask].shape[0]] = 1
            label_dict_no_coop.append(
                self.post_processor.generate_label(
                    gt_box_center=bbx_center,  # hwl
                    anchors=anchor_box,
                    mask=bbx_mask,
                )
            )
        label_dict = {"stage1": label_dict_no_coop, "stage2": label_dict}
        processed_data_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": mask,
                "object_ids": [data["object_id_stack"][i] for i in unique_indices],
                "anchor_box": anchor_box,
                "processed_lidar": merged_feature_dict,
                "label_dict": label_dict,
                "cav_num": cav_num,
            }
        )

        processed_data_dict["ego"].update({"origin_lidar": data["projected_lidar_stack"]})
        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
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
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = x1_to_x2(selected_cav_base["params"]["lidar_pose"], ego_pose)

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center([selected_cav_base], ego_pose)

        # filter lidar
        lidar_np = selected_cav_base["lidar_np"]
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np, self.params["preprocess"]["cav_lidar_range"])
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        selected_cav_processed.update(
            {
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                "object_ids": object_ids,
                "projected_lidar": lidar_np,
                "processed_features": processed_lidar,
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
        label_dict_no_coop_list = []

        origin_lidar = []

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
            label_dict_no_coop_list.append(ego_dict["label_dict"]["stage1"])
            label_dict_list.append(ego_dict["label_dict"]["stage2"])

            origin_lidar.append(ego_dict["origin_lidar"])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= 5
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)
        label_dict_no_coop_list_ = [label_dict for label_list in label_dict_no_coop_list for label_dict in label_list]
        for i in range(len(label_dict_no_coop_list_)):
            if isinstance(label_dict_no_coop_list_[i], list):
                print("debug")
        label_no_coop_torch_dict = self.post_processor.collate_batch(label_dict_no_coop_list_)
        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "processed_lidar": processed_lidar_torch_dict,
                "record_len": record_len,
                "label_dict": {"stage1": label_no_coop_torch_dict, "stage2": label_torch_dict},
                "object_ids": object_ids[0],
            }
        )

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

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]["ego"]["anchor_box"] is not None:
            output_dict["ego"].update({"anchor_box": torch.from_numpy(np.array(batch[0]["ego"]["anchor_box"]))})

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
