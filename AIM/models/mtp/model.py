from typing import Dict, Any
import torch
import numpy as np
import os
import pickle as pkl
import copy
from torch_geometric.data import Batch
import torch.nn.functional as F
import sys
from pathlib import Path

from AIM import AIMModel
from AIM.models.mtp.mtp_models.TransfAny_v2_local_coords.TransfAny_v2_local_coords import TransfAny_v2_local_coords

from .learning.learning_src.data_scripts.data_config import config
from .learning.learning_src.data_scripts.generate_csv_utils import get_map_bounding
from .learning.learning_src.data_scripts.preprocess_utils import (
    extract_needed_features,
    normalize_input_features,
    rotation_matrix_with_allign_to_X,
    rotation_matrix_with_allign_to_Y,
    rotation_matrix_back_with_allign_to_X,
    rotation_matrix_back_with_allign_to_Y,
    denormalize_yaw,
    denormalize_coords,
    z_score_denormalize,
    transform_coords,
)
from .learning.learning_src.data_scripts.preprocess_map import preprocess_object_map
from .learning.learning_src.data_scripts.dataset import TransformerCarDataset


class MTP(AIMModel):
    """
    multi-trajectory prediction model
    """

    def __init__(self, **kwargs: Any):
        """
        :param underling_model: Backend model name (default: "TransfAny_v2_local_coords")
        :param model_params: params for Backend model
        :param map_net_xml_path: path to sumo network xml map file
        :param weight: path to weights file
        """

        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._models: Dict[str, Any] = {
            "TransfAny_v2_local_coords": TransfAny_v2_local_coords,
        }

        underling_model = kwargs.get("underling_model", "TransfAny_v2_local_coords")
        model_params = kwargs.get("model_params")
        map_net_xml_path = kwargs.get("map_net_xml_path")

        model_cls = self._models[underling_model]
        weight = kwargs.get("weight")
        y_x_distr_file = kwargs.get("y_x_distr_file")
        y_y_distr_file = kwargs.get("y_y_distr_file")

        underling_model_dir = Path(sys.modules[model_cls.__module__].__file__).parent
        checkpoint = torch.load(os.path.join(underling_model_dir, "weights", weight), map_location=torch.device("cpu"))

        self.model = model_cls(**model_params)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.map = None

        self.map_net_xml_path = map_net_xml_path
        self.process_map(self.map_net_xml_path)

        if config.data_processing.zscore_normalize:
            with open(os.path.join(underling_model_dir, "distr_params", y_x_distr_file), "rb") as f:
                self.y_x_mean, self.y_x_std = pkl.load(f)

            with open(os.path.join(underling_model_dir, "distr_params", y_y_distr_file), "rb") as f:
                self.y_y_mean, self.y_y_std = pkl.load(f)

    def process_map(self, map_net_xml_path: str):
        """
        process and load map data

        :param map_net_xml_path: path to sumo network xml map file
        """
        lane_level_data, lane_object_representations_yaw = preprocess_object_map(net_file_path=map_net_xml_path)

        self.map = lane_level_data[0][0]
        self.map = torch.tensor(self.map)
        self.map_graph = TransformerCarDataset._create_map_graph(self.map, self.map.shape[0] * self.map.shape[1]).to(self.device)

    def predict(self, features: np.ndarray, target_agent_ids=None):
        """
        predict vehicle trajectories

        :param features: ndarray of shape (v, 6) containing [x, y, speed, yaw, start_yaw, last_yaw] for each vehicle, everything in sumo coordinate system
        :param target_agent_ids: list of target agent ids (optional, currently not used)

        :return: predicted trajectories as ndarray of shape (v, pred_len, 2) containing [x, y] coordinates in sumo coordinate system
        """

        map_bounding = get_map_bounding(self.map_net_xml_path)

        extracted_features = extract_needed_features(features[:, :4], features[:, 4:5], features[:, 5:6])
        normalize_input_features(extracted_features, map_bounding)
        x_global = torch.tensor(extracted_features).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            yaw_cur = x_global[..., 3].clone()
            if config.data_processing.normalize_data:
                denormalize_yaw(yaw_cur)

            if config.data_processing.align_initial_direction_to_x:
                rotations_back_current = rotation_matrix_back_with_allign_to_X(yaw_cur).to(self.device)
            else:
                rotations_back_current = rotation_matrix_back_with_allign_to_Y(yaw_cur).to(self.device)

            map_channels = self.map_data.shape[0] * self.map_data.shape[1]
            num_vechs = x_global.shape[1]
            x_data_yaw = x_global.unsqueeze(1)[:, :, :, 3:4] - x_global.unsqueeze(2)[:, :, :, 3:4]
            x_data_coords = x_global.unsqueeze(1)[:, :, :, :2] - x_global.unsqueeze(2)[:, :, :, :2]
            yaw_for_rotation = x_global[:, :, 3:4].clone()

            if config.data_processing.normalize_data:
                denormalize_yaw(yaw_for_rotation)
            if config.data_processing.align_initial_direction_to_x:
                rotation_matrixes = rotation_matrix_with_allign_to_X(yaw_for_rotation)
            else:
                rotation_matrixes = rotation_matrix_with_allign_to_Y(yaw_for_rotation)

            x_data_coords = torch.matmul(rotation_matrixes, x_data_coords.unsqueeze(-1)).squeeze(-1)
            x_data_speed = x_global.unsqueeze(1)[:, :, :, 2:3].repeat(1, x_global.shape[1], 1, 1)
            x_data_skip = x_global.unsqueeze(1)[:, :, :, 4:].repeat(1, x_global.shape[1], 1, 1)
            x_data_yaw_denormed = x_data_yaw.clone()

            if config.data_processing.normalize_data:
                denormalize_yaw(x_data_yaw_denormed)
            x_tensor = torch.cat(
                [x_data_coords, x_data_speed, x_data_yaw, x_data_skip, torch.cos(x_data_yaw_denormed), torch.sin(x_data_yaw_denormed)],
                dim=-1,
            )
            attn_mask = torch.ones((1, num_vechs, num_vechs), dtype=torch.bool).to(self.device)
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.expand(attn_mask.shape[0], attn_mask.shape[2], attn_mask.shape[2], attn_mask.shape[3]).contiguous()

            map_attn_mask = torch.ones((1, map_channels), dtype=torch.bool).to(self.device)
            map_attn_mask = map_attn_mask.unsqueeze(1)
            map_attn_mask = map_attn_mask.expand(map_attn_mask.shape[0], attn_mask.shape[1], map_attn_mask.shape[2]).contiguous()
            map_infos_list = []

            for k in range(x_tensor.shape[1]):
                g = copy.deepcopy(self.map_graph)
                map_infos_list.append(g)

            map_infos_graph_batch = Batch.from_data_list(map_infos_list)
            map_infos_graph_batch = map_infos_graph_batch.to(self.device)
            map_infos_graph_x = map_infos_graph_batch["dot"].x
            old_map_infos_shape = (map_infos_graph_x.shape[0], config.object_map.object_vector_size)
            new_map_infos_shape = (
                x_tensor.shape[0],
                x_tensor.shape[1],
                int(map_infos_graph_x.shape[0] / (x_tensor.shape[0] * x_tensor.shape[1])),
                config.object_map.object_vector_size,
            )
            map_infos_coords = self.map_graph["dot"].x.unsqueeze(0).unsqueeze(1)[..., [0, 1]] - x_global.unsqueeze(2)[..., [0, 1]]

            if config.data_processing.align_initial_direction_to_x:
                map_rotation = rotation_matrix_with_allign_to_X(yaw_cur.unsqueeze(2))
            else:
                map_rotation = rotation_matrix_with_allign_to_Y(yaw_cur.unsqueeze(2))

            map_infos_coords = torch.matmul(map_rotation, map_infos_coords.unsqueeze(-1)).squeeze(-1)
            map_infos_directions = torch.matmul(map_rotation, self.map_graph["dot"].x.unsqueeze(0).unsqueeze(1)[..., [2, 3]].unsqueeze(-1)).squeeze(
                -1
            )
            map_infos_graph_x = torch.cat(
                [
                    map_infos_coords,
                    map_infos_directions,
                    self.map_graph["dot"].x.unsqueeze(0).unsqueeze(1).repeat(1, map_infos_coords.shape[1], 1, 1)[..., 4:5],
                ],
                dim=-1,
            )
            map_infos_graph_batch["dot"].x = map_infos_graph_x.view(old_map_infos_shape)
            movement_logits, dout_coords = self.model(
                x_tensor[..., [0, 1, 2, 4, 5, -2, -1]], map_infos_graph_batch, attn_mask, map_attn_mask, new_map_infos_shape
            )
            dout_coords = dout_coords.reshape(dout_coords.shape[0], dout_coords.shape[1], config.model.pred_len, config.model.predict_vector_size)

            if config.data_processing.normalize_data and config.data_processing.zscore_normalize:
                z_score_denormalize(dout_coords[:, :, :, 0], self.y_x_mean, self.y_x_std)
                z_score_denormalize(dout_coords[:, :, :, 1], self.y_y_mean, self.y_y_std)

            dout_coords = dout_coords.permute(0, 1, 3, 2)  # [b, v, 2, PRED_LEN]
            dout_coords = (rotations_back_current @ dout_coords).permute(0, 1, 3, 2)  # [b, v, PRED_LEN, 2]
            predictions = dout_coords + x_global[:, :, [0, 1]].unsqueeze(2)

            if config.data_processing.normalize_data:
                denormalize_coords(predictions, map_bounding)

            predictions = transform_coords(predictions)
            return F.sigmoid(movement_logits) > 0, predictions
