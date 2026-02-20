import torch
import numpy as np
import os
import pickle as pkl

from AIM import AIMModel
from .mtp_models.model_factory import ModelFactory
from .mtp_models.model_factory_config import FACTORY_YAML_DIR_FIELD, FACTORY_YAML_CLASS_FIELD

from .learning.data_path_config import DATA_PATH, Y_X_DISTR_FILE, Y_Y_DISTR_FILE
from .learning.learning_src.data_scripts.data_config import (
    ALLIGN_INITIAL_DIRECTION_TO_X,
    NORMALIZE_DATA,
    ZSCORE_NORMALIZE,
    PRED_LEN,
    PREDICT_VECTOR_SIZE,
    COLLECT_DATA_RADIUS,
)
from .learning.learning_src.data_scripts.generate_csv_utils import get_map_bounding
from .learning.learning_src.data_scripts.preprocess_utils import (
    extract_needed_features,
    normalize_input_features,
    rotation_matrix_back_with_allign_to_X,
    rotation_matrix_back_with_allign_to_Y,
    denormalize_yaw,
    denormalize_coords,
    z_scrore_denormalize,
    transform_coords,
)
from .learning.learning_src.data_scripts.preprocess_map import preprocess_map
from .learning.learning_src.data_scripts.dataset import MAP_level_info


class MTP(AIMModel):
    def __init__(self, yaml_model_config_path: str, weights: str, map_net_xml_path: str):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = ModelFactory.create_model(yaml_model_config_path)
        model_info = ModelFactory.get_model_info(yaml_model_config_path)
        model_dir = model_info[FACTORY_YAML_DIR_FIELD]
        model_class_name = model_info[FACTORY_YAML_CLASS_FIELD]
        self.is_transformer = "Transf" in model_class_name

        weights_path = os.path.join(model_dir, "weights", weights)

        checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.map = None
        self.map_info_level = MAP_level_info.lane

        self.map_net_xml_path = map_net_xml_path
        self.process_map(self.map_net_xml_path)

        if ZSCORE_NORMALIZE:
            self.load_z_score_params()

    def process_map(self, map_net_xml_path: str):
        part_level_data, lane_level_data, map_level_data = preprocess_map(net_file_path=map_net_xml_path)
        if self.map_info_level == MAP_level_info.parts:
            self.map = part_level_data[0][0]

        if self.map_info_level == MAP_level_info.lane:
            self.map = lane_level_data[0][0]

        elif self.map_info_level == MAP_level_info.map:
            self.map = map_level_data[0][0]

        self.map = torch.tensor(self.map, device=self.device).unsqueeze(0)

    def load_z_score_params(self):
        with open(os.path.join(DATA_PATH, "csv", Y_X_DISTR_FILE), "rb") as f:
            self.y_x_mean, self.y_x_std = pkl.load(f)

        with open(os.path.join(DATA_PATH, "csv", Y_Y_DISTR_FILE), "rb") as f:
            self.y_y_mean, self.y_y_std = pkl.load(f)

    def predict(self, features: np.ndarray, target_agent_ids=None):
        """
        Docstring for predict

        :param map_net_xml_path: path to current net xml map
        :param features: ndarray of shape (v, 6) [x, y, speed, yaw, start_yaw, last_yaw], everything in sumo coordinate system
        :param target_agent_ids:
        """

        num_agents = features.shape[0]
        map_bounding = get_map_bounding(self.map_net_xml_path)

        extracted_features = extract_needed_features(features[:, :4], features[:, 4:5], features[:, 5:6])
        normalize_input_features(extracted_features, map_bounding)
        x_tensor = torch.tensor(extracted_features).unsqueeze(0).float().to(self.device)

        yaw_cur = x_tensor[..., 3].clone()
        if NORMALIZE_DATA:
            denormalize_yaw(yaw_cur)

        if ALLIGN_INITIAL_DIRECTION_TO_X:
            rotations_back_current = rotation_matrix_back_with_allign_to_X(yaw_cur).to(self.device)
        else:
            rotations_back_current = rotation_matrix_back_with_allign_to_Y(yaw_cur).to(self.device)

        if self.is_transformer:
            map_channels = self.map.shape[1]
            num_vechs = x_tensor.shape[1]

            map_attn_mask = torch.ones((1, map_channels, map_channels), dtype=torch.bool).to(self.device)
            attn_mask = torch.ones((1, num_vechs, num_vechs), dtype=torch.bool).to(self.device)

            dout_coords = self.model(x_tensor[:, :, [0, 1, 2, 3, 4, 5]], self.map, attn_mask, map_attn_mask)
            dout_coords = dout_coords.reshape(dout_coords.shape[0], dout_coords.shape[1], PRED_LEN, PREDICT_VECTOR_SIZE)

            if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                z_scrore_denormalize(dout_coords[:, :, :, 0], self.y_x_mean, self.y_x_std)
                z_scrore_denormalize(dout_coords[:, :, :, 1], self.y_y_mean, self.y_y_std)

            dout_coords = dout_coords.permute(0, 1, 3, 2)  # [b, v, 2, PRED_LEN]
            dout_coords = (rotations_back_current @ dout_coords).permute(0, 1, 3, 2)  # [b, v, PRED_LEN, 2]
            predictions = dout_coords + x_tensor[:, :, [0, 1]].unsqueeze(2)

        else:
            edge_index = torch.tensor([[i, j] for i in range(num_agents) for j in range(num_agents)]).T.to(self.device)
            dout_coords = self.model(x_tensor, edge_index)
            dout_coords = dout_coords.reshape(dout_coords.shape[0], dout_coords.shape[1], PRED_LEN, PREDICT_VECTOR_SIZE)

            if NORMALIZE_DATA and ZSCORE_NORMALIZE:
                z_scrore_denormalize(dout_coords[:, :, 0], self.y_x_mean, self.y_x_std)
                z_scrore_denormalize(dout_coords[:, :, 1], self.y_y_mean, self.y_y_std)

            dout_coords = dout_coords.permute(0, 2, 1)  # [v, 2, PRED_LEN]
            dout_coords = torch.bmm(rotations_back_current, dout_coords).permute(0, 2, 1)  # [v, PRED_LEN, 2]
            predictions = dout_coords + x_tensor[:, [0, 1]].unsqueeze(1)

        if NORMALIZE_DATA:
            denormalize_coords(predictions, COLLECT_DATA_RADIUS * map_bounding)

        predictions = transform_coords(predictions)
        return predictions
