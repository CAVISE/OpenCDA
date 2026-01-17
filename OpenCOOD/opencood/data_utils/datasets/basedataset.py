"""
Base dataset class for cooperative perception.

This module provides the foundational dataset class for all types of fusion
approaches (early, intermediate, and late fusion) in cooperative autonomous
driving, handling data loading, preprocessing, and augmentation.
"""

import os
import math
import logging
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


logger = logging.getLogger("cavise.OpenCOOD.opencood.data_utils.datasets.basedataset")


class BaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion.

    Mainly used to initialize the database and associate the __get_item__ index
    with the correct timestamp and scenario.

    Parameters
    ----------
    params : Dict[str, Any]
        The dictionary contains all parameters for training/testing.
    visualize : bool
        If set to true, the raw point cloud will be saved in the memory
        for visualization.
    train : bool, optional
        Whether the dataset is used for training. Default is True.

    Attributes
    ----------
    params : Dict[str, Any]
        Configuration parameters for the dataset.
    visualize : bool
        Flag for visualization mode.
    train : bool
        Training mode indicator.
    pre_processor : Optional[Any]
        Preprocessor instance for point cloud processing.
    post_processor : Optional[Any]
        Postprocessor instance for output processing.
    data_augmentor : DataAugmentor
        Data augmentation handler.
    max_cav : int
        Maximum number of connected autonomous vehicles.
    scenario_database : OrderedDict
        Database containing all scenario data.
    len_record : List[int]
        Cumulative length record for indexing.
    """

    def __init__(self, params: Dict[str, Any], visualize: bool, train: bool = True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor: Optional[Any] = None
        self.post_processor: Optional[Any] = None
        self.data_augmentor = DataAugmentor(params["data_augment"], train)

        # if the training/testing include noisy setting
        if "wild_setting" in params:
            self.seed = params["wild_setting"]["seed"]
            # whether to add time delay
            self.async_flag = params["wild_setting"]["async"]
            self.async_mode = "sim" if "async_mode" not in params["wild_setting"] else params["wild_setting"]["async_mode"]
            self.async_overhead = params["wild_setting"]["async_overhead"]

            # localization error
            self.loc_err_flag = params["wild_setting"]["loc_err"]
            self.xyz_noise_std = params["wild_setting"]["xyz_std"]
            self.ryp_noise_std = params["wild_setting"]["ryp_std"]

            # transmission data size
            self.data_size = params["wild_setting"]["data_size"] if "data_size" in params["wild_setting"] else 0
            self.transmission_speed = params["wild_setting"]["transmission_speed"] if "transmission_speed" in params["wild_setting"] else 27
            self.backbone_delay = params["wild_setting"]["backbone_delay"] if "backbone_delay" in params["wild_setting"] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = "sim"
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb (Megabits)
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        if self.train:
            root_dir = params["root_dir"]
        else:
            root_dir = params["validate_dir"]

        if "train_params" not in params or "max_cav" not in params["train_params"]:
            self.max_cav = 7
        else:
            self.max_cav = params["train_params"]["max_cav"]

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))])
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for i, scenario_folder in enumerate(scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder) if os.path.isdir(os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if "rsu" in cav_list[0]:
                cav_list = cav_list[1:] + [cav_list[0]]

            # loop over all CAV data
            for j, cav_id in enumerate(cav_list):
                if j > self.max_cav - 1:
                    logger.warning(f"Too many CAVs and RSUs: {len(cav_list)}")
                    logger.warning(f"Maximum is {self.max_cav}")
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                yaml_files = sorted([os.path.join(cav_path, x) for x in os.listdir(cav_path) if x.endswith(".yaml") and "additional" not in x])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = OrderedDict()

                    yaml_file = os.path.join(cav_path, timestamp + ".yaml")
                    lidar_file = os.path.join(cav_path, timestamp + ".pcd")
                    camera_files = self.load_camera_files(cav_path, timestamp)

                    self.scenario_database[i][cav_id][timestamp]["yaml"] = yaml_file
                    self.scenario_database[i][cav_id][timestamp]["lidar"] = lidar_file
                    self.scenario_database[i][cav_id][timestamp]["camera0"] = camera_files
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]["ego"] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]["ego"] = False

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx: int):
        """
        Abstract method, needs to be define by the children class.

        Parameters
        ----------
        idx : int
            Index of the data sample.

        Returns
        -------
        Dict[str, Any]
            Data dictionary.
        """
        # TODO: Implement this function
        raise NotImplementedError

    def retrieve_base_data(self, idx: int, cur_ego_pose_flag: bool = True) -> Dict[str, Any]:
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.
        cur_ego_pose_flag : bool, optional
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix. If set to false, meaning when other cavs
            project their LiDAR point cloud to ego, they are projecting to
            past ego pose. Default is True.

        Returns
        -------
        Dict[str, Any]
            The dictionary contains loaded yaml params and lidar data for
            each cav.
            Structure: {cav_id: {'ego': bool, 'time_delay': int,
                                'params': dict, 'lidar_np': NDArray}}
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
        # calculate distance to ego for each cav
        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]["ego"] = cav_content["ego"]

            # calculate delay for this vehicle
            timestamp_delay = self.time_delay_calculation(cav_content["ego"])

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database, timestamp_index_delay)
            # add time delay to vehicle parameters
            data[cav_id]["time_delay"] = timestamp_delay
            # load the corresponding data into the dictionary
            data[cav_id]["params"] = self.reform_param(cav_content, ego_cav_content, timestamp_key, timestamp_key_delay, cur_ego_pose_flag)
            data[cav_id]["lidar_np"] = pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]["lidar"])
        return data

    @staticmethod
    def extract_timestamps(yaml_files: List[str]) -> List[str]:
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : List[str]
            The full path of all yaml files of ego vehicle.

        Returns
        -------
        List[str]
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split("/")[-1]

            timestamp = res.replace(".yaml", "")
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database: Dict[str, Any], timestamp_index: int) -> str:
        """
        Given the timestamp index, return the correct timestamp key.

        For example: 2 --> '000078'.

        Parameters
        ----------
        scenario_database : Dict[str, Any]
            The dictionary contains all contents in the current scenario.
        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    def calc_dist_to_ego(self, scenario_database: Dict[str, Any], timestamp_key: str) -> Dict[str, Any]:
        """
        Calculate the distance of each CAV to the ego vehicle.

        Parameters
        ----------
        scenario_database : Dict[str, Any]
            Dictionary containing scenario data for all CAVs.
        timestamp_key : str
            Timestamp key to access the specific data.

        Returns
        -------
        Dict[str, Any]
            The ego vehicle's content with updated distance information.

        Raises
        ------
        ValueError
            If no ego vehicle is found in the scenario.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content["ego"]:
                ego_cav_content = cav_content
                ego_lidar_pose = load_yaml(cav_content[timestamp_key]["yaml"])["lidar_pose"]
                break

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = load_yaml(cav_content[timestamp_key]["yaml"])["lidar_pose"]

            dx = cur_lidar_pose[0] - ego_lidar_pose[0]
            dy = cur_lidar_pose[1] - ego_lidar_pose[1]
            distance = math.hypot(dx, dy)

            cav_content["distance_to_ego"] = distance
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content

    def time_delay_calculation(self, ego_flag: bool) -> int:
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : bool
            Whether the current CAV is the ego vehicle.

        Returns
        -------
        int
            The time delay in 100ms units.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == "real":
            # in the real mode, time delay = systematic async time + data
            # transmission time + backbone computation time
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == "sim":
            # in the simulation mode, the time delay is constant
            time_delay = np.abs(self.async_overhead)

        # the data is 10 hz for both opv2v and v2x-set
        # todo: it may not be true for other dataset like DAIR-V2X and V2X-Sim
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose: NDArray[np.float64], xyz_std: float, ryp_std: float) -> NDArray[np.float64]:
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : NDArray[np.float64]
            Pose parameters [x, y, z, roll, yaw, pitch].
        xyz_std : float
            Standard deviation of Gaussian noise for xyz coordinates.
        ryp_std : float
            Standard deviation of Gaussian noise for roll, yaw, pitch.

        Returns
        -------
        NDArray[np.float64]
            Noisy pose with the same shape as input.
        """

        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0], pose[1] + xyz_noise[1], pose[2] + xyz_noise[2], pose[3], pose[4] + ryp_std[1], pose[5]]
        return noise_pose

    def reform_param(
        self, cav_content: Dict[str, Any], ego_content: Dict[str, Any], timestamp_cur: str, timestamp_delay: str, cur_ego_pose_flag: bool
    ) -> Dict[str, Any]:
        """
        Reform the data params with current timestamp object groundtruth and delay timestamp LiDAR pose for other CAVs.

        Parameters
        ----------
        cav_content : Dict[str, Any]
            Dictionary that contains all file paths in the current cav/rsu.
        ego_content : Dict[str, Any]
            Ego vehicle content.
        timestamp_cur : str
            The current timestamp.
        timestamp_delay : str
            The delayed timestamp.
        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Returns
        -------
        Dict[str, Any]
            The merged parameters with added transformation matrices.
        """
        cur_params = load_yaml(cav_content[timestamp_cur]["yaml"])
        delay_params = load_yaml(cav_content[timestamp_delay]["yaml"])

        cur_ego_params = load_yaml(ego_content[timestamp_cur]["yaml"])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]["yaml"])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params["lidar_pose"]
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params["lidar_pose"]
        cur_cav_lidar_pose = cur_params["lidar_pose"]

        if not cav_content["ego"] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose, cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose, delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose, cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose, cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params["vehicles"] = cur_params["vehicles"]
        delay_params["transformation_matrix"] = transformation_matrix
        delay_params["gt_transformation_matrix"] = gt_transformation_matrix
        delay_params["spatial_correction_matrix"] = spatial_correction_matrix

        return delay_params

    @staticmethod
    def load_camera_files(cav_path: str, timestamp: str) -> List[str]:
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.
        timestamp : str
            Current timestamp.

        Returns
        -------
        List[str]
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path, timestamp + "_camera0.png")
        camera1_file = os.path.join(cav_path, timestamp + "_camera1.png")
        camera2_file = os.path.join(cav_path, timestamp + "_camera2.png")
        camera3_file = os.path.join(cav_path, timestamp + "_camera3.png")
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def project_points_to_bev_map(self, points: NDArray[np.float64], ratio: float = 0.1) -> NDArray[np.float64]:
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : NDArray[np.float64]
            Point cloud array with shape (N, 3) or (N, 4).
        ratio : float, optional
            Discretization parameters. Default is 0.1.

        Returns
        -------
        NDArray[np.float64]
            BEV occupancy map including projected points
            with shape (img_row, img_col).
        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(
        self, lidar_np: NDArray[np.float64], object_bbx_center: NDArray[np.float64], object_bbx_mask: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : NDArray[np.float64]
            Point cloud array with shape (n, 4).
        object_bbx_center : NDArray[np.float64]
            Bounding box centers with shape (n, 7) to represent
            bbx's x, y, z, h, w, l, yaw.
        object_bbx_mask : NDArray[np.float64]
            Indicate which elements in object_bbx_center are padded.

        Returns
        -------
        lidar_np : NDArray[np.float64]
            Augmented point cloud.
        object_bbx_center : NDArray[np.float64]
            Augmented bounding boxes.
        object_bbx_mask : NDArray[np.float64]
            Updated mask after augmentation.
        """
        tmp_dict = {"lidar_np": lidar_np, "object_bbx_center": object_bbx_center, "object_bbx_mask": object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict["lidar_np"]
        object_bbx_center = tmp_dict["object_bbx_center"]
        object_bbx_mask = tmp_dict["object_bbx_mask"]

        return lidar_np, object_bbx_center, object_bbx_mask

    def collate_batch_train(self, batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Customized collate function for pytorch dataloader during training for early and late fusion dataset.

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            List of data samples from __getitem__.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Reformatted batch dictionary.
            Structure: {'ego': {'object_bbx_center': torch.Tensor,
                               'object_bbx_mask': torch.Tensor,
                               'processed_lidar': dict,
                               'label_dict': dict,
                               'origin_lidar': torch.Tensor [if visualize=True]}}
        """
        # during training, we only care about ego.
        output_dict = {"ego": {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            processed_lidar_list.append(ego_dict["processed_lidar"])
            label_dict_list.append(ego_dict["label_dict"])

            if self.visualize:
                origin_lidar.append(ego_dict["origin_lidar"])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)
        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "processed_lidar": processed_lidar_torch_dict,
                "label_dict": label_torch_dict,
            }
        )
        if self.visualize:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict["ego"].update({"origin_lidar": origin_lidar})

        return output_dict

    def visualize_result(
        self,
        pred_box_tensor: torch.Tensor,
        gt_tensor: torch.Tensor,
        pcd: NDArray[np.float64],
        show_vis: bool,
        save_path: str,
        dataset: Optional[Any] = None,
    ) -> None:
        """
        Visualize the model output.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            Predicted bounding boxes.
        gt_tensor : torch.Tensor
            Ground truth bounding boxes.
        pcd : NDArray[np.float64]
            Point cloud data.
        show_vis : bool
            Whether to show visualization.
        save_path : str
            Path to save visualization.
        dataset : Optional[Any], optional
            Dataset object for additional context. Default is None.
        """
        # visualize the model output
        self.post_processor.visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=dataset)
