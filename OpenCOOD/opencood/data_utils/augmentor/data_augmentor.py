"""
Data augmentation pipeline for 3D point clouds and bounding boxes.

This module provides a flexible data augmentation pipeline that can be configured
through a YAML configuration file. It supports various augmentation techniques
such as random flipping, rotation, and scaling of point clouds and their
corresponding 3D bounding boxes.
"""

from functools import partial
from typing import Dict, Any, List, Optional, Callable, Union, cast, Tuple
from opencood.data_utils.augmentor import augment_utils


class DataAugmentor:
    """
    Data Augmentor.

    Parameters
    ----------
    augment_config : List[Dict[str, Any]]
        A list of augmentation configuration dictionaries.
    train : bool, optional
        Whether to apply augmentations. Default is True.

    Attributes
    ----------
    data_augmentor_queue : List[Callable]
        The list of data augmentation functions.
    train : bool
        Training mode flag.
    """

    def __init__(self, augment_config: List[Dict[str, Any]], train: bool = True) -> None:
        self.data_augmentor_queue = []
        self.train = train

        for cur_cfg in augment_config:
            cur_augmentor = getattr(self, cur_cfg["NAME"])(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def random_world_flip(
        self, data_dict: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], Callable]:
        """
        Randomly flip the world along specified axes.

        Parameters
        ----------
        data_dict : dict, optional
            Data dictionary containing point clouds and bounding boxes.
        config : dict, optional
            Configuration dictionary with key:
            - ALONG_AXIS_LIST: List of axes to flip (['x', 'y']).

        Returns
        -------
        dict or Callable
            If data_dict is provided, returns updated data dictionary.
            If data_dict is None, returns a partial function to apply later.
        """
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        gt_boxes, gt_mask, points = (
            data_dict["object_bbx_center"],
            data_dict["object_bbx_mask"],
            data_dict["lidar_np"],
        )
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        for cur_axis in config["ALONG_AXIS_LIST"]: #TODO Value of type "dict[str, Any] | None" is not indexable
            assert cur_axis in ["x", "y"]
            gt_boxes_valid, points = getattr(augment_utils, "random_flip_along_%s" % cur_axis)(
                gt_boxes_valid,
                points,
            )

        gt_boxes[: gt_boxes_valid.shape[0], :] = gt_boxes_valid
        data_dict["object_bbx_center"] = gt_boxes
        data_dict["object_bbx_mask"] = gt_mask
        data_dict["lidar_np"] = points

        return data_dict

    def random_world_rotation(
        self, data_dict: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], Callable]:
        """
        Apply random rotation to the world.

        Parameters
        ----------
        data_dict : dict, optional
            Data dictionary containing point clouds and bounding boxes.
        config : dict, optional
            Configuration dictionary with key:
            - WORLD_ROT_ANGLE: List of [min_angle, max_angle] in radians or single value.

        Returns
        -------
        dict or Callable
            If data_dict is provided, returns updated data dictionary.
            If data_dict is None, returns a partial function to apply later.
        """
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)

        rot_range = config["WORLD_ROT_ANGLE"] #TODO  error: Value of type "dict[str, Any] | None" is not indexable
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, gt_mask, points = (
            data_dict["object_bbx_center"],
            data_dict["object_bbx_mask"],
            data_dict["lidar_np"],
        )
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        gt_boxes_valid, points = augment_utils.global_rotation(gt_boxes_valid, points, rot_range=rot_range) #TODO Argument "rot_range" to "global_rotation" has incompatible type "list[Any]"; expected "tuple[float, float]"
        gt_boxes[: gt_boxes_valid.shape[0], :] = gt_boxes_valid
        data_dict["object_bbx_center"] = gt_boxes
        data_dict["object_bbx_mask"] = gt_mask
        data_dict["lidar_np"] = points

        return data_dict

    def random_world_scaling(
        self, data_dict: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], Callable]:
        """
        Apply random scaling to the world.

        Parameters
        ----------
        data_dict : dict, optional
            Data dictionary containing point clouds and bounding boxes.
        config : dict, optional
            Configuration dictionary with key:
            - WORLD_SCALE_RANGE: List of [min_scale, max_scale].

        Returns
        -------
        dict or Callable
            If data_dict is provided, returns updated data dictionary.
            If data_dict is None, returns a partial function to apply later.
        """
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        gt_boxes, gt_mask, points = (
            data_dict["object_bbx_center"],
            data_dict["object_bbx_mask"],
            data_dict["lidar_np"],
        )
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        gt_boxes_valid, points = augment_utils.global_scaling(gt_boxes_valid, points, config["WORLD_SCALE_RANGE"]) ##TODO Value of type "dict[str, Any] | None" is not indexable
        gt_boxes[: gt_boxes_valid.shape[0], :] = gt_boxes_valid
        data_dict["object_bbx_center"] = gt_boxes
        data_dict["object_bbx_mask"] = gt_mask
        data_dict["lidar_np"] = points

        return data_dict

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all configured augmentations to the input data.

        Parameters
        ----------
        data_dict : dict
            Data dictionary containing:
            - points: (N, 3 + C_in) point cloud coordinates and features
            - object_bbx_center: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
            - object_bbx_mask: optional, (N) mask for valid boxes
            - lidar_np: point cloud numpy array

        Returns
        -------
        dict
            Augmented data dictionary with the same structure.
        """
        if self.train:
            for cur_augmentor in self.data_augmentor_queue:
                data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict
