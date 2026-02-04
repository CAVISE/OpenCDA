from typing import Any, Dict
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.data_utils.post_processor.bev_postprocessor import BevPostprocessor
from opencood.data_utils.post_processor.ciassd_postprocessor import CiassdPostprocessor
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import FpvrcnnPostprocessor
from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor

__all__ = {
    "VoxelPostprocessor": VoxelPostprocessor,
    "BevPostprocessor": BevPostprocessor,
    "CiassdPostprocessor": CiassdPostprocessor,
    "FpvrcnnPostprocessor": FpvrcnnPostprocessor,
}


def build_postprocessor(anchor_cfg: Dict[str, Any], train: bool) -> BasePostprocessor:
    """
    Build an OpenCOOD postprocessor from configuration.

    Parameters
    ----------
    anchor_cfg : dict
        Postprocess configuration. Must contain key "core_method" specifying the
        postprocessor class name.
    train : bool
        Whether to construct the postprocessor in training mode.

    Returns
    -------
    BasePostprocessor
        Instantiated postprocessor selected by `anchor_cfg["core_method"]`.
    """
    process_method_name = anchor_cfg["core_method"]
    assert process_method_name in ["VoxelPostprocessor", "BevPostprocessor", "CiassdPostprocessor", "FpvrcnnPostprocessor"]
    anchor_generator = __all__[process_method_name](anchor_params=anchor_cfg, train=train)

    return anchor_generator
