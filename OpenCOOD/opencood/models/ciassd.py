from torch import nn
import numpy as np

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head


class CIASSD(nn.Module):
    def __init__(self, args):
        super(CIASSD, self).__init__()
        lidar_range = np.array(args["lidar_range"])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) / np.array(args["voxel_size"])).astype(np.int64)
        self.vfe = MeanVFE(args["mean_vfe"], args["mean_vfe"]["num_point_features"])
        self.spconv_block = VoxelBackBone8x(args["spconv"], input_channels=args["spconv"]["num_features_in"], grid_size=grid_size)
        self.map_to_bev = HeightCompression(args["map2bev"])
        self.ssfa = SSFA(args["ssfa"])
        self.head = Head(**args["head"])

    def forward(self, batch_dict):
        processed_lidar = batch_dict["processed_lidar"]
        batch_size = batch_dict.get("batch_size")
        if batch_size is None:
            batch_size = batch_dict["object_bbx_center"].shape[0]
        model_batch = {
            **processed_lidar,
            "batch_size": batch_size,
        }
        model_batch = self.vfe(model_batch)
        model_batch = self.spconv_block(model_batch)
        model_batch = self.map_to_bev(model_batch)
        out = self.ssfa(model_batch["spatial_features"])
        out = self.head(out)
        return {
            "preds_dict_stage1": out,
            "batch_size": batch_size,
            "anchor_box": batch_dict.get("anchor_box"),
        }


if __name__ == "__main__":
    model = SSFA(None)
    print(model)
