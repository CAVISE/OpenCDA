import numpy as np
import torch
from torch import nn

from opencood.models.communication_adapters.fpvrcnn import FpvrcnnCommunicationAdapter
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.cia_ssd_utils import SSFA, Head
from opencood.models.sub_modules.vsa import VoxelSetAbstraction
from opencood.models.sub_modules.roi_head import RoIHead
from opencood.models.sub_modules.matcher import Matcher
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import FpvrcnnPostprocessor


class FPVRCNN(nn.Module):
    # The model declares its own wire boundary; neither the manager nor the
    # dataset needs to infer it from the selected fusion dataset.
    communication_adapter_class = FpvrcnnCommunicationAdapter

    def __init__(self, args):
        super(FPVRCNN, self).__init__()
        lidar_range = np.array(args["lidar_range"])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) / np.array(args["voxel_size"])).astype(np.int64)
        self.vfe = MeanVFE(args["mean_vfe"], args["mean_vfe"]["num_point_features"])
        self.spconv_block = VoxelBackBone8x(args["spconv"], input_channels=args["spconv"]["num_features_in"], grid_size=grid_size)
        self.map_to_bev = HeightCompression(args["map2bev"])
        self.ssfa = SSFA(args["ssfa"])
        self.head = Head(**args["head"])
        self.post_processor = FpvrcnnPostprocessor(args["post_processer"], train=True)
        self.vsa = VoxelSetAbstraction(args["vsa"], args["voxel_size"], args["lidar_range"], num_bev_features=128, num_rawpoint_features=3)
        self.matcher = Matcher(args["matcher"], args["lidar_range"])
        self.roi_head = RoIHead(args["roi_head"])
        self.train_stage2 = args["activate_stage2"]

    def _forward_stage_one(self, batch_dict):
        """
        Run the per-agent backbone and proposal head.

        Parameters
        ----------
        batch_dict : dict
            Collated voxel input for one or more independent agents.

        Returns
        -------
        tuple[dict, list[torch.Tensor] | None, list[torch.Tensor] | None]
            Updated model state, proposal boxes, and proposal scores.
        """
        voxel_features = batch_dict["processed_lidar"]["voxel_features"]
        voxel_coords = batch_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = batch_dict["processed_lidar"]["voxel_num_points"]

        # save memory
        batch_dict.pop("processed_lidar")
        batch_dict.update(
            {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": voxel_num_points,
                "batch_size": int(batch_dict["record_len"].sum()),
            }
        )

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)

        out = self.ssfa(batch_dict["spatial_features"])
        batch_dict["preds_dict_stage1"] = self.head(out)

        data_dict, output_dict = {}, {}
        data_dict["ego"], output_dict["ego"] = batch_dict, batch_dict

        pred_box3d_list, scores_list = self.post_processor.post_process(data_dict, output_dict, stage1=True)
        batch_dict["det_boxes"] = pred_box3d_list
        batch_dict["det_scores"] = scores_list

        return batch_dict, pred_box3d_list, scores_list

    def encode_agent(self, batch_dict):
        """
        Build the FPV-RCNN representation transmitted by one agent.

        Parameters
        ----------
        batch_dict : dict
            Single-agent voxel input, raw LiDAR points, and anchors.

        Returns
        -------
        dict[str, torch.Tensor]
            Stage-one proposals and VSA keypoint coordinates/features in the
            sender coordinate frame.

        Raises
        ------
        RuntimeError
            If cooperative stage two is disabled in the model configuration.
        """
        if not self.train_stage2:
            raise RuntimeError("Distributed FPV-RCNN inference requires activate_stage2=true")
        batch_dict, proposals, proposal_scores = self._forward_stage_one(batch_dict)
        if proposals is None or proposal_scores is None:
            device = batch_dict["voxel_features"].device
            return {
                "proposals": torch.empty((0, 7), device=device),
                "proposal_scores": torch.empty((0,), device=device),
                "point_coords": torch.empty((0, 4), device=device),
                "point_features": torch.empty((0, self.vsa.num_point_features), device=device),
            }

        batch_dict = self.vsa(batch_dict)
        return {
            "proposals": proposals[0],
            "proposal_scores": proposal_scores[0],
            "point_coords": batch_dict["point_coords"][0],
            "point_features": batch_dict["point_features"][0],
        }

    def _forward_distributed(self, batch_dict):
        """
        Fuse local FPV-RCNN output with delivered agent outputs.

        Parameters
        ----------
        batch_dict : dict
            Ego input containing ``remote_agent_outputs`` decoded from wire
            payloads.

        Returns
        -------
        dict
            Cooperative stage-two model output.
        """
        remote_outputs = batch_dict.pop("remote_agent_outputs")
        local_output = self.encode_agent(batch_dict)
        agent_outputs = [local_output, *remote_outputs]

        batch_dict["det_boxes"] = [output["proposals"] for output in agent_outputs]
        batch_dict["det_scores"] = [output["proposal_scores"] for output in agent_outputs]
        batch_dict["point_coords"] = [output["point_coords"] for output in agent_outputs]
        batch_dict["point_features"] = [output["point_features"] for output in agent_outputs]
        batch_dict["record_len"] = torch.tensor(
            [len(agent_outputs)],
            dtype=torch.long,
            device=batch_dict["voxel_features"].device,
        )

        if not any(len(boxes) > 0 for boxes in batch_dict["det_boxes"]):
            batch_dict["fpvrcnn_out"] = None
            return batch_dict

        batch_dict = self.matcher(batch_dict)
        return self.roi_head(batch_dict)

    @staticmethod
    def _transform_agent_outputs(batch_dict):
        """
        Transform local proposals and keypoints into each scene's ego frame.

        Parameters
        ----------
        batch_dict : dict
            Model state containing per-agent detections, keypoints, and one
            ``agent_to_ego`` matrix per flattened agent.

        Returns
        -------
        dict
            Model state with proposal centers, headings, and keypoint
            coordinates expressed in their corresponding ego frames.
        """
        transformations = batch_dict.pop("agent_to_ego")
        if len(transformations) != len(batch_dict["det_boxes"]) or len(transformations) != len(batch_dict["point_coords"]):
            raise ValueError("Every FPV-RCNN agent output requires one agent-to-ego transformation")
        transformed_boxes = []
        transformed_points = []

        for boxes, points, transformation in zip(
            batch_dict["det_boxes"],
            batch_dict["point_coords"],
            transformations,
        ):
            boxes = boxes.clone()
            if len(boxes) > 0:
                box_transformation = transformation.to(dtype=boxes.dtype)
                homogeneous_centers = torch.cat(
                    [boxes[:, :3], torch.ones((len(boxes), 1), dtype=boxes.dtype, device=boxes.device)],
                    dim=1,
                )
                boxes[:, :3] = (box_transformation @ homogeneous_centers.T).T[:, :3]
                boxes[:, 6] += torch.atan2(box_transformation[1, 0], box_transformation[0, 0])
            transformed_boxes.append(boxes)

            points = points.clone()
            if len(points) > 0:
                point_transformation = transformation.to(dtype=points.dtype)
                homogeneous_points = torch.cat(
                    [points[:, :3], torch.ones((len(points), 1), dtype=points.dtype, device=points.device)],
                    dim=1,
                )
                points[:, :3] = (point_transformation @ homogeneous_points.T).T[:, :3]
            transformed_points.append(points)

        batch_dict["det_boxes"] = transformed_boxes
        batch_dict["point_coords"] = transformed_points
        return batch_dict

    def forward(self, batch_dict):
        if "remote_agent_outputs" in batch_dict:
            return self._forward_distributed(batch_dict)

        batch_dict, pred_box3d_list, _ = self._forward_stage_one(batch_dict)

        if pred_box3d_list is not None and self.train_stage2:
            batch_dict = self.vsa(batch_dict)
            if "agent_to_ego" in batch_dict:
                batch_dict = self._transform_agent_outputs(batch_dict)
            batch_dict = self.matcher(batch_dict)
            batch_dict = self.roi_head(batch_dict)
        if pred_box3d_list is None:
            batch_dict["fpvrcnn_out"] = None

        return batch_dict


if __name__ == "__main__":
    model = SSFA(None)
    print(model)
