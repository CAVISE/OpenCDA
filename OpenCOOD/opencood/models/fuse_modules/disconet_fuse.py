import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sub_modules.torch_transformation_utils import get_discretized_transformation_matrix, warp_affine


class PixelWeightedFusionSoftmax(nn.Module):
    """
    A small subnetwork for computing attention weights for each pixel.
    Takes as input the concatenation of ego and neighbor features.
    """

    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x shape: (N, C*2, H, W)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn1_3(self.conv1_3(x)))
        x = self.conv1_4(x)  # На выходе (N, 1, H, W) - карта весов
        return x


class DiscoNetFusion(nn.Module):
    """
    Main fusion module of DiscoNet.
    Transforms neighbor features into the ego vehicle coordinate system
    and fuses them via Pixel-wise Attention.
    """

    def __init__(self, args):
        super(DiscoNetFusion, self).__init__()

        self.in_channels = args.get("fusion_channels", 256)
        self.discrete_ratio = args["voxel_size"][0]
        self.downsample_rate = args["downsample_rate"]
        self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(self.in_channels)

    def forward(self, x, record_len, pairwise_t_matrix):
        """
        Parameters:
         ----------
        x : torch.Tensor
            Features of all agents in the batch, shape (B*L, C, H, W)
        record_len : torch.Tensor
            Number of agents in each scene of the batch, shape (B)
        pairwise_t_matrix : torch.Tensor
            Transformation matrices between all pairs of CAVs, shape (B, L, L, 4, 4)
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        pairwise_t_matrix = get_discretized_transformation_matrix(
            pairwise_t_matrix.reshape(-1, L, 4, 4), self.discrete_ratio, self.downsample_rate
        ).reshape(B, L, L, 2, 3)

        split_x = torch.split(x, record_len.tolist())

        out_fused_features = []

        for b in range(B):
            N = record_len[b]
            batch_node_features = split_x[b]

            updated_node_features = []

            for i in range(N):
                ego_agent_feature = batch_node_features[i].unsqueeze(0)

                neighbor_feature_list = []
                neighbor_weight_list = []

                for j in range(N):
                    t_matrix = pairwise_t_matrix[b, j, i].unsqueeze(0)
                    neighbor_feature_warped = warp_affine(batch_node_features[j].unsqueeze(0), t_matrix, (H, W))

                    cat_feat = torch.cat([ego_agent_feature, neighbor_feature_warped], dim=1)
                    weight = self.pixel_weighted_fusion(cat_feat)

                    neighbor_feature_list.append(neighbor_feature_warped)
                    neighbor_weight_list.append(torch.exp(weight))

                neighbor_weight_stack = torch.stack(neighbor_weight_list)
                sum_weight = torch.sum(neighbor_weight_stack, dim=0)

                fused_feature_i = 0
                for k in range(N):
                    normalized_weight = neighbor_weight_list[k] / (sum_weight + 1e-6)
                    fused_feature_i += normalized_weight * neighbor_feature_list[k]

                updated_node_features.append(fused_feature_i)

            out_fused_features.append(torch.cat(updated_node_features, dim=0))

        return torch.cat(out_fused_features, dim=0)
