import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3D(nn.Module):
    """
    3D convolution for processing temporal dimensions (STPN).
    Used in DiscoNet.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq, c, h, w)
        return x


class DiscoNetBackbone(nn.Module):
    """
    STPN Backbone for the DiscoNet model.
    Location in the project: opencood/models/sub_modules/
    """

    def __init__(self, args):
        super(DiscoNetBackbone, self).__init__()

        in_channels = args["in_channels"]
        self.compress_level = args.get("compress_level", 0)

        # Pre-processing слои
        self.conv_pre_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        # Encoder
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)

        # Decoder
        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6_1 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.bn7_1 = nn.BatchNorm2d(64)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.bn8_1 = nn.BatchNorm2d(32)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn8_2 = nn.BatchNorm2d(32)

    def encode(self, x):
        batch, seq, z, h, w = x.size()
        x = x.view(-1, z, h, w).float()

        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = x_1.view(batch, seq, -1, x_1.size(2), x_1.size(3)).contiguous()
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))
        x_2 = x_2.view(batch, seq, -1, x_2.size(2), x_2.size(3)).contiguous()
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        return [x, x_1, x_2, x_3, x_4]

    def decode(self, layers, batch, kd_flag=False):
        x, x_1, x_2, x_3, x_4 = layers

        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2), mode="bilinear", align_corners=True), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        def squeeze_seq(feat, b):
            if feat.size(0) == b:
                return feat
            if feat.size(0) > b:
                feat = feat.view(b, -1, feat.size(1), feat.size(2), feat.size(3))
                feat = feat.permute(0, 2, 1, 3, 4).contiguous()
                feat = F.adaptive_max_pool3d(feat, (1, None, None))
                feat = feat.permute(0, 2, 1, 3, 4).contiguous()
                return feat.view(-1, feat.size(2), feat.size(3), feat.size(4))
            return feat

        x_2 = squeeze_seq(x_2, batch)
        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2), mode="bilinear", align_corners=True), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = squeeze_seq(x_1, batch)
        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2), mode="bilinear", align_corners=True), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = squeeze_seq(x, batch)
        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2), mode="bilinear", align_corners=True), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        if kd_flag:
            return [res_x, x_7, x_6, x_5]

        return [res_x]
