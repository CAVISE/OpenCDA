from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LaneMapEncoder(nn.Module):
    def __init__(
        self,
        base_channels=16,
        stages=(3, 4, 4, 3, 3),
        out_channels=128,
    ):
        super().__init__()
        self.stem = ConvBlock(1, base_channels, kernel_size=4, stride=4, padding=0)  # image_size /4; channels
        self.stages = nn.ModuleList()
        channels = base_channels

        for i, depth in enumerate(stages):
            for j in range(depth):
                if j == 0 and i > 0:
                    stage = ConvBlock(channels, channels * 2, kernel_size=2, stride=2, padding=0)  # image_size /2; channels *2
                    channels *= 2
                else:
                    stage = ConvBlock(channels, channels)

                self.stages.append(stage)

        self.out_channels = channels
        self.norm = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, out_channels)

    def forward(self, x):
        """
        x: (b, n, k, k)
        """
        b, n, k1, k2 = x.shape
        assert k1 == k2

        x = x.reshape(b * n, 1, k1, k2)  # (b * n, 1, k, k)
        x = self.stem(x)  # (b * n, 1, k/4, k/4)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)

        x = self.pool(x)  # (b * n, p, 1, 1)
        x = x.flatten(1)  # (b * n, p)
        x = self.head(x)  # (b * n, out_channels)
        x = x.reshape(b, n, -1)  # (b, n, out_channels)

        return x
