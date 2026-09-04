import torch.nn as nn


class NaiveCompressor(nn.Module):
    """
    A very naive compression that only compress on the channel.
    """

    def __init__(self, input_dim, compress_raito):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // compress_raito, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim // compress_raito, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim // compress_raito, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        """
        Produce the channel-compressed representation for transmission.

        Parameters
        ----------
        x : torch.Tensor
            Full-channel spatial feature map.

        Returns
        -------
        torch.Tensor
            Channel-compressed feature map.
        """
        return self.encoder(x)

    def decode(self, x):
        """
        Restore a transmitted channel-compressed representation.

        Parameters
        ----------
        x : torch.Tensor
            Channel-compressed feature map.

        Returns
        -------
        torch.Tensor
            Feature map restored to the fusion network's channel count.
        """
        return self.decoder(x)
