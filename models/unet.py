import torch
import torch.nn as nn


class DepthwiseDecoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DepthwiseDecoderLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
