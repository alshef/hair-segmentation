import torch
import torch.nn as nn


class DepthwiseEncoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, preserve_size: bool = False) -> None:
        super(DepthwiseEncoderLayer, self).__init__()
        self.stride = 1 if preserve_size else int(out_channels / in_channels)

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
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=self.stride,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class InputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(InputLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
