import torch
import torch.nn as nn

from .encoders.layers import DepthwiseEncoderLayer, InputLayer


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


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super(UNet, self).__init__()

        self.encode_layer_1 = nn.Sequential(
            InputLayer(in_channels, 32),
            DepthwiseEncoderLayer(32, 64, preserve_size=True)
        )
        self.encode_layer_2 = nn.Sequential(
            DepthwiseEncoderLayer(64, 128),
            DepthwiseEncoderLayer(128, 128),
        )
        self.encode_layer_3 = nn.Sequential(
            DepthwiseEncoderLayer(128, 256),
            DepthwiseEncoderLayer(256, 256)
        )
        self.encode_layer_4 = nn.Sequential(
            DepthwiseEncoderLayer(256, 512),
            DepthwiseEncoderLayer(512, 512),
            DepthwiseEncoderLayer(512, 512),
            DepthwiseEncoderLayer(512, 512),
            DepthwiseEncoderLayer(512, 512),
            DepthwiseEncoderLayer(512, 512),
        )
        self.encode_layer_5 = nn.Sequential(
            DepthwiseEncoderLayer(512, 1024),
            DepthwiseEncoderLayer(1024, 1024)
        )

        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.upsample_3 = nn.Upsample(scale_factor=2)
        self.upsample_4 = nn.Upsample(scale_factor=2)
        self.upsample_5 = nn.Upsample(scale_factor=2)

        self.decode_layer_5 = DepthwiseDecoderLayer(in_channels=64, out_channels=64)
        self.decode_layer_4 = DepthwiseDecoderLayer(in_channels=64, out_channels=64)
        self.decode_layer_3 = DepthwiseDecoderLayer(in_channels=64, out_channels=64)
        self.decode_layer_2 = DepthwiseDecoderLayer(in_channels=64, out_channels=64)
        self.decode_layer_1 = DepthwiseDecoderLayer(in_channels=1024, out_channels=64)

        self.skip_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False)
        self.skip_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False)
        self.skip_3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=False)
        self.skip_4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, bias=False)

        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, bias=False)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encode_1 = self.encode_layer_1(x)
        encode_2 = self.encode_layer_2(encode_1)
        encode_3 = self.encode_layer_3(encode_2)
        encode_4 = self.encode_layer_4(encode_3)
        encode_5 = self.encode_layer_5(encode_4)

        skip_1 = self.skip_1(encode_1)
        skip_2 = self.skip_2(encode_2)
        skip_3 = self.skip_3(encode_3)
        skip_4 = self.skip_4(encode_4)

        decode_1 = self.decode_layer_1(self.upsample_1(encode_5) + skip_4)
        decode_2 = self.decode_layer_2(self.upsample_2(decode_1) + skip_3)
        decode_3 = self.decode_layer_3(self.upsample_3(decode_2) + skip_2)
        decode_4 = self.decode_layer_4(self.upsample_4(decode_3) + skip_1)
        decode_5 = self.decode_layer_5(self.upsample_5(decode_4))

        return self.output(decode_5)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
