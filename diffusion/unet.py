# Implement a U-Net in Pytorch.

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.channels = [hparams.input_dim[0]] + hparams.channels # [3, 64, 128, 256, 512]
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.upsampler = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.downsampler = nn.MaxPool2d(kernel_size=2)

        for idx in range(len(self.channels) - 1):
            self.down_convs.append(DoubleConv(self.channels[idx], self.channels[idx + 1]))
        for idx in range(len(self.channels) - 1, 0, -1):
            self.up_convs.append(DoubleConv(self.channels[idx] * 2, self.channels[idx - 1]))
        
        self.bottleneck_conv = DoubleConv(self.channels[-1], self.channels[-1])

    def forward(self, x):
        skip_connections = []
        for down_conv in self.down_convs:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.downsampler(x)
        x = self.bottleneck_conv(x)
        for idx, up_conv in enumerate(self.up_convs):
            x = self.upsampler(x)
            x = torch.cat([x, skip_connections[-idx - 1]], dim=1)
            x = up_conv(x)
        return x
