""" MobileNet model definition """
import torch
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from gumi.ops import *

__all__ = ["MobileNet", "mobilenet"]


class DepthwiseSeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, sparse=False):
        super().__init__()

        self.add_module(
            "d_conv",
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
        )
        self.add_module("bn1", nn.BatchNorm2d(in_channels))
        self.add_module("relu1", nn.ReLU(inplace=True))

        self.add_module(
            "p_conv",
            self.get_pointwise_conv2d(in_channels, out_channels, groups, sparse=sparse),
        )
        self.add_module("bn2", nn.BatchNorm2d(out_channels))
        self.add_module("relu2", nn.ReLU(inplace=True))

    def get_pointwise_conv2d(self, in_channels, out_channels, groups, sparse=True):
        """ Create the pointwise conv2d module """
        if sparse:
            return SparseGroupConv2d(
                in_channels, out_channels, 1, groups=groups, bias=False
            )
        else:
            return nn.Conv2d(in_channels, out_channels, 1, groups=groups, bias=False)


class MobileNet(nn.Module):
    """ MobileNet v1 224 1.0 definition """

    def __init__(self, groups=1, sparse=False, num_channels=1000):
        super().__init__()

        self.in_channels = 3
        self.channels = 32

        # the first layer
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # all the depthwise separable layers
        self.layer1 = nn.Sequential(
            DepthwiseSeparableConv2d(32, 64, 1), DepthwiseSeparableConv2d(64, 128, 2),
        )
        self.layer2 = nn.Sequential(
            DepthwiseSeparableConv2d(128, 128, 1),
            DepthwiseSeparableConv2d(128, 256, 2),
        )
        self.layer3 = nn.Sequential(
            DepthwiseSeparableConv2d(256, 256, 1),
            DepthwiseSeparableConv2d(256, 512, 2),
        )
        self.layer4 = nn.Sequential(
            DepthwiseSeparableConv2d(512, 512, 1, groups, sparse),
            DepthwiseSeparableConv2d(512, 512, 1, groups, sparse),
            DepthwiseSeparableConv2d(512, 512, 1, groups, sparse),
            DepthwiseSeparableConv2d(512, 512, 1, groups, sparse),
            DepthwiseSeparableConv2d(512, 512, 1, groups, sparse),
            DepthwiseSeparableConv2d(512, 1024, 2, groups, sparse),
        )
        self.layer5 = nn.Sequential(
            DepthwiseSeparableConv2d(1024, 1024, 1, groups, sparse)
        )
        self.pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, num_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = x.view(-1, 1024)

        return self.fc(x)


def mobilenet(**kwargs):
    return MobileNet(**kwargs)
