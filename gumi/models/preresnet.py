""" Pre-activated ResNet models.

As shown in:
  He et al. Identity mappings in deep residual networks.
Reference code:
  https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
"""

import torch.nn as nn
import numpy as np
import math

from gumi.model_utils import *

__all__ = ["PreResNet", "preresnet164"]


class BasicBlock(nn.Module):
    """ bn -> relu -> conv3x3 -> bn -> relu -> conv3x3 """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        """ CTOR. """
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride, **kwargs)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, **kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # conv1
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        # conv2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        residual = x  # identity
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class BottleneckBlock(nn.Module):
    """ bottleneck. """

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        """ CTOR. """
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv1x1(in_channels, out_channels, **kwargs)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride, **kwargs)

        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(
            out_channels, out_channels * BottleneckBlock.expansion, **kwargs
        )
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """ forward """
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):
    """ pre-activated ResNet. """

    def __init__(self, depth, num_classes=1000, block_name="BasicBlock", **kwargs):
        """ CTOR for the model. """
        super().__init__()

        Block = self.name_to_block(block_name)  # the class
        N = self.num_blocks_per_layer(Block, depth)  # num blocks per layer

        self.in_channels = 3  # very first input channels
        self.channels = 16  # BASE output channels

        # first layer, different from other layers
        self.conv1 = nn.Conv2d(
            self.in_channels, self.channels, kernel_size=3, padding=1, bias=False
        )

        # layers
        self.in_channels = 16
        self.layer1 = self.make_layer(Block, self.channels, N, **kwargs)
        self.layer2 = self.make_layer(Block, self.channels * 2, N, stride=2, **kwargs)
        self.layer3 = self.make_layer(Block, self.channels * 4, N, stride=2, **kwargs)
        self.bn = nn.BatchNorm2d(self.channels * 4 * Block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.channels * 4 * Block.expansion, num_classes)

        # initialization
        for m in self.modules():
            if is_conv2d(m):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, Block, channels, num_blocks, stride=1, **kwargs):
        """ create a layer of blocks.

    NOTE: self.in_channels will be updated
    NOTE: channels is the unexpended number of channels
    """
        out_channels = channels * Block.expansion

        # create the downsample unit
        downsample = None
        if stride != 1 or self.in_channels != channels * Block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride=stride, **kwargs)
            )

        layers = []
        layers.append(
            Block(
                self.in_channels,
                channels,
                stride=stride,
                downsample=downsample,
                **kwargs
            )
        )

        # update in_channels
        self.in_channels = out_channels

        for i in range(1, num_blocks):
            layers.append(Block(self.in_channels, channels, **kwargs))

        return nn.Sequential(*layers)

    def name_to_block(self, block_name):
        """ From block_name to block class. """
        if block_name == "BasicBlock":
            return BasicBlock
        if block_name == "BottleneckBlock":
            return BottleneckBlock

        raise ValueError("block name {} cannot be recognized".format(block_name))

    def num_blocks_per_layer(self, Block, depth):
        """ Compute the number of blocks per BIG ResNet layer (unit). """
        if Block == BasicBlock:
            assert (depth - 2) % 6 == 0
            return (depth - 2) // 6  # 3 basic blocks per unit

        if Block == BottleneckBlock:
            assert (depth - 2) % 9 == 0
            return (depth - 2) // 9  # 3 bottleneck blocks per unit

        raise ValueError("Block class {} cannot be recognized".format(Block))

    def forward(self, x):
        """ forward """
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def preresnet164(mask=True, **kwargs):
    """ """
    return PreResNet(164, block_name="BottleneckBlock", mask=mask, **kwargs)
