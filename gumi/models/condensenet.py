""" This script implements the CondenseNet, only that we first need 

  to train the un-grouped version from scratch, and then learn the
  permutation.

https://github.com/ShichenLiu/CondenseNet/blob/master/models/condensenet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from gumi.model_utils import *

__all__ = ['CondenseNet', 'condensenet86']


class Bottleneck(nn.Module):
  """ Bottleneck module used in CondenseNet. """

  def __init__(self,
               in_channels,
               expansion=4,
               growth_rate=12,
               drop_rate=0,
               **kwargs):
    """ CTOR.
    Args:
      in_channels(int)
      expansion(int)
      growth_rate(int): the k value
      drop_rate(float): the dropout rate
    """
    super().__init__()

    # the input channels to the second 3x3 convolution layer
    channels = expansion * growth_rate

    # conv1: C -> 4 * k (according to the paper)
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.conv1 = conv1x1(in_channels, channels, **kwargs)
    # conv2: 4 * k -> k
    self.bn2 = nn.BatchNorm2d(channels)
    self.conv2 = conv3x3(channels, growth_rate, **kwargs)
    self.relu = nn.ReLU(inplace=True)

    self.drop_rate = drop_rate

  def forward(self, x):
    """ forward """
    # conv1
    out = self.bn1(x)
    out = self.relu(out)
    # dropout (This is the different part)
    if self.drop_rate > 0:
      out = F.dropout(out, p=self.drop_rate, training=self.training)

    out = self.conv1(out)
    # conv2
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)

    # concatenate results
    out = torch.cat((x, out), 1)

    return out


class DenseBlock(nn.Sequential):
  """ Handles the logic of creating Dense layers """

  def __init__(self, num_layers, in_channels, growth_rate, **kwargs):
    """ CTOR.
    Args:
      num_layers(int): from stages
      growth_rate(int): from growth
    """
    super().__init__()

    for i in range(num_layers):
      layer = Bottleneck(
          in_channels + i * growth_rate, growth_rate=growth_rate, **kwargs)
      self.add_module('denselayer_%d' % (i + 1), layer)


class Transition(nn.Module):
  """ CondenseNet's transition, no convolution involved """

  def __init__(self, in_channels):
    super().__init__()
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    x = self.pool(x)
    return x


class CondenseNet(nn.Module):
  """ Main function to initialise CondenseNet. """

  def __init__(self, stages, growth, num_classes=10, **kwargs):
    """ CTOR.
    Args:
      stages(list): per layer depth
      growth(list): per layer growth rate
    """
    super().__init__()

    self.stages = stages
    self.growth = growth
    assert len(self.stages) == len(self.growth)

    # NOTE(): we removed the imagenet related branch
    self.init_stride = 1
    self.pool_size = 8

    self.features = nn.Sequential()
    ### Initial nChannels should be 3
    # NOTE: this is a variable that traces the output size
    self.num_features = 2 * self.growth[0]

    ### Dense-block 1 (224x224)
    # NOTE: this block will not be turned into a GConv
    self.features.add_module(
        'init_conv',
        nn.Conv2d(
            3,
            self.num_features,
            kernel_size=3,
            stride=self.init_stride,
            padding=1,
            bias=False))

    for i in range(len(self.stages)):
      ### Dense-block i
      self.add_block(i, **kwargs)

    ### Linear layer
    self.classifier = nn.Linear(self.num_features, num_classes)

    ### initialize
    for m in self.modules():
      if is_conv2d(m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

  def add_block(self, i, **kwargs):
    ### Check if ith is the last one
    last = (i == len(self.stages) - 1)
    block = DenseBlock(
        num_layers=self.stages[i],
        in_channels=self.num_features,
        growth_rate=self.growth[i],
        **kwargs)

    self.features.add_module('denseblock_%d' % (i + 1), block)

    self.num_features += self.stages[i] * self.growth[i]
    if not last:
      trans = Transition(in_channels=self.num_features)
      self.features.add_module('transition_%d' % (i + 1), trans)
    else:
      self.features.add_module('norm_last', nn.BatchNorm2d(self.num_features))
      self.features.add_module('relu_last', nn.ReLU(inplace=True))
      self.features.add_module('pool_last', nn.AvgPool2d(self.pool_size))

  def forward(self, x, progress=None):
    # if progress:
    #   LearnedGroupConv.global_progress = progress
    features = self.features(x)
    out = features.view(features.size(0), -1)
    out = self.classifier(out)
    return out


def condensenet86(mask=True, **kwargs):
  return CondenseNet([14, 14, 14], [8, 16, 32], mask=mask, **kwargs)