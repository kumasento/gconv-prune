""" Implements DenseNet.

References:
- Huang et al. Densely Connected Convolutional Networks. 2017
- https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/densenet.py
- https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from gumi.model_utils import *

__all__ = ['DenseNet', 'densenet40']


class Bottleneck(nn.Module):
  """ Bottleneck module used in DenseNet-BC. """

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
    out = self.conv1(out)
    # conv2
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)
    # dropout
    if self.drop_rate > 0:
      out = F.dropout(out, p=self.drop_rate, training=self.training)
    # concatenate results
    out = torch.cat((x, out), 1)

    return out


class BasicBlock(nn.Module):
  """ Basic block """

  def __init__(self, in_channels, growth_rate=12, drop_rate=0, **kwargs):
    """ CTOR.
    Args:
      in_channels(int)
      expansion(int)
      growth_rate(int): the k value
      drop_rate(float): the dropout rate
    """
    super().__init__()

    self.bn = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv = conv3x3(in_channels, growth_rate, **kwargs)
    self.drop_rate = drop_rate

  def forward(self, x):
    """ forward """
    out = self.bn(x)
    out = self.relu(out)
    out = self.conv(out)
    # dropout
    if self.drop_rate > 0:
      out = F.dropout(out, p=self.drop_rate, training=self.training)
    # concatenate results
    out = torch.cat((x, out), 1)

    return out


class Transition(nn.Module):
  """ transition layer in between two blocks """

  def __init__(self, in_channels, out_channels, **kwargs):
    super().__init__()

    self.bn1 = nn.BatchNorm2d(in_channels)
    self.conv1 = conv1x1(in_channels, out_channels, **kwargs)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    """ forward """
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv1(x)
    x = F.avg_pool2d(x, 2)
    return x


class DenseNet(nn.Module):
  """ Base class for DenseNet.
  
    The way to convert L to number of blocks
    (https://github.com/liuzhuang13/DenseNet/blob/b18a75dc9750becae3ba8aff1cfccef84deb5e86/models/densenet.lua#L24)

      N = (L - 4) / 6

    It is assumed that there are 3 Dense blocks.
  """

  def __init__(self,
               depth=40,
               Block=Bottleneck,
               num_classes=10,
               drop_rate=0,
               growth_rate=12,
               compression_rate=0.5,
               **kwargs):
    """ CTOR. """
    super().__init__()

    self.growth_rate = growth_rate
    self.drop_rate = drop_rate
    self.compression_rate = compression_rate

    # number of blocks per layer
    N = self.get_num_blocks_per_layer(Block, depth)

    # channels
    if Block == Bottleneck:
      self.in_channels = self.growth_rate * 2  # based on the paper
    elif Block == BasicBlock:
      self.in_channels = 16

    self.conv1 = conv3x3(3, self.in_channels, **kwargs)

    self.dense1 = self.make_dense(Block, N, **kwargs)
    self.trans1 = self.make_trans(**kwargs)
    self.dense2 = self.make_dense(Block, N, **kwargs)
    self.trans2 = self.make_trans(**kwargs)
    self.dense3 = self.make_dense(Block, N, **kwargs)
    self.bn = nn.BatchNorm2d(self.in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.avgpool = nn.AvgPool2d(8)
    self.fc = nn.Linear(self.in_channels, num_classes)

    for m in self.modules():
      if is_conv2d(m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def get_num_blocks_per_layer(self, Block, depth):
    """ Get number of blocks per layer. """
    if Block == Bottleneck:
      assert (depth - 4) % 6 == 0
      return (depth - 4) // 6
    if Block == BasicBlock:
      assert (depth - 4) % 3 == 0
      return (depth - 4) // 3

    raise ValueError('Class for block {} cannot be recognized.'.format(Block))

  def make_dense(self, Block, num_blocks, **kwargs):
    """ Create a dense block that contains multiple Block """
    layers = []

    for i in range(num_blocks):
      layers.append(
          Block(
              self.in_channels,
              growth_rate=self.growth_rate,
              drop_rate=self.drop_rate,
              **kwargs))
      self.in_channels += self.growth_rate  # concatenating

    return nn.Sequential(*layers)

  def make_trans(self, **kwargs):
    """ Create the transition layer. """
    in_channels = self.in_channels

    # compute number of output channels
    out_channels = int(math.floor(self.in_channels * self.compression_rate))
    self.in_channels = out_channels

    return Transition(in_channels, out_channels, **kwargs)

  def forward(self, x):
    x = self.conv1(x)

    x = self.trans1(self.dense1(x))
    x = self.trans2(self.dense2(x))
    x = self.dense3(x)
    x = self.bn(x)
    x = self.relu(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def densenet40(mask=True, **kwargs):
  """ L = 40, k = 12 """
  return DenseNet(
      depth=40,
      Block=BasicBlock,  # remember its BasicBlock
      growth_rate=12,
      compression_rate=1.0,
      mask=mask,
      **kwargs)
