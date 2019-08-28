""" The model definition of VGG.

Credit to 
https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/vgg.py
"""
import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from gumi.ops import *
from gumi.model_utils import *

__all__ = [
    'VGG',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19_bn',
    'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):
  """ VGG model """

  def __init__(self, features, num_classes=1000):
    """ CTOR. Layers have been initialized as features. """
    super(VGG, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(nn.Linear(512, num_classes))
    self._initialize_weights()

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if is_conv2d(m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, **kwargs):
  """ Create a list of layers, composed as a Sequence. """
  layers = []
  in_channels = 3

  for v in cfg:
    if v == 'M':  # max-pool
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      # Convolution
      if in_channels == 3:  # The first convolution layer
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      else:
        conv2d = conv3x3(in_channels, v, bias=True, **kwargs)

      # Layers append convolution
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]

      # Update in_channels
      in_channels = v

  return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
        512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
        'M', 512, 512, 512, 512, 'M'
    ],
}


def vgg11(**kwargs):
  """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  model = VGG(make_layers(cfg['A']), **kwargs)
  return model


def vgg11_bn(**kwargs):
  """VGG 11-layer model (configuration "A") with batch normalization"""
  model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
  return model


def vgg13(**kwargs):
  """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  model = VGG(make_layers(cfg['B']), **kwargs)
  return model


def vgg13_bn(**kwargs):
  """VGG 13-layer model (configuration "B") with batch normalization"""
  model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
  return model


def vgg16(**kwargs):
  """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  model = VGG(make_layers(cfg['D']), **kwargs)
  return model


def vgg16_bn(**kwargs):
  """VGG 16-layer model (configuration "D") with batch normalization"""
  model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
  return model


def vgg19(**kwargs):
  """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  model = VGG(make_layers(cfg['E']), **kwargs)
  return model


def vgg19_bn(**kwargs):
  """VGG 19-layer model (configuration 'E') with batch normalization"""
  model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
  return model
