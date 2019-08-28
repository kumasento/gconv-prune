""" Unittests for densenet. """

import unittest

import torch

from gumi import model_utils
from gumi.ops import *
from gumi.models import densenet


class TestDenseNet(unittest.TestCase):
  """ """

  def test_ctor(self):
    """ Build BasicBlocks and BottleneckBlocks, and the model. """
    # the network model itself
    model = densenet.DenseNet(
        depth=40,
        Block=densenet.BasicBlock,
        growth_rate=12,
        compression_rate=1.0,
        mask=True,
        num_classes=100)
    num_params = model_utils.get_model_num_params(model)

    self.assertAlmostEqual(num_params, 1.06, places=1)  # around 1.7
    self.assertEqual(model_utils.get_num_conv2d_layers(model), 40)

  def test_forward(self):
    """ forward should be OK """
    # CIFAR
    model = densenet.DenseNet(
        depth=40,
        Block=densenet.BasicBlock,
        growth_rate=12,
        mask=True,
        compression_rate=1.0,
        num_classes=100)
    model.forward(torch.randn((1, 3, 32, 32)))


if __name__ == '__main__':
  unittest.main()