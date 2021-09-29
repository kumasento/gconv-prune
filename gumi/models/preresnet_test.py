""" Unittests for preresnet. """

import unittest

import torch

from gumi import model_utils
from gumi.ops import *
from gumi.models import preresnet


class TestPreResNet(unittest.TestCase):
    """ """

    def test_ctor(self):
        """ Build BasicBlocks and BottleneckBlocks, and the model. """

        # check masked BasicBlock
        basic_block = preresnet.BasicBlock(32, 32, mask=True)
        self.assertIsInstance(basic_block.conv1, MaskConv2d)
        self.assertIsInstance(basic_block.conv2, MaskConv2d)

        # masked BottleneckBlock
        bottleneck_block = preresnet.BottleneckBlock(32, 32, mask=True)
        self.assertIsInstance(bottleneck_block.conv1, MaskConv2d)
        self.assertIsInstance(bottleneck_block.conv2, MaskConv2d)
        self.assertIsInstance(bottleneck_block.conv3, MaskConv2d)

        # the network model itself
        model = preresnet.PreResNet(
            164, mask=True, block_name="BottleneckBlock", num_classes=100
        )
        num_params = model_utils.get_model_num_params(model)

        self.assertAlmostEqual(num_params, 1.70, places=1)  # around 1.7
        self.assertEqual(model_utils.get_num_conv2d_layers(model), 164)

    def test_forward(self):
        """ forward should be OK """
        # CIFAR
        model = preresnet.PreResNet(
            164, mask=True, block_name="BottleneckBlock", num_classes=100
        )
        model.forward(torch.randn((1, 3, 32, 32)))


if __name__ == "__main__":
    unittest.main()
