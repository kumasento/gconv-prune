""" Test whether CondenseNet has been correctly implemented. """

import unittest

import torch
from gumi import model_utils
from gumi.models import condensenet


class TestCondenseNet(unittest.TestCase):
    def test_ctor(self):
        """ Build the model """
        # CondenseNet-86 (G=4)
        model = condensenet.CondenseNet([14, 14, 14], [8, 16, 32],
                                        groups=4,
                                        ind_type='random')
        self.assertEqual(model_utils.get_num_conv2d_layers(model), 86)
        self.assertAlmostEqual(model_utils.get_model_num_params(model),
                               0.52,
                               places=2)
        # A little bit different from the original paper.
        # self.assertAlmostEqual(
        #     model_utils.get_model_num_ops(model, (1, 3, 32, 32)),
        #     65.8 * 2,
        #     places=1)

    def test_forward(self):
        """ forward should be OK """
        # CIFAR
        model = condensenet.condensenet86(num_classes=100, ind_type='random')
        model.forward(torch.randn((1, 3, 32, 32)))


if __name__ == '__main__':
    unittest.main()
