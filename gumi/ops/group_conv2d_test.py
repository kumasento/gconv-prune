""" Unit test for GroupConv2d. """

import unittest

import torch
import torch.nn as nn
import numpy as np

from gumi.ops.group_conv2d import GroupConv2d


class TestGroupConv2d(unittest.TestCase):
    def test_ctor(self):
        """ Simply test the constructor """
        group_conv2d = GroupConv2d(32, 32, 3, padding=1, groups=16)

        self.assertIsInstance(group_conv2d.indices, nn.Parameter)
        self.assertIsInstance(group_conv2d.conv2d, nn.Conv2d)
        # groups number of modules
        self.assertEqual(len(list(group_conv2d.children())), 1)
        # groups number of weights w/ indices
        self.assertEqual(len(list(group_conv2d.parameters())), 3)

    def test_create_indices_parameters(self):
        """ check whether indices are created correctly """
        gconv = GroupConv2d(2, 2, 3, padding=1, groups=1)

        self.assertIsInstance(gconv.ind_in, nn.Parameter)
        self.assertIsInstance(gconv.ind_out, nn.Parameter)
        # when passed as None
        self.assertTrue(torch.allclose(gconv.ind_in, torch.tensor([0, 1])))
        self.assertTrue(torch.allclose(gconv.ind_out, torch.tensor([0, 1])))
        # choose a different index mapping
        gconv = GroupConv2d(2, 2, 3, ind_in=[1, 0], ind_out=[1, 0], padding=1, groups=1)
        self.assertTrue(torch.allclose(gconv.ind_in, torch.tensor([1, 0])))
        self.assertTrue(torch.allclose(gconv.ind_out, torch.tensor([1, 0])))

    def test_forward_one_group(self):
        group_conv2d = GroupConv2d(2, 2, 3, padding=1, groups=1)
        conv2d = nn.Conv2d(2, 2, 3, padding=1, bias=False)

        # init both conv2d implementation by the same weight ndarray
        weight = torch.randn((2, 2, 3, 3))
        conv2d.weight.data = weight
        group_conv2d.conv2d.weight.data = weight

        x = torch.randn((1, 2, 4, 4))
        result = group_conv2d.forward(x)
        golden = conv2d.forward(x)

        self.assertTrue(torch.allclose(result, golden))

    def test_forward_multi_groups(self):
        group_conv2d = GroupConv2d(2, 2, 3, padding=1, groups=2)
        conv2d_a = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        conv2d_b = nn.Conv2d(1, 1, 3, padding=1, bias=False)

        # assign weights
        weight_a = torch.randn((1, 1, 3, 3))
        weight_b = torch.randn((1, 1, 3, 3))
        conv2d_a.weight.data = weight_a
        conv2d_b.weight.data = weight_b
        group_conv2d.conv2d.weight.data = torch.cat([weight_a, weight_b])

        x = torch.randn((1, 2, 4, 4))
        x_split = x.split(1, dim=1)
        result = group_conv2d.forward(x)
        golden = torch.cat(
            [conv2d_a.forward(x_split[0]), conv2d_b.forward(x_split[1])], dim=1
        )

        self.assertTrue(torch.allclose(result, golden))

    def test_forward_cuda(self):
        x = torch.randn((1, 32, 4, 4))

        group_conv2d = GroupConv2d(32, 32, 3, padding=1, groups=4)
        golden = group_conv2d.forward(x)  # CPU forward
        result = group_conv2d.cuda().forward(x.cuda())  # GPU forward

        self.assertTrue(torch.allclose(result, golden.cuda(), rtol=1e-02))

    def test_forward_with_indices(self):
        """ Specify output channel permutation and see whether the output
            are actually permuted. """
        x = torch.randn((1, 4, 4, 4))
        ind_out = [3, 0, 1, 2]

        group_conv2d = GroupConv2d(4, 4, 3, padding=1, groups=1, ind_out=ind_out)
        conv2d = nn.Conv2d(4, 4, 3, padding=1, bias=False)

        weight = torch.randn((4, 4, 3, 3))
        conv2d.weight.data = weight
        group_conv2d.conv2d.weight.data = weight

        result = group_conv2d.forward(x)
        golden = conv2d.forward(x)

        self.assertTrue(torch.allclose(result[:, 0, :, :], golden[:, 3, :, :]))
        self.assertTrue(torch.allclose(result[:, 1, :, :], golden[:, 0, :, :]))
        self.assertTrue(torch.allclose(result[:, 2, :, :], golden[:, 1, :, :]))
        self.assertTrue(torch.allclose(result[:, 3, :, :], golden[:, 2, :, :]))

    def test_get_num_groups(self):
        """ Test the get_num_groups function. """
        self.assertEqual(GroupConv2d.get_num_groups(16, 32, 8), 4)
        self.assertEqual(GroupConv2d.get_num_groups(64, 32, 8), 8)
        self.assertEqual(GroupConv2d.get_num_groups(16, 24, 6), 4)

        with self.assertRaises(AssertionError):
            GroupConv2d.get_num_groups(16, 24, 4)

    def test_groupable(self):
        """ Test the groupable function. """
        self.assertFalse(GroupConv2d.groupable(16, 32, groups=3))
        self.assertTrue(GroupConv2d.groupable(16, 32, groups=4))

        self.assertFalse(GroupConv2d.groupable(16, 32, max_channels_per_group=64))
        self.assertTrue(GroupConv2d.groupable(16, 32, max_channels_per_group=32))


if __name__ == "__main__":
    unittest.main()
