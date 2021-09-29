""" Unit test for prune_utils. """

import unittest

import torch

from gumi.ops.mask_conv2d import MaskConv2d
from gumi.pruning import prune_utils


class PruneUtilsTest(unittest.TestCase):
    def test_prune_module(self):
        conv = MaskConv2d(32, 32, 3)
        G = 4
        prune_utils.prune_module(conv, G)

        self.assertFalse(torch.allclose(conv.mask, torch.ones(conv.mask.shape)))


if __name__ == "__main__":
    unittest.main()
