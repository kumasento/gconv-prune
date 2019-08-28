""" Test model_utils. """

import unittest

import torch
import numpy as np

from gumi import model_utils
from gumi.ops import *


class TestModelUtils(unittest.TestCase):

  def test_get_weight_parameter(self):
    """ Check whether parameters can be get from Module. """
    self.assertIsNotNone(
        model_utils.get_weight_parameter(MaskConv2d(32, 32, 3)))

    weight_groups = model_utils.get_weight_parameter(
        GroupConv2d(32, 64, 3, groups=2))
    self.assertIsNotNone(weight_groups)
    self.assertIsInstance(weight_groups, torch.Tensor)
    self.assertEqual(weight_groups.shape[0], 64)
    self.assertEqual(weight_groups.shape[1], 16)

  def test_get_group_allocation(self):
    """ Test GSP based group allocation. """
    mask_conv = MaskConv2d(16, 32, 3)

    gaf, gac = model_utils.get_group_allocation(mask_conv, 2)
    self.assertIsNone(gaf)
    self.assertIsNone(gac)

    mask_conv.mask.data[:16, :8] = 0
    mask_conv.mask.data[16:, 8:] = 0

    gaf, gac = model_utils.get_group_allocation(mask_conv, 2)
    self.assertTrue(np.allclose(gaf[16:], np.ones(16)))
    self.assertTrue(np.allclose(gaf[:16], np.ones(16) * 2))
    self.assertTrue(np.allclose(gac[:8], np.ones(8)))
    self.assertTrue(np.allclose(gac[8:], np.ones(8) * 2))

  def test_is_gsp_satisfied(self):
    """ Test whether GSP can be detected. """
    mask_conv = MaskConv2d(32, 32, 3)

    self.assertTrue(model_utils.is_gsp_satisfied(mask_conv, 1))
    self.assertFalse(model_utils.is_gsp_satisfied(mask_conv, 2))

    mask_conv.mask.data[:16, :16] = 0
    mask_conv.mask.data[16:, 16:] = 0
    self.assertTrue(model_utils.is_gsp_satisfied(mask_conv, 2))

    mask_conv.mask.data[15, 15] = 1
    mask_conv.mask.data[15, 16] = 0
    mask_conv.mask.data[16, 15] = 1
    mask_conv.mask.data[16, 16] = 0
    self.assertFalse(model_utils.is_gsp_satisfied(mask_conv, 2))
    self.assertIsNone(model_utils.get_group_allocation(mask_conv, 2)[0])
    self.assertIsNone(model_utils.get_group_allocation(mask_conv, 2)[1])

  def test_get_group_parameters(self):
    """ Test getting groups of weight. """
    mask_conv = MaskConv2d(16, 32, 3)
    mask_conv.mask.data[:16, :8] = 0
    mask_conv.mask.data[16:, 8:] = 0

    W = mask_conv.weight.detach().numpy()
    wg, ind_in, ind_out = model_utils.get_group_parameters(mask_conv, 2)
    self.assertTrue(np.allclose(W[16:, :8, :, :], wg[:16, :, :, :]))
    self.assertTrue(np.allclose(W[:16, 8:, :, :], wg[16:, :, :, :]))

    # nothing changed for input channel sequence
    self.assertTrue(np.allclose(ind_in, np.arange(16)))
    self.assertTrue(np.allclose(ind_out[:16], np.arange(16, 32)))
    self.assertTrue(np.allclose(ind_out[16:], np.arange(16)))

  def test_get_group_parameters_in_out(self):
    """ Get indices for both in and out. """
    # a simple case
    mconv = MaskConv2d(4, 8, 3)
    mconv.mask.data.zero_()
    mconv.mask.data[:8:2, :4:2] = 1
    mconv.mask.data[1:8:2, 1:4:2] = 1

    G = 2
    wg, ind_in, ind_out = model_utils.get_group_parameters(mconv, G)

    # this is the correct order
    W = mconv.weight.detach().numpy()
    self.assertTrue(np.allclose(wg[:4, :, :, :], W[:8:2, :4:2, :, :]))
    self.assertTrue(np.allclose(wg[4:, :, :, :], W[1:8:2, 1:4:2, :, :]))
    self.assertTrue(np.allclose(ind_in, np.array([0, 2, 1, 3])))
    self.assertTrue(np.allclose(ind_out, np.array([0, 4, 1, 5, 2, 6, 3, 7])))


if __name__ == '__main__':
  unittest.main()