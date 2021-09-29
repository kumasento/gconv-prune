""" Unit test for mask_utils. """

import unittest

import numpy as np
import torch
from gumi.pruning import mask_utils


class MaskUtilsTest(unittest.TestCase):
    """ Find the mask for pruning. """
    def test_get_criterion(self):
        """ Test get criterion """
        with self.assertRaises(AssertionError):
            mask_utils.get_criterion(1)
        with self.assertRaises(AssertionError):
            mask_utils.get_criterion(torch.randn((10, 10)))

        W = torch.randn((32, 32, 3, 3))
        self.assertTrue(
            torch.allclose(mask_utils.get_criterion(W), W.norm(dim=(2, 3))))

    def test_get_taylor_criterion(self):
        """ Test the performance of taylor criterion. """
        W = torch.randn((32, 32, 3, 3))
        W.grad = torch.randn((32, 32, 3, 3))

        C = mask_utils.get_taylor_criterion(W)
        self.assertTrue(len(C.shape) == 2)

    def test_get_sort_perm(self):
        """ get_sort_perm """
        W = torch.randn((32, 32, 3, 3))
        C = mask_utils.get_criterion(W)

        with self.assertRaises(AssertionError):
            mask_utils.get_sort_perm(C, dim=3)

        C = C.numpy()
        self.assertTrue(
            np.allclose(mask_utils.get_sort_perm(C),
                        np.argsort(C.sum(axis=1))))
        self.assertTrue(
            np.allclose(mask_utils.get_sort_perm(C, dim=1),
                        np.argsort(C.sum(axis=0))))

    def test_group_sort(self):
        """ recur_group_sort """
        F, C = 6, 6
        W = torch.randn((F, C, 3, 3))
        C_ = mask_utils.get_criterion(W)
        G = 3

        C_ = mask_utils._get_numpy(C_)
        print('')
        print(C_)

        ind_in, ind_out = mask_utils.group_sort(C_, G)

        print(C_[ind_out, :][:, ind_in])

    def test_permute_criterion(self):
        """ Test permuting criterion. """
        W = torch.randn((16, 32, 3, 3))
        C = mask_utils.get_criterion(W)

        with self.assertRaises(ValueError):
            mask_utils.permute_criterion(C, method='')

        C_ = C.numpy()
        self.assertTrue(
            np.allclose(C_,
                        mask_utils.permute_criterion(C, 'SAME')[0]))
        self.assertFalse(
            np.allclose(C_,
                        mask_utils.permute_criterion(C, 'SORT')[0]))

        C[:8, :16] += 100.
        C_ = C.numpy()
        CS, ind_in, ind_out = mask_utils.permute_criterion(C, 'SORT')
        # test sorted
        self.assertTrue((ind_in[:16] >= 16).all())
        self.assertTrue((ind_in[16:] < 16).all())
        self.assertTrue((ind_out[:8] >= 8).all())
        self.assertTrue((ind_out[8:] < 8).all())
        self.assertTrue(
            (CS[8:, 16:] > 100).all())  # the tensor has been sorted

    def test_create_mbm_mask(self):
        """ Check whether get_mbm_mask returns correct results."""
        F, C = 4, 8
        W = torch.randn((F, C, 3, 3))
        W[::2, ::2] += 100.0

        G = 4
        mask = mask_utils.create_mbm_mask(W, G)
        self.assertEqual(mask.numpy().shape, (F, C))
        self.assertEqual(mask.sum(), (F // G) * (C // G) * G)
        self.assertTrue((mask.sum(dim=1) == torch.ones(F) * (C // G)).all())
        self.assertTrue((mask.sum(dim=0) == torch.ones(C) * (F // G)).all())

        mask = mask_utils.create_mbm_mask(W, G, perm='SORT')
        self.assertEqual(mask.numpy().shape, (F, C))
        self.assertEqual(mask.sum(), (F // G) * (C // G) * G)
        self.assertTrue((mask.sum(dim=1) == torch.ones(F) * (C // G)).all())
        self.assertTrue((mask.sum(dim=0) == torch.ones(C) * (F // G)).all())

    def test_run_mbm(self):
        """ GRPS N_S=10"""
        torch.manual_seed(0)
        ten = torch.randn((32, 32, 3, 3))
        ten[0:8, 0:8] += 100.0
        ten[8:16, 8:16] += 100.0
        ten[16:24, 16:24] += 100.0
        ten[24:32, 24:32] += 100.0

        G = 4
        row_ind, col_ind, cost = mask_utils.run_mbm(ten, G)
        crit = ten[row_ind, col_ind, :, :].norm(dim=(2, 3))
        print(crit)
        # criterion should be around 300 in this case
        self.assertTrue(((crit - 300).abs() < 3).all())

    def test_run_mbm_with_perm(self):
        torch.manual_seed(0)
        W = torch.randn((16, 32, 3, 3))
        W[::2, ::2] += 100.0

        G = 4
        gnd_in1, gnd_out1, cost1 = mask_utils.run_mbm(W,
                                                      G,
                                                      perm=None,
                                                      num_iters=1)
        gnd_in2, gnd_out2, cost2 = mask_utils.run_mbm(W, G, perm='SORT')
        self.assertTrue(cost1 > cost2)

        C = mask_utils.get_criterion(W)
        C /= torch.norm(C)
        C = mask_utils._get_numpy(C)
        sum1 = 0
        for ind_in, ind_out in zip(gnd_in1, gnd_out1):
            sum1 += (C[ind_out, :])[:, ind_in].sum()
        self.assertTrue(np.allclose(C.sum() - sum1, cost1, rtol=1e-4))

        sum2 = 0
        for ind_in, ind_out in zip(gnd_in2, gnd_out2):
            sum2 += (C[ind_out, :])[:, ind_in].sum()
        self.assertTrue(np.allclose(C.sum() - sum2, cost2, rtol=1e-4))


if __name__ == '__main__':
    unittest.main()
