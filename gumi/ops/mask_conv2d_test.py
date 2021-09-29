""" Unit test for MaskConv2d. """

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gumi.ops.mask_conv2d import MaskConv2d


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()

        # suppose the input image has shape 32 x 4 x 4
        self.conv = MaskConv2d(32, 32, 3)
        self.fc = nn.Linear(32 * 2 * 2, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 2 * 2)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def find_param(model, name):
    """ Find a parameter by name """
    for p in model.named_parameters():
        if p[0] == name:
            return p[1]


class TestMaskConv2d(unittest.TestCase):
    def test_ctor(self):
        # now mask is all ONE
        mask_conv2d = MaskConv2d(32, 32, 3, bias=False)
        conv2d = nn.Conv2d(32, 32, 3, bias=False)

        # assign the same weight
        weight = torch.randn((32, 32, 3, 3))
        conv2d.weight.data = weight
        mask_conv2d.weight.data = weight

        # check equivalence
        x = torch.randn((1, 32, 4, 4))
        result = mask_conv2d.forward(x)
        golden = conv2d.forward(x)

        self.assertTrue(torch.allclose(result, golden))

    def test_params(self):
        mask_conv2d = MaskConv2d(32, 32, 3)
        self.assertEqual(len(list(mask_conv2d.parameters())), 2)
        mask_conv2d = MaskConv2d(32, 32, 3, bias=True)
        self.assertEqual(len(list(mask_conv2d.parameters())), 3)

    def test_mask_assign(self):
        """ Assign value to the internal mask. """
        mask_conv2d = MaskConv2d(32, 32, 3, bias=False)
        mask_conv2d.mask.data = torch.zeros((32, 32))
        x = torch.randn((1, 32, 4, 4))
        result = mask_conv2d.forward(x)

        self.assertTrue(torch.allclose(result, torch.zeros(result.shape)))

    def test_grad_update(self):
        """ Mask should not be updated by gradients.
    
      This test simply creates a CNN and runs training.
    """
        model = TestNet()
        model.train()

        before = find_param(model, "conv.mask")

        optimizer = optim.SGD(model.parameters(), lr=0.1)

        x = torch.randn((1, 32, 4, 4))
        output = model(x)
        target = torch.randint(0, 10, (1,)).long()

        loss = F.nll_loss(output, target)
        loss.backward()

        # run one step of optimization
        optimizer.step()

        after = find_param(model, "conv.mask")
        self.assertTrue(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
