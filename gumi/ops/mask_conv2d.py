""" Weight masked convolution.

Author: Ruizhe Zhao
Date: 2019/02/11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MaskConv2d"]


class MaskConv2d(nn.Module):
    """ The module of weight masked convolution.

    Attributes:
        conv2d(nn.Module): inner convolution
        mask(Tensor)
        G(int): number of groups
        fake_mask(bool): whether the mask will be ineffective or not
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        fake_mask=False,
        **kwargs
    ):
        """ CTOR. Parameters from nn.Conv2d """
        super().__init__()

        # NOTE: it will not be used to configure conv2d
        self.G = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # convolution parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # convolution coefficients
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.weight = nn.Parameter(
            torch.randn((out_channels, in_channels, kernel_size, kernel_size))
        )

        # create a mask variable
        # mask is initialised by all 1s, shouldn't be updated.
        self.mask = nn.Parameter(torch.ones(self.weight.shape[:2]), requires_grad=False)
        self.fake_mask = fake_mask

    @staticmethod
    def create_from_conv2d(conv2d, no_weight=False, use_cuda=True):
        """ Create MaskConv2d from Conv2d. """
        assert isinstance(conv2d, nn.Conv2d)
        assert conv2d.groups == 1  # not a group conv
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1]

        mask_conv = MaskConv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size[0],
            stride=conv2d.stride,
            padding=conv2d.padding,
            bias=conv2d.bias is not None,
        )

        # update data
        if no_weight:
            nn.init.kaiming_normal_(
                mask_conv.weight, mode="fan_out", nonlinearity="relu"
            )
        else:
            mask_conv.weight.data = conv2d.weight

        if conv2d.bias is not None:
            mask_conv.bias.data = conv2d.bias

        # assume we always put the new mask on CUDA
        if torch.cuda.is_available() and use_cuda:
            mask_conv.cuda()

        return mask_conv

    def apply_mask(self):
        """ Freeze the mask into weights to speed up. """
        if self.fake_mask:
            self.fake_mask = False

    @staticmethod
    def count_num_ops(mod, x, y):
        """ Count the total number of operators. """
        x = x[0]

        C_in = mod.in_channels
        C_out = mod.out_channels
        K = mod.kernel_size

        out_h = y.size(2)
        out_w = y.size(3)

        # compute the number of operations per element
        NUM_MAC_OPS = 2
        kernel_ops = NUM_MAC_OPS * K * K * C_in
        bias_ops = 1 if mod.bias is not None else 0
        ops_per_element = kernel_ops + bias_ops

        total_ops = out_w * out_h * C_out * ops_per_element

        nnz = mod.mask.nonzero().size(0)
        if mod.G > 0:
            total_ops /= mod.G

        # in case same conv is used multiple times
        mod.total_ops += torch.Tensor([int(total_ops)])

    def forward(self, x):
        """ Actual computation """

        # masking, using in-place multiplication, change mask's view
        weight = self.weight
        if not self.fake_mask:
            mask_ = self.mask.view(*self.mask.shape, 1, 1)
            # NOTE: just mask the weight in-place
            # weight = torch.mul(weight, mask_)
            weight.data.mul_(mask_)

        # use the functional API
        return F.conv2d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
