""" Implementation of group conv2d.

Author: Ruizhe Zhao
Date: 2019/01/08
"""

import numpy as np
import torch
import torch.nn as nn
import torch.sparse

from gumi.ops.mask_conv2d import MaskConv2d

__all__ = [
    'GroupConv2d', 'GroupPointwiseConv', 'SparseGroupConv2d',
    'MMPointwiseConv2d'
]


class GroupConv2d(nn.Module):
  """ Group convolution with channel permutation.

  This Module directly implements the GConv operator used in our paper, which
  permutes input and output channels based on specific mappings.
  """

  def __init__(self,
               in_planes,
               out_planes,
               kernel_size,
               stride=1,
               padding=0,
               groups=1,
               ind_in=None,
               ind_out=None,
               bias=False,
               **kwargs):
    """ CTOR. """
    super(GroupConv2d, self).__init__()

    # setup parameters
    self.in_planes = in_planes
    self.out_planes = out_planes
    self.stride = stride
    self.padding = padding
    self.kernel_size = kernel_size
    self.groups = groups

    # extract number of groups from groups parameter
    assert isinstance(self.groups, int) and self.groups > 0
    assert in_planes % self.groups == 0
    assert out_planes % self.groups == 0

    # create indices for channel permutation
    self.ind_in = self.create_indices_parameter(self.in_planes, indices=ind_in)
    self.ind_out = self.create_indices_parameter(self.out_planes,
                                                 indices=ind_out)

    # initialise all groups
    self.setup_conv2d(self.groups)
    # the bias term
    self.bias = nn.Parameter(torch.randn(out_planes)) if bias else None

  @property
  def indices(self):
    """ To be compatible with legacy codes. """
    return self.ind_out

  @staticmethod
  def create_indices_parameter(num_planes, indices=None):
    """ Create indices as nn.Parameter.

    If indices is provided as None, we create a indices by their original
    order, by the `range` method. 
    """
    if indices is None:
      indices = list(range(num_planes))
    assert isinstance(indices, list)

    return nn.Parameter(data=torch.LongTensor(indices), requires_grad=False)

  @staticmethod
  def create_from_mask_conv2d(m_conv):
    """ """
    assert isinstance(m_conv, MaskConv2d)

    return GroupConv2d(m_conv.in_channels,
                       m_conv.out_channels,
                       m_conv.kernel_size,
                       padding=m_conv.padding,
                       stride=m_conv.stride,
                       groups=m_conv.G,
                       bias=m_conv.bias is not None)

  @staticmethod
  def get_num_groups(in_channels,
                     out_channels,
                     max_channels_per_group,
                     groups=1):
    """ Get the number of groups.
    
    G = max(C_in, C_out) / C_max 
    """
    if not max_channels_per_group:
      return groups

    assert isinstance(max_channels_per_group, int)
    max_c = max(in_channels, out_channels)
    min_c = min(in_channels, out_channels)
    assert max_c % max_channels_per_group == 0

    G = max_c // max_channels_per_group
    assert min_c % G == 0

    return G

  @staticmethod
  def groupable(in_channels,
                out_channels,
                groups=1,
                max_channels_per_group=None):
    """ Check whether the provided configuration can be turned into a group convolution. """
    if max_channels_per_group:  # overrides groups
      try:
        GroupConv2d.get_num_groups(in_channels, out_channels,
                                   max_channels_per_group)
        return True
      except AssertionError:
        return False
    else:
      # Here we only check by the provided groups value,
      # which can be a value smaller than 2, or larger than 2 but
      # not divisible. These two cases should be rejected.
      return (groups > 1 and in_channels % groups == 0 and
              out_channels % groups == 0)

  def setup_conv2d(self, G):
    """ Updated the number of groups and build conv2d. """
    self.conv2d = nn.Conv2d(self.in_planes,
                            self.out_planes,
                            self.kernel_size,
                            groups=G,
                            stride=self.stride,
                            padding=self.padding,
                            bias=False)

  def forward(self, x):
    """ Forward inference """
    channel_axis = 1

    x = torch.index_select(x, channel_axis, self.ind_in)  # indexing input
    y = self.conv2d(x)  # CORE convolution
    y = torch.index_select(y, channel_axis, self.ind_out)  # indexing output

    if self.bias is not None:
      y += self.bias

    return y


class GroupPointwiseConv(nn.Sequential):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               groups,
               padding=0,
               stride=1,
               bias=True):
    """ Normal group convolution followed by pointwise """
    super().__init__()
    self.add_module(
        'g_conv',
        nn.Conv2d(in_channels,
                  in_channels,
                  kernel_size,
                  stride=stride,
                  padding=padding,
                  bias=False,
                  groups=groups))
    self.add_module(
        'p_conv',
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias))


class SparseGroupConv2d(nn.Module):
  """ This module uses SpMM to implement GroupConv2d. """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               groups=1,
               bias=None,
               stride=1,
               padding=0):
    super().__init__()

    assert kernel_size == 1
    assert stride == 1
    assert padding == 0
    assert not bias

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.groups = groups
    self.stride = stride
    self.padding = padding

    self.weight = nn.Parameter(
        torch.sparse.FloatTensor(self.out_channels, self.in_channels))

  def update_weight(self, W, threshold=None):
    """ With the given W, use it to update self.weight """
    if isinstance(W, np.ndarray):
      # nullify
      if isinstance(threshold, float):
        W[np.abs(W) < threshold] = 0.0
      row_ind, col_ind = np.nonzero(W)
      i = torch.LongTensor([row_ind.tolist(), col_ind.tolist()])
      v = torch.FloatTensor(W[(row_ind, col_ind)])
    elif isinstance(W, torch.Tensor):
      # nullify
      W = W.view([W.shape[0], W.shape[1]])  # will raise error

      if isinstance(threshold, float):
        print(torch.abs(W) < threshold)
        W[torch.abs(W) < threshold] = 0.0

      i = torch.nonzero(W).t()
      v = W[i[0], i[1]]
    else:
      raise TypeError('W should be ndarray or Tensor')

    # we basically create a new Tensor for the parameter
    self.weight.data = torch.sparse.FloatTensor(i, v,
                                                torch.Size(W.shape)).coalesce()
    assert self.weight.data.is_coalesced()

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
    # bias_ops = 1 if mod.bias is not None else 0
    ops_per_element = kernel_ops  #  + bias_ops

    total_ops = out_w * out_h * C_out * ops_per_element
    total_ops = total_ops / (C_in * C_out) * mod.weight._nnz()

    # in case same conv is used multiple times
    mod.total_ops += torch.Tensor([int(total_ops)])

  def forward(self, x):
    """ update the view of weight and x.
    NOTE: number of pixels doesn't change, batch size is always 1
    """
    num_pixels = x.shape[2] * x.shape[3]
    x_ = x.view([self.in_channels, num_pixels])
    y_ = torch.sparse.mm(self.weight, x_)
    return y_.view([1, self.out_channels, x.shape[2], x.shape[3]])


class MMPointwiseConv2d(nn.Module):
  """ matrix-multiply based pointwise conv2d. """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               groups=1,
               bias=True,
               stride=1,
               padding=0):
    super().__init__()

    assert kernel_size == 1  # select from where to get the group size
    assert stride == 1
    assert padding == 0

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.groups = groups
    self.stride = stride
    self.padding = padding

    self.weight = nn.Parameter(
        torch.FloatTensor(self.out_channels, self.in_channels))

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
    # bias_ops = 1 if mod.bias is not None else 0
    ops_per_element = kernel_ops  #  + bias_ops

    total_ops = out_w * out_h * C_out * ops_per_element

    # in case same conv is used multiple times
    mod.total_ops += torch.Tensor([int(total_ops)])

  def forward(self, x):
    """ update the view of weight and x.
    NOTE: number of pixels doesn't change, batch size is always 1
    """
    num_pixels = x.shape[2] * x.shape[3]
    x_ = x.view([self.in_channels, num_pixels])
    y_ = torch.mm(self.weight, x_)
    return y_.view([1, self.out_channels, x.shape[2], x.shape[3]])
