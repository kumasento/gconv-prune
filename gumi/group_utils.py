""" Group convolution related utilities. """

import os
import functools

import torch
import torch.nn as nn
import numpy as np

from gumi.ops import MaskConv2d, GroupConv2d

__all__ = [
    'get_group_allocation',
    'is_gsp_satisfied',
    'get_group_parameters',
    'get_group_param',
]


def get_group_allocation(mask, G):
  """ Allocate group ID to each kernel. """
  F, C = mask.shape

  gaf = np.zeros(F, dtype=np.uint8)  # filter alloc
  gac = np.zeros(C, dtype=np.uint8)  # channel alloc
  gid = 1  # current group ID

  for c in range(C):
    for f in range(F):
      # f, c is the leading kernel of the current group
      if (mask[f][c] == 0 or
          (gaf[f] != 0 and gac[c] != 0)):  # already allocated
        continue

      # allocate along the filter and the channel
      # NOTE if there is any conflict, we will find out in the end
      gaf[mask[:, c] > 0] = gid
      gac[mask[f, :] > 0] = gid

      gid += 1  # update group

  if gid != G + 1:  # insufficient splits
    return None, None

  # check the allocation
  for i in range(G):
    if (np.sum(gaf == i + 1) != F // G or np.sum(gac == i + 1) != C // G):
      return None, None

  return gaf, gac


def is_gsp_satisfied(mod, G):
  """ Check whether the mask from MaskConv2D satisfies GSP. """
  if not isinstance(mod, MaskConv2d):
    raise TypeError('You should provide a MaskConv2D module, got {}'.format(
        type(mod)))

  mask = mod.mask
  F, C = mask.shape

  if ((mask.sum(dim=0) == F // G).all().item() != 1 or
      (mask.sum(dim=1) == C // G).all().item() != 1):
    return False

  gaf, gac = get_group_allocation(mod, G)
  if gaf is None or gac is None:  # cannot find valid allocation
    return False

  return True


def get_group_parameters(mod, G):
  """ Create weight groups from MaskConv2d. """
  if not isinstance(mod, MaskConv2d):
    raise TypeError('You should provide a MaskConv2D module, got {}'.format(
        type(mod)))

  # NOTE no bias involved
  weight = mod.weight.detach().cpu().numpy()
  mask = mod.mask.detach().cpu().numpy()

  return get_group_param(weight, mask, G)


def get_group_param(weight, mask, G):
  """ Create weight groups from MaskConv2d. """
  F, C, K = weight.shape[:3]

  # collect the group
  weight_group = np.zeros((F, C // G, K, K))
  fg, cg = F // G, C // G
  gaf, gac = get_group_allocation(mask, G)

  # NOTE: this mapping won't be affected by the permutation
  for g in range(G):
    wg = weight[gaf == (g + 1), :, :, :]  # select weight group
    wg = wg[:, gac == (g + 1), :, :]
    weight_group[g * fg:(g + 1) * fg, :, :, :] = wg

  # get the permutation indices
  ind_in = np.zeros(C, dtype=np.int32)
  ind_out = np.zeros(F, dtype=np.int32)
  for g in range(G):
    ind_in[g * cg:(g + 1) * cg] = np.where(gac == (g + 1))[0]
    ind_out[gaf == (g + 1)] = np.arange(g * fg, (g + 1) * fg)

  return weight_group, ind_in, ind_out