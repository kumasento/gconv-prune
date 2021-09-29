""" Pruning utilities functions.

Author: Ruizhe Zhao
Date: 13/02/2019
"""

import logging

import torch
import torch.nn as nn

from gumi.ops import *
from gumi.pruning import mask_utils

__all__ = ["prune_module"]


def prune_module(mod, G=0, MCPG=0, *args, **kwargs):
    """ Prune a single module.
  
    The content of module will be updated IN-PLACE.
  Args:
    mod(nn.Module)
    G(int): number of groups
  """
    assert isinstance(mod, nn.Module)

    if isinstance(mod, MaskConv2d):
        prune_mask_conv2d(mod, G=G, MCPG=MCPG, *args, **kwargs)
    else:
        raise RuntimeError(
            "Type of module cannot be recognized: {}".format(type(mod).__name__)
        )


def prune_mask_conv2d(
    mod, G=0, MCPG=0, *args, no_weight=False, keep_mask=False, **kwargs
):
    """ Prune a module of type MaskConv2d. """
    assert isinstance(mod, MaskConv2d)

    # fetch weight and compute MBM
    weight = mod.weight
    F, C = weight.shape[:2]

    # if MCPG is set and it is groupable, we should use a new G instead
    if MCPG > 0 and GroupConv2d.groupable(C, F, max_channels_per_group=MCPG):
        G = GroupConv2d.get_num_groups(C, F, max_channels_per_group=MCPG)

    # filter modules with channels not divisible by G
    if weight.shape[0] % G != 0 or weight.shape[1] % G != 0:
        logging.warn("G={} is not divisible by weight shape {}".format(G, weight.shape))
        return

    # compute mask and update the mask value
    # can be considered as pruned.
    mod.G = G

    if not keep_mask:
        mbm_mask = mask_utils.create_mbm_mask(weight, G, **kwargs)
        mod.mask.data = mbm_mask.data
        mod.weight.data[mod.mask.data == 0] = 0  # zero

    if no_weight:
        nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")

    # log
    logging.debug(
        "Module {} with weight shape {} has been pruned into G={}.".format(
            mod, weight.shape, G
        )
    )
