""" Utilities to build models.

Including - 
  - basic building blocks, e.g., conv3x3, conv1x1, etc.
"""
import functools
import os

import numpy as np
import torch
import torch.nn as nn
from thop import profile

from gumi.ops import (GroupConv2d, MaskConv2d, MMPointwiseConv2d,
                      SparseGroupConv2d)

__all__ = ["conv3x3", "conv1x1", "is_conv2d"]


def _get_indices(indices, channels, groups=None):
    """ Create a list of indices from a given instructor.

    - If indices is a list, return a list;
    - If indices is a str, return a list created based on the content
      of the string;
    - If indices is None, return None.

    Args:
      indices(object)
      channels(int): number of channels
    Returns:
      A list of indices.
    """
    if indices is None:
        return None

    if isinstance(indices, list):
        return indices

    if isinstance(indices, str):
        if indices == "random":
            return np.random.permutation(channels).tolist()

        if indices == "trans":
            assert isinstance(groups, int) and groups > 0

            perm = np.arange(channels)
            perm = perm.reshape((channels // groups, groups))
            perm = perm.T.reshape(channels)
            return perm.tolist()

        raise ValueError(
            "Indices pattern {} cannot be recognised.".format(indices))

    raise TypeError("Cannot recognise type of indices: {}".format(
        type(indices)))


def get_perm_indices(ind_type, in_channels, out_channels, groups=0):
    """ Create a pair of permutation indices.

      ind_type is a string that specifies how we create indices.
    """
    assert isinstance(ind_type, str)
    assert ind_type in ["shuffle", "random", "none"]

    if ind_type == "shuffle":
        ind_in = np.arange(in_channels).tolist()
        ind_out = _get_indices("trans", out_channels, groups=groups)
    elif ind_type == "random":
        ind_in = _get_indices("random", in_channels)
        ind_out = _get_indices("random", out_channels)
    elif ind_type == "none":
        ind_in = np.arange(in_channels).tolist()
        ind_out = np.arange(out_channels).tolist()

    return ind_in, ind_out


def get_conv2d_fn(in_planes,
                  out_planes,
                  groups=1,
                  max_channels_per_group=None,
                  ind_type=None,
                  mask=False,
                  **kwargs):
    """ Decide which Conv2d function to use. This function will only
    be utilized when doing """
    if mask or not GroupConv2d.groupable(
            in_planes,
            out_planes,
            groups=groups,
            max_channels_per_group=max_channels_per_group,
    ):  # if mask specified
        return functools.partial(MaskConv2d, in_planes, out_planes)

    # now we know Conv2d is groupable
    G = GroupConv2d.get_num_groups(in_planes,
                                   out_planes,
                                   max_channels_per_group,
                                   groups=groups)
    # maybe update the indices
    ind_in, ind_out = get_perm_indices(ind_type,
                                       in_planes,
                                       out_planes,
                                       groups=G)

    # all group parameters are wrapped
    return functools.partial(
        GroupConv2d,
        in_planes,
        out_planes,
        groups=groups,
        max_channels_per_group=max_channels_per_group,
        ind_in=ind_in,
        ind_out=ind_out,
    )


def conv3x3(in_planes, out_planes, stride=1, bias=False, **kwargs):
    """ 3x3 convolution with padding, support mask and group """
    return get_conv2d_fn(in_planes, out_planes, **kwargs)(kernel_size=3,
                                                          stride=stride,
                                                          padding=1,
                                                          bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False, **kwargs):
    """ 1x1 convolution """
    return get_conv2d_fn(in_planes, out_planes, **kwargs)(kernel_size=1,
                                                          stride=stride,
                                                          bias=bias)


def get_model_num_params(model):
    """ Compute the number of parameters. """
    num_params = 0
    filters = ["mask", "ind_in", "ind_out"]

    # HACK we need to get an accurate calculation of MaskConv2d
    mask_map = {}
    for name, mod in model.named_modules():
        if isinstance(mod, MaskConv2d):
            mask_map[name] = mod

    for name, p in model.named_parameters():
        if functools.reduce(lambda r, f: r or (f in name), filters, False):
            continue

        # HACK for sparse tensors
        if isinstance(p.data, torch.sparse.FloatTensor):
            num = p.data._nnz() / 1e6
        else:
            num = p.numel() / 1e6

        # HACK
        mod_name = name.replace(".weight", "")
        if mod_name in mask_map and mask_map[mod_name].G > 0:
            num /= mask_map[mod_name].G

        num_params += num

    return num_params


def get_model_num_ops(model, input_size):
    """ Return FLOPS. """
    flops, _ = profile(
        model,
        input_size=input_size,
        custom_ops={
            MaskConv2d: MaskConv2d.count_num_ops,
            SparseGroupConv2d: SparseGroupConv2d.count_num_ops,
            MMPointwiseConv2d: MMPointwiseConv2d.count_num_ops,
        },
        quiet=True,
    )

    return flops / 1e6


def is_conv2d(mod):
    """ Check whether a module can be seen as Conv2d """
    return isinstance(mod, nn.Conv2d) or isinstance(mod, MaskConv2d)


def get_num_conv2d_layers(model, exclude_downsample=True, include_linear=True):
    """ Check the number of Conv2D layers. """

    num = 0
    for n, m in model.named_modules():
        if "downsample" in n and exclude_downsample:
            continue
        if is_conv2d(m) or (include_linear and isinstance(m, nn.Linear)):
            num += 1

    return num


def load_checkpoint(checkpoint, model):
    """ Load checkpoint content from the given path. """
    if not os.path.isfile(checkpoint):
        raise RuntimeError(
            "Checkpoint should be a valid file: {}".format(checkpoint))
    if not isinstance(model, nn.Module):
        raise TypeError("model should be a valid nn.Module, got {}".format(
            type(model)))

    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt["state_dict"])


def get_num_classes(args):
    """ Get number of classes. """
    # TODO: args is something from the early_stage module.
    if args.dataset == "cifar10":
        return 10
    elif args.dataset == "cifar100":
        return 100
    raise ValueError("Cannot recognise dataset {}".format(args.dataset))


def get_weight_parameter(mod):
    """ Get the weight from a given module """
    if isinstance(mod, MaskConv2d):
        return mod.weight
    if isinstance(mod, GroupConv2d):
        return mod.conv2d.weight

    raise TypeError("Cannot recognise mod type: {}".format(type(mod)))
