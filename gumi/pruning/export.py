""" Code migrated from evaluation/early_stage/group_exporter.py """
from collections import OrderedDict
import logging

logging.getLogger().setLevel(logging.DEBUG)

import torch
import torch.nn as nn

from gumi import model_utils
from gumi.ops import *


class GroupExporter(object):
    @staticmethod
    def export_to_group_conv2d(child):
        wg, ind_in, ind_out = model_utils.get_group_parameters(child, child.G)

        g_conv = GroupConv2d.create_from_mask_conv2d(child)
        g_conv.conv2d.weight.data = torch.from_numpy(wg).float()
        g_conv.ind_in.data = torch.from_numpy(ind_in).long()
        g_conv.ind_out.data = torch.from_numpy(ind_out).long()

        if g_conv.bias:
            g_conv.bias.data = child.bias
        return g_conv

    @staticmethod
    def export_to_sparse_pointwise(child):
        """ Use SparseGroupConv2d. """
        assert isinstance(child, MaskConv2d)
        assert child.kernel_size == 1
        assert child.stride[0] == 1
        assert child.padding[0] == 0
        assert child.G > 1
        assert child.bias is None

        mod = SparseGroupConv2d(
            child.in_channels, child.out_channels, 1, groups=child.G, bias=False
        )
        W = child.weight
        W = torch.mul(W.view(W.shape[0], W.shape[1]), child.mask)
        mod.update_weight(W)

        logging.info(
            "==> Weights sparsity: {:.2f}%".format(
                mod.weight._nnz() / mod.weight.numel() * 100
            )
        )

        return mod

    @staticmethod
    def export_to_mm_pointwise(child):
        """ Use MMPointwiseConv2d. """
        assert isinstance(child, MaskConv2d)
        assert child.kernel_size == 1
        assert child.stride[0] == 1
        assert child.padding[0] == 0
        assert child.bias is None

        mod = MMPointwiseConv2d(child.in_channels, child.out_channels, 1, bias=False)
        W = child.weight
        W = torch.mul(W.view(W.shape[0], W.shape[1]), child.mask)
        mod.weight.data = W
        # mod.update_weight(W)
        # logging.info('==> Weights sparsity: {:.2f}%'.format(
        #     mod.weight._nnz() / mod.weight.numel() * 100))

        return mod

    @staticmethod
    def export_to_std_conv2d(child, mm=False):
        if mm and child.kernel_size == 1 and child.stride[0] == 1:
            return GroupExporter.export_to_mm_pointwise(child)

        conv = nn.Conv2d(
            child.in_channels,
            child.out_channels,
            child.kernel_size,
            stride=child.stride,
            padding=child.padding,
            bias=child.bias,
        )

        conv.weight.data = torch.mul(
            child.weight, child.mask.view(*child.mask.shape, 1, 1)
        )
        if conv.bias:
            conv.bias.data = child.bias

        return conv

    @staticmethod
    def export(
        model, use_cuda=True, sparse=False, std=False, mm=False, min_sparse_channels=512
    ):
        """ Export function.
    
      Takes the model as input, returns a GroupConv2d based model

      TODO: this function is very similar with apply_mask
    """

        for name, mod in model.named_modules():
            name_to_mod = {}

            for child_name, child in mod.named_children():
                # we need to replace in this case
                if isinstance(child, MaskConv2d):
                    if child.G > 1:
                        assert len(list(child.children())) == 0

                        if (
                            sparse
                            and child.kernel_size == 1
                            and child.padding[0] == 0
                            and child.stride[0] == 1
                            and child.in_channels >= min_sparse_channels
                            and child.out_channels >= min_sparse_channels
                        ):
                            logging.info(
                                "==> Exporting mod {} to sparse pointwise ...".format(
                                    name + "." + child_name
                                )
                            )
                            mod_ = GroupExporter.export_to_sparse_pointwise(child)
                        elif std:
                            logging.info(
                                "==> Exporting mod {} to normal conv2d ...".format(
                                    name + "." + child_name
                                )
                            )
                            mod_ = GroupExporter.export_to_std_conv2d(child, mm=mm)
                        else:
                            logging.info(
                                "==> Exporting mod {} to group conv2d ...".format(
                                    name + "." + child_name
                                )
                            )
                            mod_ = GroupExporter.export_to_group_conv2d(child)

                        name_to_mod[child_name] = mod_
                    else:
                        # logging.info(
                        #     '==> Exporting mod {} to normal conv2d without groups ...'.
                        #     format(name + '.' + child_name))
                        conv = nn.Conv2d(
                            child.in_channels,
                            child.out_channels,
                            child.kernel_size,
                            stride=child.stride,
                            padding=child.padding,
                            bias=child.bias,
                        )

                        conv.weight.data = child.weight
                        if conv.bias:
                            conv.bias.data = child.bias

                        name_to_mod[child_name] = conv

            if not name_to_mod:
                continue

            for child_name, child in name_to_mod.items():
                mod._modules[child_name] = child

        # The replacement is done
        if torch.cuda.is_available() and use_cuda:
            model.cuda()

        return model
