""" Sparse model downloaded from https://s3-us-west-1.amazonaws.com/nndistiller/agp-pruning/mobilenet/checkpoint.pth.tar

python profiling/profile_sparse_mobilenet.py -a mobilenet -d imagenet --dataset-dir $NAS_HOME/datasets/ILSVRC2012 --resume $ICCV19/sparse/mobilenet_distiller

"""

import os
import sys
import argparse
import copy
import time
import shutil
import json
import logging
import functools

logging.getLogger().setLevel(logging.DEBUG)

import pandas as pd
import numpy as np
import torch
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from gumi import model_utils
from gumi.ops import *
from gumi.model_runner.model_runner import ModelRunner
from gumi.model_runner.parser import create_cli_parser

parser = create_cli_parser(prog="CLI tool for profiling GConv models.")
parser.add_argument(
    "--iters", type=int, default=10000, help="Number of profiling iterations"
)
parser.add_argument(
    "--use-cuda", action="store_true", default=False, help="Whether to use GPU"
)
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
cudnn.benchmark = True


class SparsityCalculator(object):
    @staticmethod
    def get_model_sparsity(model, threshold=1e-5, level=None):
        """ Compute the total sparsity of a model """
        sum_nnz = 0
        sum_total = 0  # total elements

        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Conv2d):
                continue

            nnz, total = SparsityCalculator.get_sparsity(
                mod, threshold=threshold, level=level
            )
            sum_nnz += nnz
            sum_total += total

        return float(sum_nnz) / sum_total

    @staticmethod
    def get_sparsity(mod, threshold=1e-5, level=None):
        """ Compute the sparsity of a module. """
        assert isinstance(mod, nn.Conv2d)

        W = mod.weight.cpu().detach().numpy()
        if level is None or level == "elem":
            nz = (np.abs(W) < threshold).sum()
            total = W.size
        elif level == "kern":
            nz = (np.sqrt((W ** 2).sum(axis=2).sum(axis=2)) < threshold).sum()
            total = W.shape[0] * W.shape[1]
        else:
            raise ValueError("Level {} cannot be recognised.".format(level))

        return total - nz, total


class SparsifyDistillerRunner(ModelRunner):
    """ Try to make the MobileNet trained by Distiller sparsified. """

    def validate_args(self, args):
        pass

    def get_sparsity(self, model):
        """ Turn the original model into a sparsified one. """
        data = []
        columns = [
            "Name",
            "Elem NNZ",
            "Elem Total",
            "Elem (%)",
            "Kern NNZ",
            "Kern Total",
            "Kern (%)",
        ]

        # let's compute their NNZ
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d):
                logging.info('==> Working on module "{}" ...'.format(name))

                e_nnz, e_tot = SparsityCalculator.get_sparsity(mod, level="elem")
                logging.info(
                    "==> Element-wise sparsity: nnz={} total={} sparsity={:.2f}%".format(
                        e_nnz, e_tot, 100 - e_nnz / e_tot * 100
                    )
                )

                k_nnz, k_tot = SparsityCalculator.get_sparsity(mod, level="kern")
                logging.info(
                    "==> Kernel-wise sparsity: nnz={} total={} sparsity={:.2f}%".format(
                        k_nnz, k_tot, 100 - k_nnz / k_tot * 100
                    )
                )

                data.append(
                    [
                        name,
                        e_nnz,
                        e_tot,
                        100 - e_nnz / e_tot * 100,
                        k_nnz,
                        k_tot,
                        100 - k_nnz / k_tot * 100,
                    ]
                )

        df = pd.DataFrame(data, columns=columns)
        return df

    def sparsify(self, model):
        """ iterate every pointwise module, turn them into SparsePointwise """

        for name, mod in model.named_modules():
            if "layer4" not in name and "layer5" not in name:
                continue

            name_to_mod = {}
            for child_name, child in mod.named_children():
                if isinstance(child, nn.Conv2d) and child.kernel_size[0] == 1:

                    logging.info(
                        "==> Sparsifying module {} ...".format(name + "." + child_name)
                    )

                    sp_conv = SparseGroupConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size[0],
                        stride=child.stride[0],
                        padding=child.padding[0],
                        bias=child.bias is not None,
                    )
                    sp_conv.update_weight(child.weight.data, threshold=None)

                    logging.info("NNZ: {}".format(sp_conv.weight._nnz()))
                    name_to_mod[child_name] = sp_conv

            for child_name, child_mod in name_to_mod.items():
                mod._modules[child_name] = child_mod


def create_update_state_dict_fn():
    def update_state_dict(state_dict):
        """ Here are several update rules:
    
      - In this new script, we won't have "module." prefix
      - There won't be any '.conv2d' in the module
    """
        state_dict_ = copy.deepcopy(state_dict)

        for key, val in state_dict.items():
            key_ = key

            if "module" in key_:
                del state_dict_[key_]
                key_ = key_.replace("module.", "")
                state_dict_[key_] = val

        return state_dict_

    return update_state_dict


class Profiler(object):
    def __init__(self, args):
        self.args = args
        self.use_cuda = args.use_cuda
        self.iters = args.iters

    def profile_case(self, x, model, iters=None):
        """ Profile a single layer. W is input as a sparse tensor. """
        if iters is None:
            iters = self.iters

        model.eval()
        # dry run
        model.forward(x)

        if not self.use_cuda:
            start = time.time()
            for _ in range(iters):
                model.forward(x)
            end = time.time()
            elapsed = end - start
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(iters):
                model.forward(x)
            end.record()

            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) * 1e-3

        return elapsed * 1e3 / iters


def main():
    sdr = SparsifyDistillerRunner(args)
    model = sdr.load_model(
        update_state_dict_fn=create_update_state_dict_fn(), use_cuda=args.use_cuda
    )

    df = sdr.get_sparsity(model)
    print(df)
    print(
        "Elem-wise sparsity: {:.2f}%".format(
            (1 - df["Elem NNZ"].sum() / df["Elem Total"].sum()) * 100
        )
    )
    print(
        "Kern-wise sparsity: {:.2f}%".format(
            (1 - df["Kern NNZ"].sum() / df["Kern Total"].sum()) * 100
        )
    )

    sdr.sparsify(model)

    x = torch.randn((1, 3, 224, 224))

    if args.use_cuda:
        x = x.cuda()
        model.cuda()
    else:
        x = x.cpu()
        model.cpu()

    logging.info("==> Start profiler ...")
    profiler = Profiler(args)
    elapsed = profiler.profile_case(x, model, iters=args.iters)

    logging.info("==> Elapsed: {:.2f} ms".format(elapsed))
    # sdr.validate(model)


if __name__ == "__main__":
    main()
