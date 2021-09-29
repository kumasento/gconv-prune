""" In this profiling example, we directly build a MobileNet
  and test its performance with different group configuration.
  NOTE: only speed is what we care for in this script.
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
from gumi.model_runner.parser import create_cli_parser
from gumi.model_runner import utils
from gumi.models.imagenet.mobilenet import MobileNet

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


def generate_sparse_matrix(num_rows, num_cols, groups, block=True):
    """ Generate test matrices """
    W = np.zeros((num_rows, num_cols), dtype=np.float32)

    if block:
        logging.info(
            "==> Generating block-sparse matrix {} x {} with G={} ...".format(
                num_rows, num_cols, groups
            )
        )
        rpg, cpg = num_rows // groups, num_cols // groups
        for g in range(groups):
            W[rpg * g : rpg * (g + 1), cpg * g : cpg * (g + 1)] = np.random.rand(
                rpg, cpg
            ).astype(np.float32)

        perm_rows = np.random.permutation(num_rows)
        perm_cols = np.random.permutation(num_cols)

        return W[perm_rows, :][:, perm_cols]
    else:
        nnz = num_rows * num_cols // groups

        logging.info(
            "==> Generating random sparse matrix {} x {} with {:.2f}% sparsity ...".format(
                num_rows, num_cols, 1.0 - nnz / (num_cols * num_rows)
            )
        )

        row_ind = np.array([], dtype=np.int32)
        col_ind = np.array([], dtype=np.int32)
        cur_nnz = nnz
        for i in range(num_rows):
            if cur_nnz == 0:
                break
            elif i == num_rows - 1:
                col_nnz = cur_nnz
            elif cur_nnz == 1:
                col_nnz = np.random.randint(0, 1)
            else:
                col_nnz = np.random.randint(
                    cur_nnz // (num_rows - i), min(num_cols, cur_nnz)
                )

            # print(num_cols, cur_nnz, col_nnz)
            cur_nnz -= col_nnz
            row_ind = np.concatenate((row_ind, np.ones(col_nnz, dtype=np.int32) * i))
            col_ind = np.concatenate((col_ind, np.arange(col_nnz)))

        W[(row_ind, col_ind)] = np.random.rand(nnz).astype(np.float32)

        perm_rows = np.random.permutation(num_rows)
        perm_cols = np.random.permutation(num_cols)

        return W[perm_rows, :][:, perm_cols]


def update_sparse_weight(model, block=True):
    for name, mod in model.named_modules():
        if isinstance(mod, SparseGroupConv2d):
            mod.update_weight(
                generate_sparse_matrix(
                    mod.out_channels, mod.in_channels, mod.groups, block=block
                )
            )


class Profiler(object):
    def __init__(self, args):
        self.args = args
        self.use_cuda = args.use_cuda
        self.iters = args.iters

    def profile_case(self, x, model, iters=None):
        """ Profile a single layer. W is input as a sparse tensor. """
        if iters is None:
            iters = self.iters

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

    def profile(self, iters, use_cuda=False):
        """ Run the profile function """
        groups = [1, 2, 4, 8, 16, 32, 64]
        sparse = (True, False)

        data = []
        columns = ["G", "is_sparse", "num_params", "num_ops", "time"]

        for g in groups:
            for is_sparse in sparse:
                if g != 1 and not is_sparse:
                    continue

                model = MobileNet(groups=g, sparse=is_sparse, num_channels=1000)
                update_sparse_weight(model, block=True)

                num_params = model_utils.get_model_num_params(model)
                num_ops = utils.get_model_num_ops(model, self.args.dataset)

                x = torch.rand((1, 3, 224, 224))

                if use_cuda:
                    x = x.cuda()
                    model.cuda()

                logging.info(
                    "==> Profiling G={} sparse={} for {} iters ...".format(
                        g, is_sparse, iters
                    )
                )
                elapsed = self.profile_case(x, model, iters=iters)

                logging.info(
                    "\t# params. {:.2f}M # ops {:.2f}M Time elapsed: {:.2f} ms ".format(
                        num_params, num_ops, elapsed
                    )
                )

                data.append([g, is_sparse, num_params, num_ops, elapsed])

        return pd.DataFrame(data, columns=columns)


def main():
    profiler = Profiler(args)
    df = profiler.profile(iters=args.iters, use_cuda=args.use_cuda)
    df.to_csv(
        "data/profile/mobilenet_v1{}.csv".format("_cuda" if args.use_cuda else ""),
        index=False,
    )


if __name__ == "__main__":
    main()
