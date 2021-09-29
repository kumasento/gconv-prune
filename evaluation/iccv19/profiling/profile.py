""" The minimalist profiling code. """

import os
import sys
import argparse
import copy
import time
import shutil
import json
import logging

logging.getLogger().setLevel(logging.DEBUG)

import numpy as np
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from gumi import model_utils
from gumi.ops import *
from gumi.pruning.export import GroupExporter
from gumi.model_runner import utils
from gumi.model_runner.model_runner import ModelRunner
from gumi.model_runner.parser import create_cli_parser

parser = create_cli_parser(prog="CLI tool for profiling GConv models.")
parser.add_argument(
    "--iters", type=int, default=100, help="Number of profiling information"
)
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Whether to use CUDA"
)
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
cudnn.benchmark = True


def main():
    use_cuda = not args.cpu

    if args.pretrained:
        logging.info("==> Loading pre-trained model ...")
        model = utils.load_model(
            args.arch, args.dataset, pretrained=args.pretrained, use_cuda=use_cuda
        )
    else:
        logging.info("==> Loading GConv model directly from Pickle ...")
        model = torch.load(args.resume)
        if not use_cuda:
            model.cpu()

        model.eval()
        logging.info("==> Sparsify model ...")
        utils.apply_sparse(model)
        print(model)

        logging.debug(
            "Total params: {:.2f}M FLOPS: {:.2f}M".format(
                model_utils.get_model_num_params(model),
                utils.get_model_num_ops(model, args.dataset),
            )
        )

    if args.dataset in utils.IMAGENET_DATASETS:
        x = torch.rand((args.test_batch, 3, 224, 224))
    else:
        x = torch.rand((args.test_batch, 3, 32, 32))

    if not use_cuda:
        x = x.cpu()

    # setup the input and model
    if use_cuda:
        x = x.cuda()
        model.cuda()

    logging.info("==> Dry running ...")
    dry_run_iters = 10 if not use_cuda else 100
    for _ in range(dry_run_iters):
        y = model.forward(x)

    logging.info("==> Print profiling info ...")
    with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
        y = model.forward(x)
    print(prof)

    # start timing
    logging.info("==> Start timing ...")
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(args.iters):
            y = model.forward(x)
        end.record()

        # synchronize
        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end)
    else:
        start = time.time()
        for _ in range(args.iters):
            y = model.forward(x)
        end = time.time()
        elapsed = (end - start) * 1e3

    print(
        "Elapsed time: {:10.2f} sec (total) {:6.2f} ms (per run) {:6.2f} FPS.".format(
            elapsed * 1e-3,
            elapsed / args.iters,
            args.iters * args.test_batch / elapsed * 1e3,
        )
    )


if __name__ == "__main__":
    main()
