""" Profile a single GConv layer """

import os
import sys
import argparse
import copy
import time
import shutil
import json
import logging

logging.getLogger().setLevel(logging.DEBUG)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from gumi import model_utils
from gumi.ops import *
from gumi.model_runner.parser import create_cli_parser

parser = create_cli_parser(prog="CLI tool for profiling GConv models.")
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
cudnn.benchmark = True


class Profiler(object):
    def __init__(self, args):
        self.args = args

    def get_conv_module(
        self, in_channels, out_channels, kernel_size, groups=1, mode=None, **kwargs
    ):
        """ Create convolution modules based on different configurations. """
        if mode is None or mode == "conv":
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                groups=groups,
                bias=False,
                **kwargs
            )
        elif mode == "gconv":
            return GroupConv2d(
                in_channels,
                out_channels,
                kernel_size,
                groups=groups,
                bias=False,
                **kwargs
            )
        elif mode == "mconv":
            return MaskConv2d(
                in_channels,
                out_channels,
                kernel_size,
                groups=groups,
                bias=False,
                **kwargs
            )
        elif mode == "pwise":
            return GroupPointwiseConv(
                in_channels,
                out_channels,
                kernel_size,
                groups=groups,
                bias=False,
                **kwargs
            )
        else:
            raise ValueError("mode={} cannot be recognised".format(mode))

    def run_layer(
        self,
        in_channels,
        *args,
        batch_size=32,
        in_size=224,
        use_cuda=True,
        iters=10000,
        **kwargs
    ):
        """ First create module, then run it for given iterate times """
        mod = self.get_conv_module(in_channels, *args, **kwargs)
        x = torch.rand((batch_size, in_channels, in_size, in_size))

        if use_cuda:
            mod.cuda()
            x = x.cuda()

        logging.info(
            "==> Start timing of {in_channels:} x {out_channels:} x {kernel_size:} G={groups:} in mode={mode:} ...".format(
                in_channels=in_channels,
                out_channels=args[0],
                kernel_size=args[1],
                groups=kwargs.get("groups"),
                mode=kwargs.get("mode"),
            )
        )
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(iters):
                mod.forward(x)
            end.record()

            # synchronize
            torch.cuda.synchronize()

            elapsed = start.elapsed_time(end)
        else:
            start = time.time()
            for _ in range(iters):
                mod.forward(x)
            end = time.time()
            elapsed = (end - start) * 1e3

        print(
            "Elapsed time: {:10.2f} sec (total) {:6.2f} ms (per run) {:6.2f} FPS.".format(
                elapsed * 1e-3, elapsed / iters, iters * batch_size / elapsed * 1e3
            )
        )

        del mod
        del x

        return elapsed / iters

    def run(self):
        """ Iterate every layer given. """
        # setup
        scales = [
            (64, 56),
            (128, 28),
            (256, 14),
            (512, 7),
        ]
        groups = (1, 2, 4, 8, 16, 32, 64)
        modes = ("conv", "gconv", "mconv", "pwise")
        # collect run-time
        data = []
        columns = ["Channels", "Scale", "G", "Mode", "Time"]

        for channels, in_size in scales:
            for g in groups:
                for mode in modes:
                    elapsed_ms = self.run_layer(
                        channels,
                        channels,
                        3,
                        in_size=in_size,
                        padding=1,
                        groups=g,
                        mode=mode,
                    )
                    data.append([channels, in_size, g, mode, elapsed_ms])

        return pd.DataFrame(data, columns=columns)


def main():
    profiler = Profiler(args)

    df = profiler.run()
    df.to_csv("data/profile/resnet50.csv", index=False)


if __name__ == "__main__":
    main()
