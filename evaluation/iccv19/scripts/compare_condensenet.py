""" Check whether the CondenseNet produced by our definition
  is the same as the original one. """

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
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.onnx
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from gumi import model_utils
from gumi.ops import *
from gumi.model_runner import utils
from gumi.model_runner.model_runner import ModelRunner
from gumi.model_runner.parser import create_cli_parser

# model will be loaded from args.resume
# and exported to args.checkpoint
parser = create_cli_parser(prog="CLI tool for comparing CondenseNet")
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True


class CondenseNetRunner(ModelRunner):
    def validate_args(self, args):
        pass


def update_state_dict(state_dict):
    state_dict_ = copy.deepcopy(state_dict)

    for key, val in state_dict.items():
        key_ = key

        if "module." in key_:
            del state_dict_[key_]
            key_ = key_.replace("module.", "")
            state_dict_[key_] = val

        # map conv2d
        for i in (1, 2):
            if "conv_{}.conv".format(i) in key_:
                del state_dict_[key_]
                key_ = key_.replace("conv_{}.conv".format(i), "conv{}".format(i))
                state_dict_[key_] = val
            if "conv_{}.norm".format(i) in key_:
                del state_dict_[key_]
                key_ = key_.replace("conv_{}.norm".format(i), "bn{}".format(i))
                state_dict_[key_] = val

    return state_dict_


def main():
    logging.info("==> Loading from the original CondenseNet {} ...".format(args.resume))
    runner = CondenseNetRunner(args)
    model = runner.load_model(update_state_dict_fn=update_state_dict)


if __name__ == "__main__":
    main()
