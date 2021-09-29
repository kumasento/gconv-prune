""" Pruning a pre-trained model by GSP.

Author: Ruizhe Zhao
Date: 12/02/2019

The work-flow of this script:
- load a pre-trained model (suffixed by 'm')
- compute the mask based on weights
- fine-tune the model

"""

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
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from gumi.ops.mask_conv2d import MaskConv2d
from gumi.pruning import prune_utils
from gumi import model_utils
from gumi import models  # a module contains all supported models

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

import cifar_utils
from utils import *  # import utilities provided by pytorch-classification
from parser import create_parser  # argument parser for evaluation tasks
from pruner import Pruner

parser = create_parser()
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True


def write_summary(args, file_name="summary.json", **kwargs):
    """ Write summary to a JSON file. """
    summary_file = "{}/{}".format(args.checkpoint, file_name)
    with open(summary_file, "w") as f:
        json.dump(kwargs, f)


def main():
    # initialize the pruner
    pruner = Pruner(args)
    # pruner.prune(args.checkpoint)
    pruner.evaluate()

    # Run regularization
    pruner.prune(
        args.checkpoint, fake_mask=True, perm=args.perm, num_iters=args.num_sort_iters
    )
    pruner.evaluate()
    pruner.regularize()
    pruner.apply_mask()
    pruner.evaluate()

    logging.debug("Fine-tuning model for {} epochs".format(args.epochs))
    best_acc = pruner.fine_tune(args.epochs)
    logging.debug("Fine-tuned model")
    pruner.evaluate()

    write_summary(args, best_acc=best_acc)


if __name__ == "__main__":
    main()
