""" Training from scratch with different conditions. """

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

from gumi import model_utils
from gumi.ops import *
from gumi.pruning.export import GroupExporter
from gumi.model_runner import utils
from gumi.model_runner.model_runner import ModelRunner
from gumi.model_runner.parser import create_cli_parser

# CLI parser
parser = create_cli_parser(prog="CLI tool for pruning")
parser.add_argument(
    "--skip-train",
    action="store_true",
    default=False,
    help="Whether to skip the training step.",
)
parser.add_argument(
    "--fine-tune",
    action="store_true",
    default=False,
    help="Whether to fine-tune ONLY the linear classifiers.",
)

args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True


class TransferRunner(ModelRunner):
    """ Runner for transfer learning. """

    def validate_args(self, args):
        pass


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

            if "fc" in key_:
                del state_dict_[key_]

        return state_dict_

    return update_state_dict


def main():
    """ Main """
    # initialise runner
    logging.info("==> Initializing TransferRunner ...")
    runner = TransferRunner(args)

    # load model
    logging.info("==> Loading model ...")
    model = runner.load_model(
        update_state_dict_fn=create_update_state_dict_fn(), fine_tune=args.fine_tune
    )

    # Validate
    logging.info("==> Validating the loaded model ...")
    loss1, acc1 = runner.validate(model)

    # Train
    if args.skip_train:
        logging.info("==> Training has been skipped.")
    else:
        logging.info("==> Run training ...")
        best_acc = runner.train(model)  # parameters are in args

    # Validate again
    logging.info("==> Validating the trained model ...")
    loss2, acc2 = runner.validate(model)


if __name__ == "__main__":
    main()
