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
parser = create_cli_parser(prog='CLI tool for pruning')
parser.add_argument(
    '--skip-train',
    action='store_true',
    default=False,
    help='Whether to skip the training step.')

args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True


class ScratchRunner(ModelRunner):
  """ Runner for training from scratch. """

  def load_model(self):
    """ Prepare some parameters to configure group convolution. """
    args = self.args
    return super().load_model(
        mask=False,  # NOTE: explicit
        groups=args.num_groups,
        max_channels_per_group=args.mcpg,
        ind_type=args.ind)


def main():
  """ Main """
  # initialise runner
  logging.info('==> Initializing ScratchRunner ...')
  runner = ScratchRunner(args)

  # load model
  logging.info('==> Loading model ...')
  model = runner.load_model()

  # Validate
  logging.info('==> Validating the loaded model ...')
  loss1, acc1 = runner.validate(model)

  # Train
  if args.skip_train:
    logging.info('==> Training has been skipped.')
  else:
    logging.info('==> Run training ...')
    best_acc = runner.train(model)  # parameters are in args

  # Validate again
  logging.info('==> Validating the trained model ...')
  loss2, acc2 = runner.validate(model)


if __name__ == '__main__':
  main()
