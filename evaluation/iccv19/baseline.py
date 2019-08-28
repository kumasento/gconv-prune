""" Run collect baseline models.

  We load model from args.resume while dumping trained models to args.checkpoint.
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

from gumi import model_utils
from gumi.ops import *
from gumi.model_runner import utils
from gumi.model_runner.model_runner import ModelRunner
from gumi.model_runner.parser import create_cli_parser

# CLI parser
parser = create_cli_parser(prog='CLI tool for creating baseline models.')
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


class BaselineRunner(ModelRunner):
  """ Specialised for training and evaluating the baseline. """

  def validate_args(self, args):
    pass


def main():
  """ MAIN """
  runner = BaselineRunner(args)

  logging.info('==> Loading model {} for {} ...'.format(args.arch,
                                                        args.dataset))
  model = runner.load_model()

  logging.info('==> Validating the loaded model ...')
  loss1, acc1 = runner.validate(model)

  if args.skip_train:
    logging.info('==> Training has been skipped.')
  else:
    logging.info('==> Run training ...')
    best_acc = runner.train(model)  # parameters are in args

  logging.info('==> Validating the trained model ...')
  loss2, acc2 = runner.validate(model)


if __name__ == '__main__':
  main()