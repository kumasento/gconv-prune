""" Unit tests for model_runner """

import os
import unittest

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from gumi.model_runner.model_pruner import ModelPruner
from gumi.model_runner.parser import create_cli_parser

CIFAR_DIR = os.path.expandvars('$NAS_HOME/datasets')
IMAGENET_DIR = os.path.expandvars('$NAS_HOME/datasets/ILSVRC2012')


class TestModelRunner(unittest.TestCase):
  """ """

  def test_ctor(self):
    """ CIFAR """
    parser = create_cli_parser()
    args = parser.parse_args(
        ['--dataset', 'cifar10', '--dataset-dir', CIFAR_DIR])
    pruner = ModelPruner(args)


if __name__ == '__main__':
  unittest.main()