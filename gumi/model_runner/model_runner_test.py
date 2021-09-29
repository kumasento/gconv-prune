""" Unit tests for model_runner """

import os
import unittest

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from gumi.model_runner.model_runner import ModelRunner
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
        model_runner = ModelRunner(args)

    # NOTE: has been tested
    # def test_ctor_imagenet(self):
    #   """ ImageNet """
    #   parser = create_cli_parser()
    #   args = parser.parse_args(
    #       ['--dataset', 'imagenet', '--dataset-dir', IMAGENET_DIR])
    #   model_runner = ModelRunner(args)

    def test_load_model(self):
        """ Test the load_model function. """
        # CIFAR-10 models
        parser = create_cli_parser()
        args = parser.parse_args([
            '-a', 'densenet40', '--dataset', 'cifar10', '--dataset-dir',
            CIFAR_DIR
        ])
        model_runner = ModelRunner(args)

        model_runner.load_model()


if __name__ == '__main__':
    unittest.main()
