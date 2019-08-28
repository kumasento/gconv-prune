""" Export a fine-tuned mask based model to a grouped one. """

import os
import sys
import argparse
import copy
import time
import shutil
import json
import itertools
import functools
from subprocess import Popen, PIPE  # launching pruning processes
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

from parser import create_parser
from group_exporter import GroupExporter

parser = create_parser(prog='Export from a mask based model to a grouped one.')
args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True


def main():
  exporter = GroupExporter(args)
  exporter.export()


if __name__ == '__main__':
  main()