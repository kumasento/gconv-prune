""" An inference demo. """

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
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# image
from PIL import Image
from skimage import io, transform

from gumi import model_utils
from gumi.ops import *
from gumi.pruning.export import GroupExporter
from gumi.model_runner import utils
from gumi.model_runner.model_runner import ModelRunner
from gumi.model_runner.parser import create_cli_parser

parser = create_cli_parser(prog='CLI tool for pruning')
parser.add_argument(
    '--image', type=str, metavar='PATH', help='Path to an image file.')
args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True


class ModelInferRunner(ModelRunner):

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

      if 'module' in key_:
        del state_dict_[key_]
        key_ = key_.replace('module.', '')
        state_dict_[key_] = val

    return state_dict_

  return update_state_dict


def create_update_model_fn():
  """ """

  def update_model(model):
    """ """
    utils.apply_mask(model, use_cuda=False)
    return model

  return update_model


def main():
  # initialise runner
  logging.info('==> Initializing ModelInferRunner ...')
  runner = ModelInferRunner(args)

  # load model
  logging.info('==> Loading model ...')
  model = runner.load_model(
      update_model_fn=create_update_model_fn(),
      update_state_dict_fn=create_update_state_dict_fn())
  model.eval()

  logging.info('==> Loading an image from {} ...'.format(args.image))
  image = Image.open(args.image)

  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
  ])

  x = transform(image)
  y = model.forward(x.view([1, *x.shape]))
  label = torch.argmax(F.softmax(y, dim=1), dim=1)[0]

  print('Predicted label index: {}'.format(label))


if __name__ == '__main__':
  main()
