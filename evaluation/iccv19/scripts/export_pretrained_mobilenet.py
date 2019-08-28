""" We try to export a pre-trained MobileNet model from:
  https://github.com/marvis/pytorch-mobilenet
  
  to a state_dict that is acceptable by our model. """

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
parser = create_cli_parser(prog='CLI tool for pruning')
args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True


class PretrainedExporter(ModelRunner):
  """ inherited """

  def validate_args(self, args):
    pass


def get_conv_name(part_id):
  if part_id // 2 == 0:
    conv_type = 'd'
  else:
    conv_type = 'p'
  return '{}_conv'.format(conv_type)


def get_layer_index(layer_id):
  bounds = (1, 3, 5, 7, 13)
  for idx, b in enumerate(bounds):
    if layer_id < b:
      if idx == 0:
        return 0, 0
      return idx, layer_id - bounds[idx - 1]

  return len(bounds), layer_id - bounds[-1]


def update_state_dict_fn(state_dict):
  state_dict_ = copy.deepcopy(state_dict)

  for key, val in state_dict.items():
    key_ = key

    if 'module.' in key_:
      del state_dict_[key_]
      key_ = key_.replace('module.', '')
      state_dict_[key_] = val
    if 'model.' in key_:
      del state_dict_[key_]
      key_ = key_.replace('model.', '')
      state_dict_[key_] = val

    # map other keys to different layers
    strs = key_.split('.')
    if len(strs) <= 2:
      continue  # this is the FC case

    # generate new key
    layer_id, part_id = int(strs[0]), int(strs[1])
    index, offset = get_layer_index(layer_id)

    prefix = 'layer{index:}.{offset:}'.format(index=index, offset=offset)
    if layer_id == 0:
      if part_id == 0:
        prefix_ = 'conv1'
      else:
        prefix_ = 'bn'
    elif part_id % 3 == 0:  # conv
      prefix_ = '{prefix:}.{name:}'.format(
          prefix=prefix, name=get_conv_name(part_id))
    else:
      prefix_ = '{prefix:}.bn{id:}'.format(prefix=prefix, id=(part_id // 3 + 1))

    # print('{} -> {}'.format(key_, prefix))

    del state_dict_[key_]
    key_ = key_.replace('{}.{}'.format(layer_id, part_id), prefix_)
    state_dict_[key_] = val

  return state_dict_


def main():
  exporter = PretrainedExporter(args)
  model = exporter.load_model(update_state_dict_fn=update_state_dict_fn)
  # NOTE: validated
  # exporter.validate(model)

  os.makedirs(args.checkpoint, exist_ok=True)
  torch.save({
      'state_dict': model.state_dict()
  }, os.path.join(args.checkpoint, 'checkpoint.pth.tar'))
  shutil.copy(
      os.path.join(args.checkpoint, 'checkpoint.pth.tar'),
      os.path.join(args.checkpoint, 'model_best.pth.tar'))


if __name__ == '__main__':
  main()