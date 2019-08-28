""" Export group convolution based models from a mask based one. 

Usage: 
  python export.py --no-data-loader -a resnet34 -d imagenet --group-cfg config/resnet34_A.json --resume $ICCV19/imagenet/resnet34_A_GRPS_LR_1e-3_30-10-20_NEW/model_best.pth.tar

  There will be a new model gconv.pth.tar written over in the resume directory.
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
import torch.onnx
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from gumi import model_utils
from gumi.ops import *
from gumi.pruning.export import GroupExporter
from gumi.model_runner import utils
from gumi.model_runner.model_pruner import ModelPruner
from gumi.model_runner.parser import create_cli_parser

parser = create_cli_parser(prog='CLI tool for exporting GConv models.')
parser.add_argument(
    '--onnx',
    action='store_true',
    default=False,
    help='Whether to export to ONNX')
parser.add_argument(
    '--mm', action='store_true', default=False, help='MM pointwise')
parser.add_argument(
    '--sparse',
    action='store_true',
    default=False,
    help='Export model with sparse pointwise')
parser.add_argument(
    '--std',
    action='store_true',
    default=False,
    help='Export model with standard convolution')
parser.add_argument(
    '--min-sparse-channels',
    type=int,
    default=0,
    help='Min channels to sparsify.')
parser.add_argument(
    '--val', action='store_true', default=False, help='Validate')
args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True


class GroupModelExporter(ModelPruner):
  """ inherited """

  def validate_args(self, args):
    pass


def create_update_model_fn():
  """ """

  def update_model(model):
    """ """
    utils.apply_mask(model, use_cuda=False)
    return model

  return update_model


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


def main():
  """ Load the model and prune it """
  logging.info('==> Initializing GroupModelExporter ...')
  exporter = GroupModelExporter(args)

  # load model
  logging.info('==> Loading model ...')
  model = exporter.load_model(
      use_cuda=True,
      data_parallel=False,
      update_model_fn=create_update_model_fn(),
      update_state_dict_fn=create_update_state_dict_fn())

  # prune it to setup each MaskConv2d
  get_num_groups_fn = utils.create_get_num_groups_fn(
      G=args.num_groups,
      MCPG=args.mcpg,
      group_cfg=args.group_cfg,
      use_cuda=True,
      data_parallel=False)

  exporter.prune(
      model, get_num_groups_fn=get_num_groups_fn, keep_mask=True, use_cuda=True)
  for name, mod in model.named_modules():
    if isinstance(mod, MaskConv2d):
      print(name, mod.G)

  # export
  logging.info('==> Exporting the model ...')
  model = GroupExporter.export(
      model,
      use_cuda=True,
      mm=args.mm,
      sparse=args.sparse,
      std=args.std,
      min_sparse_channels=args.min_sparse_channels)

  # evaluate
  if args.val:
    exporter.validate(model)
  # move back to CPU
  model.cpu()
  # print(model)
  logging.debug('Total params: {:.2f}M FLOPS: {:.2f}M'.format(
      model_utils.get_model_num_params(model),
      utils.get_model_num_ops(model, args.dataset)))

  logging.debug('==> Densify model ...')
  utils.apply_dense(model)

  # save model
  if not args.onnx:
    suffix = ''
    if args.mm:
      suffix += '_mm'
    if args.sparse:
      suffix += '_sparse'
    if args.std:
      suffix += '_std'
    if args.min_sparse_channels > 0:
      suffix += '_min{}'.format(args.min_sparse_channels)

    fp = os.path.join(
        os.path.dirname(args.resume), 'gconv{}.pth.tar'.format(suffix))
    logging.info('==> Saving the model to {} ...'.format(fp))
    torch.save(model, fp)
  else:
    fp = os.path.join(os.path.dirname(args.resume), 'gconv.onnx')
    logging.info('==> Exporting model to ONNX {} ...'.format(fp))

    if args.dataset in utils.IMAGENET_DATASETS:
      x = torch.randn(1, 3, 224, 224, requires_grad=True)
    else:
      raise RuntimeError('Do not support {}'.format(args.dataset))

    onnx = torch.onnx._export(model, x, fp, export_params=True)


if __name__ == '__main__':
  main()