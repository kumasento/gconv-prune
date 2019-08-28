""" Profile the sparse pointwise convolution.
  Compare with normal convolution.
  Can assign different sparse patterns.
"""

import os
import sys
import argparse
import copy
import time
import shutil
import json
import logging
import functools
logging.getLogger().setLevel(logging.DEBUG)

import pandas as pd
import numpy as np
import torch
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from gumi import model_utils
from gumi.ops import *
from gumi.model_runner.parser import create_cli_parser

parser = create_cli_parser(prog='CLI tool for profiling GConv models.')
parser.add_argument(
    '--iters', type=int, default=10000, help='Number of profiling iterations')
parser.add_argument(
    '--use-cuda', action='store_true', default=False, help='Whether to use GPU')
args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cudnn.benchmark = True


class SparseMatrixGenerator(object):
  """ Generate sparse matrices. """

  @staticmethod
  def generate(num_rows, num_cols, groups, block=True):
    """ if block is False, the sparse matrix will be randomly
    generated and the sparsity is (1-groups)/groups. """

    if block:
      logging.info(
          '==> Generating block-sparse matrix {} x {} with G={} ...'.format(
              num_rows, num_cols, groups))
      W = np.zeros((num_rows, num_cols), dtype=np.float32)

      rpg, cpg = num_rows // groups, num_cols // groups
      for g in range(groups):
        W[rpg * g:rpg * (g + 1), cpg * g:cpg * (g + 1)] = np.random.rand(
            rpg, cpg).astype(np.float32)

      perm_rows = np.random.permutation(num_rows)
      perm_cols = np.random.permutation(num_cols)

      return SparseMatrixGenerator.to_sparse_tensor(
          W[perm_rows, :][:, perm_cols])
    else:
      raise ValueError('block=False is not supported yet.')

  @staticmethod
  def to_sparse_tensor(W):
    """ Convert a NumPy matrix to a sparse tensor. """
    row_ind, col_ind = np.nonzero(W)
    i = torch.LongTensor([row_ind.tolist(), col_ind.tolist()])
    v = torch.FloatTensor(W[(row_ind, col_ind)])
    # NOTE: coalesce is critical here
    t = torch.sparse.FloatTensor(i, v, torch.Size(W.shape)).coalesce()
    return t


class Profiler(object):

  def __init__(self, args):
    self.args = args
    self.use_cuda = args.use_cuda
    self.iters = args.iters

  def get_inputs(self, in_channels, out_channels, groups, img_size):
    W = SparseMatrixGenerator.generate(out_channels, in_channels, groups)
    X = torch.rand((in_channels, img_size * img_size))
    return W, X

  def get_fn(self, groups, mode=None):
    if mode is None or mode == 'sparse':
      return torch.sparse.mm

    if mode == 'group':

      def fn(W, X):
        return F.conv2d(X, W, groups=groups)

      return fn

    if mode == 'dense':

      def fn2(W, X):
        return F.conv2d(X, W)

      return fn2

    raise RuntimeError('mode={} cannot be recognised'.format(mode))

  def prepare_inputs(self,
                     W,
                     X,
                     in_channels,
                     out_channels,
                     img_size,
                     groups,
                     mode=None):
    if self.use_cuda:
      W, X = W.cuda(), X.cuda()
    if mode is None or mode == 'sparse':
      return W, X

    X_ = X.view([1, in_channels, img_size, img_size])
    W_ = W.to_dense().view([out_channels, in_channels, 1, 1])

    if mode == 'group':
      W_ = W_[:, :(in_channels // groups), :, :]

    return W_, X_

  def profile_layer(self, W, X, groups, mode=None, iters=None):
    """ Profile a single layer. W is input as a sparse tensor. """
    fn = self.get_fn(groups, mode=mode)

    if iters is None:
      iters = self.iters

    if not self.use_cuda:
      start = time.time()
      for _ in range(iters):
        Y = fn(W, X)
      end = time.time()
      elapsed = end - start
    else:
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)

      start.record()
      for _ in range(iters):
        Y = fn(W, X)
      end.record()

      torch.cuda.synchronize()
      elapsed = start.elapsed_time(end) * 1e-3

    return elapsed * 1e3 / iters

  def profile(self):
    """ Run the profile function """
    scales = [(112, 64), (56, 128), (28, 256), (14, 512), (7, 1024)]
    groups = [1, 2, 4, 8, 16, 32, 64]
    modes = ('sparse', 'group', 'dense')

    data = []
    columns = ['Channels', 'Scale', 'G', 'Mode', 'Time']
    for img_size, channels in scales:
      for g in groups:
        for mode in modes:
          W, X = self.get_inputs(channels, channels, g, img_size)
          W_, X_ = self.prepare_inputs(
              W, X, channels, channels, img_size, g, mode=mode)

          iters = self.iters
          if g >= 8:
            iters *= 10

          logging.info(
              '==> Profiling C={} H=W={} G={} mode={} for {} iters ...'.format(
                  channels, img_size, g, mode, iters))
          elapsed = self.profile_layer(W_, X_, g, mode=mode, iters=iters)
          logging.info('\tTime elapsed: {:.2f} ms '.format(elapsed))

          data.append([channels, img_size, g, mode, elapsed])

    return pd.DataFrame(data, columns=columns)


def main():
  profiler = Profiler(args)
  df = profiler.profile()
  df.to_csv('data/profile/sparse_pointwise_mobilenet_v1_new.csv', index=False)


if __name__ == '__main__':
  main()