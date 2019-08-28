""" Evaluate the performance of GroupConv2d on CPU and GPU.

Reference:
- https://github.com/pytorch/pytorch/issues/10229
"""

import time
import math
import argparse

import torch
from gumi.ops.group_conv2d import GroupConv2d

parser = argparse.ArgumentParser('PyTorch group convolution profiling tool.')
parser.add_argument(
    '-i', '--in-channel', default=1024, type=int, help='Input channels')
parser.add_argument(
    '-o', '--out-channel', default=1024, type=int, help='Output channels')
parser.add_argument('-s', '--size', default=32, type=int, help='Image size')
parser.add_argument('-b', '--batch', default=1, type=int, help='Batch size')

torch.backends.cudnn.benchmark = True


def count(a, m):
  torch.cuda.synchronize()
  t0 = time.time()
  for i in range(1000):
    b = m(a)
  torch.cuda.synchronize()
  return time.time() - t0


def test(in_channel=1024, out_channel=1024, size=32, batch=1):
  x = torch.randn((batch, in_channel, size, size))
  x = x.cuda()

  n = int(math.log(min(in_channel, out_channel), 2)) + 1
  for i in range(n):
    g = int(math.pow(2, i))
    m = GroupConv2d(in_channel, out_channel, 3, num_groups=g).cuda()
    b = m(x)  #warm up
    t = count(x, m)

    print('OutChannel:{}, Group:{}, Time:{}'.format(out_channel, g, t))


if __name__ == '__main__':
  args = parser.parse_args()
  test(
      in_channel=args.in_channel,
      out_channel=args.out_channel,
      size=args.size,
      batch=args.batch)
