""" Training script for CIFAR-10/100.

Reference:
  https://github.com/bearpaw/pytorch-classification/blob/master/cifar.py
  https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from gumi import models  # a module contains all supported models
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and
                     callable(models.__dict__[name]))

# import utilities provided by pytorch-classification
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import cifar_utils

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10/100 Training')

# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument(
    '--dataset-dir', default='data', help='Path to dataset', type=str)
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument(
    '--epochs',
    default=300,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '--train-batch',
    default=128,
    type=int,
    metavar='N',
    help='train batch size')
parser.add_argument(
    '--test-batch', default=100, type=int, metavar='N', help='test batch size')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--drop',
    '--dropout',
    default=0,
    type=float,
    metavar='Dropout',
    help='Dropout ratio')
parser.add_argument(
    '--schedule',
    type=int,
    nargs='+',
    default=[150, 225],
    help='Decrease learning rate at these epochs.')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.1,
    help='LR is multiplied by gamma on schedule.')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=5e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument(
    '-c',
    '--checkpoint',
    default='checkpoint',
    type=str,
    metavar='PATH',
    help='path to save checkpoint (default: checkpoint)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument(
    '--arch',
    '-a',
    metavar='ARCH',
    default='resnet20',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet18)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
# Device options
parser.add_argument(
    '--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument(
    '--num-runs', default=-1, type=int, help='Number of random runs')

# Parse input arguments
args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True


def set_random_seed(args):
  """ Setup random seed. """
  if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
    print('Random seed set as {}'.format(args.manualSeed))

  random.seed(args.manualSeed)
  torch.manual_seed(args.manualSeed)
  if use_cuda:  # global variable
    torch.cuda.manual_seed_all(args.manualSeed)


def save_checkpoint(state,
                    is_best,
                    checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
  filepath = os.path.join(checkpoint, filename)
  torch.save(state, filepath)
  if is_best:
    shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class Runner(object):
  """ CIFAR runner.

  Each runner corresponds to a checkpoint.
  
  Attributes:
    best_acc(Tensor): best accuracy after training
    start_epoch(int)
  """

  def __init__(self, args):
    """ CTOR.
    Args:
      args(object)
    """
    assert args.dataset in ['cifar10', 'cifar100']

    self.state = {k: v for k, v in args._get_kwargs()}
    self.title = 'cifar-10-' + args.arch
    self.best_acc = 0
    self.start_epoch = args.start_epoch
    self.epochs = args.epochs
    self.checkpoint = self.update_checkpoint_path(args)
    os.makedirs(self.checkpoint, exist_ok=True)

    print('==> Preparing dataset %s' % args.dataset)
    self.num_classes = self.get_num_classes(args)
    self.train_loader = self.get_data_loader(args, is_training=True)
    self.test_loader = self.get_data_loader(args, is_training=False)

    print("==> Creating model '{}'".format(args.arch))
    self.model = models.__dict__[args.arch](num_classes=self.num_classes)
    self.model = torch.nn.DataParallel(self.model).cuda()  # parallelise on GPU
    print('    Total params: {:.2f}M'.format(
        sum(p.numel() for p in self.model.parameters()) / 1e6))

    # Create the optimiser
    print("==> Creating model optimizer")
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.SGD(
        self.model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # Load checkpoint if necessary
    if args.resume:
      self.load_checkpoint(args)

    # Logger
    self.logger = self.get_logger(args)

    # Finally store args
    self.args = args

  @staticmethod
  def update_checkpoint_path(args):
    """ Update and retrieve checkpoint path. """
    if args.resume:
      assert os.path.isfile(args.resume)
      args.checkpoint = os.path.dirname(args.resume)

    return args.checkpoint

  @staticmethod
  def get_num_classes(args):
    """ Get number of classes. """
    if args.dataset == 'cifar10':
      return 10
    elif args.dataset == 'cifar100':
      return 100
    raise ValueError('Cannot recognise dataset {}'.format(args.dataset))

  @staticmethod
  def get_data_loader(args, is_training=False):
    """ Create DataLoader from given args.
    
    Args:
      args(object): command-line arguments
      is_training(bool): data loader for training
    Returns:
      A DataLoader object.
    """
    # Get dataset and number of classes
    if args.dataset == 'cifar10':
      Dataset = datasets.CIFAR10
    else:
      Dataset = datasets.CIFAR100

    if is_training:
      transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010)),
      ])
      batch_size = args.train_batch
    else:
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010)),
      ])
      batch_size = args.test_batch

    dataset = Dataset(
        root=args.dataset_dir,
        train=is_training,
        download=is_training,  # only download when creating train set
        transform=transform)
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=args.workers)

    return data_loader

  def load_checkpoint(self, args):
    """ Load checkpoint. """
    assert args.resume  # only call when resume is specified

    print('==> Resuming from checkpoint {}'.format(args.resume))

    # checkpoint content
    checkpoint = torch.load(args.resume)

    # extract checkpoint contents
    self.best_acc = checkpoint['best_acc']
    self.start_epoch = checkpoint['epoch']
    self.model.load_state_dict(checkpoint['state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer'])

  def get_logger(self, args):
    """ Create logger. """
    if args.resume:
      return Logger(
          os.path.join(args.checkpoint, 'log.txt'),
          title=self.title,
          resume=True)
    else:
      logger = Logger(
          os.path.join(args.checkpoint, 'log.txt'), title=self.title)
      logger.set_names([
          'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.',
          'Valid Acc.'
      ])
      return logger

  def adjust_learning_rate(self, epoch):
    """ Adjust learning rate.
    
    Args:
      epoch(int): current epoch
    """
    if epoch in self.args.schedule:  # adjust learning rate at schedule points
      self.state['lr'] *= self.args.gamma
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.state['lr']

  def run(self):
    """ Run train/eval from the start_epoch.

    Returns:
      Final results
    """
    if self.args.evaluate:
      print('\nEvaluation only')
      test_loss, test_acc = cifar_utils.test(self.test_loader, self.model,
                                             self.criterion, self.start_epoch,
                                             use_cuda)
      print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
      return test_acc

    # Main train/eval loop
    for epoch in range(self.start_epoch, self.args.epochs):
      self.adjust_learning_rate(epoch)

      print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, self.epochs,
                                           self.state['lr']))
      # Run train and test for one pass
      train_loss, train_acc = cifar_utils.train(self.train_loader, self.model,
                                                self.criterion, self.optimizer,
                                                epoch, use_cuda)
      test_loss, test_acc = cifar_utils.test(self.test_loader, self.model,
                                             self.criterion, epoch, use_cuda)
      # Append message to Logger
      self.logger.append(
          [self.state['lr'], train_loss, test_loss, train_acc, test_acc])

      # Update best accuracy
      is_best = test_acc > self.best_acc
      self.best_acc = max(test_acc, self.best_acc)

      # Save checkpoint
      checkpoint_state = {
          'epoch': epoch + 1,
          'state_dict': self.model.state_dict(),
          'acc': test_acc,
          'best_acc': self.best_acc,
          'optimizer': self.optimizer.state_dict(),
      }
      save_checkpoint(checkpoint_state, is_best, checkpoint=self.checkpoint)

    # Finalising
    self.logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print("Best acc: {}".format(self.best_acc))
    return self.best_acc


def launch_random_run(args):
  """ Launch a randomized runner. """
  # create a random checkpoint suffix
  new_args = copy.deepcopy(args)
  new_args.checkpoint = args.checkpoint + '.{}'.format(
      time.strftime('%Y%m%d%H%M%S'))

  runner = Runner(new_args)
  return runner.run()


def main():
  """ Main function. """
  set_random_seed(args)

  if args.num_runs == -1:  # run once
    runner = Runner(args)
    runner.run()
  else:
    xs = []

    for i in range(args.num_runs):
      best_acc = launch_random_run(args)
      xs.append(best_acc.item())

    print(xs)
    print('Mean: {:.2f} Median: {:.2f} Var: {:.2f} Stddev: {:.2f}'.format(
        np.mean(xs), np.median(xs), np.var(xs), np.std(xs)))


if __name__ == '__main__':
  main()
