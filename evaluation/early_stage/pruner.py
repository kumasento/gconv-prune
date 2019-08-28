""" Defines the Pruner """

import os
import sys
import argparse
import copy
import time
import shutil
import json
import logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
root.addHandler(handler)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from gumi.ops.mask_conv2d import MaskConv2d
from gumi.pruning import prune_utils
from gumi.pruning.reg import *
from gumi import model_utils
from gumi import models  # a module contains all supported models
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and
                     callable(models.__dict__[name]))

import cifar_utils
from utils import *  # import utilities provided by pytorch-classification

__all__ = ['Pruner']


class Pruner:
  """ Entity that executes pruning. """

  def __init__(self, args):
    """ CTOR.

    Args:
      args: command-line arguments.
    """
    assert args.dataset in ['cifar10', 'cifar100']
    assert os.path.isfile(args.resume)

    self.args = args
    self.title = 'cifar-10-' + args.arch
    self.num_classes = self.get_num_classes(args)
    self.criterion = nn.CrossEntropyLoss()
    self.epochs = args.epochs
    self.best_acc = 0
    self.state = {k: v for k, v in args._get_kwargs()}
    self.checkpoint = args.checkpoint
    os.makedirs(self.checkpoint, exist_ok=True)

    # create the original model
    self.model = self.create_model()

    # load checkpoint data from resume
    self.load_checkpoint(args.resume, self.model)
    logging.debug('Loaded checkpoint from {}'.format(args.resume))

    # logger
    self.logger = self.get_logger(args)

  def get_logger(self, args):
    """ Create logger. """
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=self.title)
    logger.set_names([
        'Learning Rate', 'Train Loss', 'Reg Loss', 'Valid Loss', 'Train Acc.',
        'Valid Acc.'
    ])
    return logger

  def load_group_cfg(self, args):
    """ Load group_cfg file. """
    # TODO place it somewhere else
    if not os.path.isfile(args.group_cfg):
      return None
    with open(args.group_cfg, 'r') as f:
      return json.load(f)

  def create_model(self):
    """ Create a new model """
    model = models.__dict__[self.args.arch](num_classes=self.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    logging.debug(
        'Created model by arch={} and num_classes={} on GPU={}'.format(
            self.args.arch, self.num_classes, self.args.gpu_id))
    logging.debug('    Total params: {:.2f}M'.format(
        model_utils.get_model_num_params(model)))

    return model

  @staticmethod
  def get_num_classes(args):
    """ Get number of classes. """
    if args.dataset == 'cifar10':
      return 10
    elif args.dataset == 'cifar100':
      return 100
    raise ValueError('Cannot recognise dataset {}'.format(args.dataset))

  def load_checkpoint(self, checkpoint, model):
    """ Load checkpoint content from the given path. """
    assert os.path.isfile(checkpoint)

    ckpt = torch.load(checkpoint)
    state_dict = self.update_state_dict(ckpt['state_dict'])
    model.load_state_dict(state_dict)

  def update_state_dict(self, state_dict):
    """ Create a copy of the original dict, make it suitable
      for initializing a masked one.

      We need this method because there sometimes will be
      mismatching between keys in the original model and 
      the masked model.
      
      NOTE: don't support loading checkpoint already with mask
    
    Args:
      state_dict(dict)
    Returns:
      A new state_dict
    """
    state_dict_ = copy.deepcopy(state_dict)

    for key, val in state_dict.items():
      if 'conv2d' in key:
        del state_dict_[key]
        state_dict_[key.replace('.conv2d', '')] = val

    return state_dict_

  def regularize(self, *args, **kwargs):
    """ Perform regularization on the loaded model. """
    # figure out the group size
    g_cfg = None
    if os.path.isfile(self.args.group_cfg):
      g_cfg = self.load_group_cfg(self.args)

    # TODO: merge the logic with fine_tune
    self.model.cuda()

    # prepare
    use_cuda = True
    epochs = self.args.reg_epochs
    test_loader = self.get_data_loader()
    train_loader = self.get_data_loader(is_training=True)
    optimizer = optim.SGD(
        self.model.parameters(),
        lr=self.args.reg_lr,
        momentum=self.args.momentum,
        weight_decay=self.args.weight_decay)

    # Create the regularizer
    reg = MaskGroupLASSO(self.model, self.args.reg_scale)

    print('===> Starting regularization ...')
    reg_ckpt = 'reg.pth.tar'

    for epoch in range(epochs):
      print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, epochs,
                                           self.state['reg_lr']))
      # Run train and test for one pass
      train_loss, reg_loss, train_acc = cifar_utils.train(
          train_loader,
          self.model,
          self.criterion,
          optimizer,
          epoch,
          use_cuda,
          reg=reg)
      test_loss, test_acc = cifar_utils.test(
          test_loader,
          self.model,
          self.criterion,
          epoch,
          use_cuda,
          show_bar=not self.args.no_bar)
      # Append message to Logger
      self.logger.append([
          self.state['lr'], train_loss, reg_loss, test_loss, train_acc, test_acc
      ])

      # TODO: save regularizer checkpoint
      checkpoint_state = {
          'epoch': epoch + 1,
          'state_dict': self.model.state_dict(),
          'acc': test_acc,
          'best_acc': self.best_acc,
          'optimizer': optimizer.state_dict(),
      }
      self.save_checkpoint(
          checkpoint_state,
          False,  # no need to check best
          checkpoint=self.checkpoint,
          filename=reg_ckpt)

    print('===> Done regularization')

    return None

  def apply_mask(self):
    """ Set all fake_mask to False """
    for name, m in self.model.named_modules():
      if isinstance(m, MaskConv2d):
        m.apply_mask()

  def prune(self, *args, **kwargs):
    """ Prunes the model stored at checkpoint.
    
    Args:
    Returns:
      None
    """
    g_cfg = None
    if os.path.isfile(self.args.group_cfg):
      g_cfg = self.load_group_cfg(self.args)

    # currently we prune all modules at once
    for name, m in self.model.named_modules():
      if isinstance(m, MaskConv2d):
        self.prune_module(name, m, *args, g_cfg=g_cfg, **kwargs)

    if self.args.scratch:  # requires to train from scratch
      # create a new model, assign mask into it
      self.purge_model()

  def prune_module(self,
                   name,
                   mod,
                   *args,
                   g_cfg=None,
                   fake_mask=False,
                   **kwargs):
    """ Prune a single module.

      We expect that after pruning, the mask in mod can be
      updated to a pruned result.
    
    Args:
      name(str): name of the module
      mod(MaskConv2d): the module to be pruned.
    """
    # TODO maybe not pass by kwargs
    G = self.args.num_groups
    if g_cfg is not None and name in g_cfg:
      G = g_cfg[name]['G']
      # do some verification
      W = model_utils.get_weight_parameter(mod)
      F, C = W.shape[:2]
      assert F == g_cfg[name]['F'] and C == g_cfg[name]['C']

    if fake_mask and isinstance(mod, MaskConv2d):
      mod.fake_mask = True

    prune_utils.prune_module(mod, G=G, MCPG=self.args.mcpg, **kwargs)

  def purge_model(self):
    """ Purge weight in model to the initialized values.
    
    Only masks are remained.
    """
    model = self.create_model()

    mask_dict = {}  # build a dict of masks from ORIGIN
    for name, param in self.model.named_parameters():
      if 'mask' in name:
        mask_dict[name] = param
    # update the new model
    for name, param in model.named_parameters():
      if 'mask' in name:
        param.data = mask_dict[name].data  # update

    self.model = model

  def get_data_loader(self, is_training=False):
    """ Create DataLoader from given args.
    
    Args:
      is_training(bool): data loader for training
    Returns:
      A DataLoader object.
    """
    args = self.args

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

  def adjust_learning_rate(self, epoch, optimizer):
    """ Adjust learning rate.
    
    Args:
      epoch(int): current epoch
    """
    if epoch in self.args.schedule:  # adjust learning rate at schedule points
      self.state['lr'] *= self.args.gamma
      for param_group in optimizer.param_groups:
        param_group['lr'] = self.state['lr']

  def save_checkpoint(self,
                      state,
                      is_best,
                      checkpoint='checkpoint',
                      filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
      shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

  def evaluate(self):
    """ Evaluate accuracy of the current model. """
    self.model.cuda()  # NOTE: should assign to CUDA again

    test_loader = self.get_data_loader()
    test_loss, test_acc = cifar_utils.test(
        test_loader,
        self.model,
        self.criterion,
        0,
        True,
        show_bar=not self.args.no_bar)

    logging.debug('Loss={:.2f} Accuracy={:.2f}%'.format(test_loss, test_acc))

    return test_loss, test_acc

  def fine_tune(self, num_epochs):
    """ Further tuning the pruned model. """
    self.model.cuda()

    # prepare
    use_cuda = True
    test_loader = self.get_data_loader()
    train_loader = self.get_data_loader(is_training=True)
    optimizer = optim.SGD(
        self.model.parameters(),
        lr=self.args.lr,
        momentum=self.args.momentum,
        weight_decay=self.args.weight_decay)

    for epoch in range(self.args.epochs):
      # TODO(13/02/2019): learning rate adjustment
      self.adjust_learning_rate(epoch, optimizer)

      print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, self.epochs,
                                           self.state['lr']))
      # Run train and test for one pass
      train_loss, train_acc = cifar_utils.train(
          train_loader, self.model, self.criterion, optimizer, epoch, use_cuda)
      test_loss, test_acc = cifar_utils.test(
          test_loader,
          self.model,
          self.criterion,
          epoch,
          use_cuda,
          show_bar=not self.args.no_bar)
      # Append message to Logger
      self.logger.append(
          [self.state['lr'], train_loss, 0.0, test_loss, train_acc, test_acc])

      # Update best accuracy
      is_best = test_acc > self.best_acc
      self.best_acc = max(test_acc, self.best_acc)

      checkpoint_state = {
          'epoch': epoch + 1,
          'state_dict': self.model.state_dict(),
          'acc': test_acc,
          'best_acc': self.best_acc,
          'optimizer': optimizer.state_dict(),
      }
      self.save_checkpoint(
          checkpoint_state, is_best, checkpoint=self.checkpoint)

    # Finalising
    self.logger.close()
    logging.info('Best accuracy while fine-tuning: {:.2f}%'.format(
        self.best_acc))

    return self.best_acc