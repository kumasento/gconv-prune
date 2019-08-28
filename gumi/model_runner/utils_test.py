""" Unit tests for model_runner.utils """

import os
os.environ['CUDA_VISABLE_DEVICES'] = '0'
import unittest

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import gumi.model_runner.utils as utils

CIFAR_DIR = os.path.expandvars('$NAS_HOME/datasets')
IMAGENET_DIR = os.path.expandvars('$NAS_HOME/datasets/ILSVRC2012')


class TestModelRunnerUtils(unittest.TestCase):
  """ """

  def test_get_dataset(self):
    """ get_dataset """

    with self.assertRaises(AssertionError):
      # should provide dataset dir
      utils.get_dataset('cifar10')
    with self.assertRaises(AssertionError):
      # should provide dataset dir
      utils.get_dataset('imagenet')

    dataset = utils.get_dataset('cifar10', dataset_dir=CIFAR_DIR)
    self.assertIsInstance(dataset, datasets.CIFAR10)
    self.assertFalse(dataset.train)
    dataset = utils.get_dataset(
        'cifar10', dataset_dir=CIFAR_DIR, is_training=True)
    self.assertTrue(dataset.train)

    dataset = utils.get_dataset('cifar100', dataset_dir=CIFAR_DIR)
    self.assertIsInstance(dataset, datasets.CIFAR100)

    dataset = utils.get_dataset('imagenet', dataset_dir=IMAGENET_DIR)
    self.assertIsInstance(dataset, datasets.ImageFolder)
    self.assertEqual(dataset.root, os.path.join(IMAGENET_DIR, 'val'))

    # NOTE: skipped
    # dataset = utils.get_dataset(
    #     'imagenet', dataset_dir=IMAGENET_DIR, is_training=True)
    # self.assertIsInstance(dataset, datasets.ImageFolder)
    # self.assertEqual(dataset.root, os.path.join(IMAGENET_DIR, 'train'))

  def test_get_data_loader(self):
    """ get_data_loader """
    batch_size = 128

    train_loader = utils.get_data_loader(
        'cifar10', CIFAR_DIR, batch_size, is_training=True)
    test_loader = utils.get_data_loader('cifar10', CIFAR_DIR, batch_size)
    self.assertIsInstance(train_loader, data.DataLoader)
    self.assertIsInstance(test_loader, data.DataLoader)

    # NOTE: skipped
    # train_loader = utils.get_data_loader(
    #     'imagenet', IMAGENET_DIR, batch_size, is_training=True)
    # test_loader = utils.get_data_loader('imagenet', IMAGENET_DIR, batch_size)
    # self.assertIsInstance(train_loader, data.DataLoader)
    # self.assertIsInstance(test_loader, data.DataLoader)

  def test_load_model(self):
    """ load_model """
    # load a CIFAR model
    model = utils.load_model('preresnet164', 'cifar10', use_cuda=False)
    self.assertIsInstance(model, nn.Module)

    model = utils.load_model('vgg16', 'cifar10', use_cuda=False)
    self.assertIsInstance(model, nn.Module)

    model = utils.load_model('resnet50', 'imagenet', use_cuda=False)
    self.assertIsInstance(model, nn.Module)
    model = utils.load_model(
        'resnet50', 'imagenet', use_cuda=False, pretrained=True)
    self.assertIsInstance(model, nn.Module)

    # TODO: test resume

  def test_validate(self):
    """ Test validation """
    # NOTE: skipped
    # batch_size = 128
    # val_loader = utils.get_data_loader('imagenet', IMAGENET_DIR, batch_size)
    # model = utils.load_model(
    #     'resnet50', 'imagenet', use_cuda=True, pretrained=True)
    # top1 = utils.validate(val_loader, model, nn.CrossEntropyLoss(), gpu=0)
    # print(top1)


if __name__ == '__main__':
  unittest.main()