""" Unit tests for model_runner.utils """

import os
import unittest

import gumi.model_runner.utils as utils
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets

os.environ["CUDA_VISABLE_DEVICES"] = "0"

CIFAR_DIR = os.path.expandvars("$NAS_HOME/datasets")
IMAGENET_DIR = os.path.expandvars("$NAS_HOME/datasets/ILSVRC2012")


class TestModelRunnerUtils(unittest.TestCase):
    """ """

    def test_get_dataset(self):
        """ get_dataset """

        with self.assertRaises(AssertionError):
            # should provide dataset dir
            utils.get_dataset("cifar10")
        with self.assertRaises(AssertionError):
            # should provide dataset dir
            utils.get_dataset("imagenet")

        # Skip the rest if CIFAR is not there
        if not os.path.isdir(CIFAR_DIR):
            return

        dataset = utils.get_dataset("cifar10", dataset_dir=CIFAR_DIR)
        self.assertIsInstance(dataset, datasets.CIFAR10)
        self.assertFalse(dataset.train)
        dataset = utils.get_dataset("cifar10", dataset_dir=CIFAR_DIR, is_training=True)
        self.assertTrue(dataset.train)

        dataset = utils.get_dataset("cifar100", dataset_dir=CIFAR_DIR)
        self.assertIsInstance(dataset, datasets.CIFAR100)

        dataset = utils.get_dataset("imagenet", dataset_dir=IMAGENET_DIR)
        self.assertIsInstance(dataset, datasets.ImageFolder)
        self.assertEqual(dataset.root, os.path.join(IMAGENET_DIR, "val"))

    def test_get_data_loader(self):
        """ get_data_loader """
        batch_size = 128
        train_loader = utils.get_data_loader(
            "cifar10", CIFAR_DIR, batch_size, is_training=True
        )
        test_loader = utils.get_data_loader("cifar10", CIFAR_DIR, batch_size)
        self.assertIsInstance(train_loader, data.DataLoader)
        self.assertIsInstance(test_loader, data.DataLoader)

    def test_load_model(self):
        """ load_model """
        # load a CIFAR model
        model = utils.load_model("preresnet164", "cifar10", use_cuda=False)
        self.assertIsInstance(model, nn.Module)

        model = utils.load_model("vgg16", "cifar10", use_cuda=False)
        self.assertIsInstance(model, nn.Module)

        model = utils.load_model("resnet50", "imagenet", use_cuda=False)
        self.assertIsInstance(model, nn.Module)
        model = utils.load_model(
            "resnet50", "imagenet", use_cuda=False, pretrained=True
        )
        self.assertIsInstance(model, nn.Module)


if __name__ == "__main__":
    unittest.main()
