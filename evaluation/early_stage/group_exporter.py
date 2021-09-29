""" A utility you can use to export a model from mask to group conv. """

import os
import sys
import argparse
import copy
import time
import shutil
import json
import itertools
import functools
import logging
from subprocess import Popen, PIPE  # launching pruning processes

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

import cifar_utils
from gumi.ops import *
from gumi import models
from gumi import model_utils


class GroupExporter(object):
    """ Exporter from mask model to group convolution. """

    def __init__(self, args):
        """ CTOR. """
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

    def create_model(self, args, **kwargs):
        """ Create model only. """
        num_classes = model_utils.get_num_classes(args)
        model = models.__dict__[args.arch](num_classes=num_classes, **kwargs)
        model = torch.nn.DataParallel(model).cuda()

        logging.info(
            "Created model by arch={} and num_classes={} on GPU={}".format(
                args.arch, num_classes, args.gpu_id
            )
        )
        logging.info(
            "    Total params: {:.2f}M".format(model_utils.get_model_num_params(model))
        )

        return model

    def load_model(self, args, **kwargs):
        """ Load the original Mask based model. """
        model = self.create_model(args, **kwargs)

        model_utils.load_checkpoint(args.resume, model)
        logging.debug("Checkpoint loaded from {}".format(args.resume))

        for name, mod in model.named_modules():
            if isinstance(mod, MaskConv2d):
                mod.apply_mask()

        return model

    def find_groupable_modules(self, model, G=1, MCPG=None, g_cfg=None):
        """ Find modules that can be turned into GroupConv2d. """
        mods = []

        for name, mod in model.named_modules():
            if isinstance(mod, MaskConv2d):
                # we assume that if the number of groups is not divisible
                # it is not grouped
                # TODO: support hybrid group size
                weight = model_utils.get_weight_parameter(mod)
                F, C = weight.shape[:2]

                if not GroupConv2d.groupable(
                    C, F, groups=G, max_channels_per_group=MCPG
                ):
                    continue

                # get the desired group number
                if g_cfg and name in g_cfg:
                    G_ = g_cfg[name]["G"]
                else:
                    G_ = GroupConv2d.get_num_groups(
                        C, F, groups=G, max_channels_per_group=MCPG
                    )

                # we also need to check whether GSP conditions are met.
                if not model_utils.is_gsp_satisfied(mod, G_):
                    continue

                mods.append((name, mod))

        return mods

    def load_group_cfg(self, args):
        """ Load group_cfg file. """
        # TODO place it somewhere else
        if not os.path.isfile(args.group_cfg):
            return None

        with open(args.group_cfg, "r") as f:
            return json.load(f)

    def update_model_by_group_cfg(self, model, g_cfg):
        """ Post update the G value of each GroupConv2d. """
        for name, mod in model.named_modules():
            if name in g_cfg:
                assert isinstance(mod, GroupConv2d)
                G = g_cfg[name]["G"]
                mod.setup_conv2d(G)

        logging.info("Updated G of each model by group_cfg.")
        logging.info(
            "    Total params: {:.2f}M".format(model_utils.get_model_num_params(model))
        )

    def get_data_loader(self, is_training=False):
        """ Create DataLoader from given args.
    
    Args:
      is_training(bool): data loader for training
    Returns:
      A DataLoader object.
    """
        args = self.args

        # Get dataset and number of classes
        if args.dataset == "cifar10":
            Dataset = datasets.CIFAR10
        else:
            Dataset = datasets.CIFAR100

        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            batch_size = args.train_batch
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            batch_size = args.test_batch

        dataset = Dataset(
            root=args.dataset_dir,
            train=is_training,
            download=is_training,  # only download when creating train set
            transform=transform,
        )
        data_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=args.workers,
        )

        return data_loader

    def evaluate(self, model):
        """ Evaluate accuracy of the current model. """
        model.cuda()  # NOTE: should assign to CUDA again

        test_loader = self.get_data_loader()
        test_loss, test_acc = cifar_utils.test(
            test_loader, model, self.criterion, 0, True, show_bar=True
        )

        logging.debug("Loss={:.2f} Accuracy={:.2f}%".format(test_loss, test_acc))

        return test_loss, test_acc

    def export(self):
        """ Export function """
        # load the original model
        model = self.load_model(self.args)
        self.evaluate(model)

        # all these modules can be further converted to GroupConv2d
        G = self.args.num_groups
        MCPG = self.args.mcpg
        g_cfg = self.load_group_cfg(self.args)

        logging.info("Finding all groupable modules ...")
        mods = self.find_groupable_modules(model, G=G, MCPG=MCPG, g_cfg=g_cfg)
        logging.info("Found {} modules".format(len(mods)))

        # for each module, create corresponding group convolution
        # parameters, and use them to update the state_dict
        logging.info("Generating GroupConv2d parameters ...")
        state_dict = torch.load(self.args.resume)["state_dict"]
        for name, mod in mods:
            weight = model_utils.get_weight_parameter(mod)
            F, C = weight.shape[:2]

            # will update G correspondingly
            # TODO put this logic somewhere else, frequently reused
            if g_cfg and name in g_cfg:
                G_ = g_cfg[name]["G"]
            else:
                G_ = GroupConv2d.get_num_groups(C, F, MCPG, groups=G)

            wg, ind_in, ind_out = model_utils.get_group_parameters(mod, G_)

            # update the state_dict
            del state_dict[name + ".mask"]  # delete mask key

            # purge weight in MaskConv2d
            if (name + ".weight") in state_dict:
                del state_dict[name + ".weight"]

            state_dict[name + ".conv2d.weight"] = torch.from_numpy(wg)
            state_dict[name + ".ind_in"] = torch.from_numpy(ind_in).long()
            state_dict[name + ".ind_out"] = torch.from_numpy(ind_out).long()

        # create model
        # leave indices to None, will set up when loading state_dict
        model = self.create_model(
            self.args, groups=G, max_channels_per_group=MCPG, mask=False
        )
        if g_cfg:  # should post update
            self.update_model_by_group_cfg(model, g_cfg)
        # insert weights
        model.load_state_dict(state_dict)

        self.evaluate(model)
