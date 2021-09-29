""" Optimisation module for group size. """

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
from collections import OrderedDict
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

from gumi.ops import *
from gumi import models
from gumi import model_utils
from gumi.pruning import mask_utils

import cifar_utils
from parser import create_parser

parser = create_parser(prog="Generate an optimised group configuration.")
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True


def factors(n):
    """
  Copied from - https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
  """
    return set(
        functools.reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0),
        )
    )


class GOpt(object):
    """ Group size optimizer. """

    def __init__(self, args):
        """ CTOR. """
        self.args = args

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

        return model

    def find_groupable_modules(self, model):
        """ Find modules that can be grouped. """
        mods = []  # return list
        for name, mod in model.named_modules():
            if isinstance(mod, MaskConv2d):
                W = model_utils.get_weight_parameter(mod)
                F, C = W.shape[:2]
                ff, fc = factors(F), factors(C)

                if len(ff.intersection(fc)) <= 1:
                    continue

                mods.append((name, mod))
        return mods

    def find_group_candidates(self, mod, **kwargs):
        """ Find group number candidates in module.
    
    Note: use kwargs to pass additional requirements.
    """
        W = model_utils.get_weight_parameter(mod)
        F, C = W.shape[:2]

        # common divisors
        Gs = list(sorted(factors(F).intersection(factors(C))))
        del Gs[0]  # should be 1

        costs = []
        for G in Gs:
            _, _, cost = mask_utils.run_mbm(W, G)
            costs.append(cost)

        return Gs, costs

    def run_opt(self):
        """ Run the actual optimization """
        logging.info("Finding the optimal group configuration ...")

        model = self.load_model(self.args)

        g_conf = OrderedDict()
        for idx, (name, mod) in enumerate(self.find_groupable_modules(model)):
            # the sequence here is important
            Gs, costs = list(self.find_group_candidates(mod))

            W = model_utils.get_weight_parameter(mod)
            F, C = W.shape[:2]
            # TODO: simply select the one with the most cost
            G = Gs[np.argmax(costs)]
            # print(np.max(costs) / (F * C))

            # update the dictionary
            g_conf[name] = {"id": idx, "F": F, "C": C, "G": G}

        return g_conf

    def dump_group_conf(self, g_conf, file_name):
        """ Dump a group configuration file. """
        logging.info("Dumping group config to {} ...".format(file_name))

        with open(file_name, "w") as f:
            json.dump(g_conf, f, indent=2)


def main():
    """ Main runner. """
    gopt = GOpt(args)
    g_conf = gopt.run_opt()
    gopt.dump_group_conf(g_conf, args.group_cfg)


if __name__ == "__main__":
    main()
