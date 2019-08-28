""" Pruning CLI tool. """

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
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from gumi import model_utils
from gumi.ops import *
from gumi.pruning.export import GroupExporter
from gumi.model_runner import utils
from gumi.model_runner.model_pruner import ModelPruner
from gumi.model_runner.parser import create_cli_parser

# CLI parser
parser = create_cli_parser(prog='CLI tool for pruning')
parser.add_argument('--apply-mask',
                    action='store_true',
                    default=False,
                    help='Whether to apply mask when loading model')
parser.add_argument('--skip-prune',
                    action='store_true',
                    default=False,
                    help='Whether to perform the fine-tuning step only.')
parser.add_argument('--skip-fine-tune',
                    action='store_true',
                    default=False,
                    help='Whether to skip file tune.')
parser.add_argument('--skip-validation',
                    action='store_true',
                    default=False,
                    help='Skip all validation.')
parser.add_argument('--condensenet',
                    action='store_true',
                    default=False,
                    help='Custom rules for updating condensenet state dict')
parser.add_argument('--keep-mask',
                    action='store_true',
                    default=False,
                    help='Keep the mask loaded from pre-trained models')
parser.add_argument('--fine-tune',
                    action='store_true',
                    default=False,
                    help='DEPRECATED Only fine-tunes the classifier.')
parser.add_argument('--train-from-scratch',
                    action='store_true',
                    default=False,
                    help='Train from scratch in the post-pruning phase')
parser.add_argument('--manual-seed',
                    default=None,
                    type=int,
                    help='Manual seed for reproducibility.')
args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True

# TODO:  Move it somewhere else
# Follows: https://pytorch.org/docs/stable/notes/randomness.html
if args.manual_seed is not None:
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_update_model_fn(arch,
                           dataset,
                           pretrained,
                           resume,
                           condensenet=False,
                           apply_mask=False):
    """ """

    def update_model(model):
        """ """
        if not resume:
            return model

        if apply_mask:
            # apply mask right now
            utils.apply_mask(model)

        return model

    return update_model


def create_update_state_dict_fn(no_mask=False, condensenet=False):
    # TODO: remove these condensenet hacks

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

            if 'conv2d' in key_:
                del state_dict_[key_]
                key_ = key_.replace('.conv2d', '')
                state_dict_[key_] = val

                if condensenet:
                    # append a mask
                    mask = torch.ones(val.shape[:2])
                    state_dict_[key_.replace('.weight', '.mask')] = mask

            if no_mask and 'mask' in key_:
                del state_dict_[key_]

            if condensenet and 'ind' in key_:
                del state_dict_[key_]

        return state_dict_

    return update_state_dict


def create_get_num_groups_fn(G=0, MCPG=0, group_cfg=None):
    """ Create the hook function for getting 
    the number of groups for a given module. """

    g_cfg = None
    if isinstance(group_cfg, str) and os.path.isfile(group_cfg):
        with open(group_cfg, 'r') as f:
            g_cfg = json.load(f)

    def get_num_groups(name, mod):
        G_ = G  # choose G in the beginning

        W = model_utils.get_weight_parameter(mod)
        F, C = W.shape[:2]

        # how to override G_
        if g_cfg is not None:
            if name in g_cfg:
                G_ = g_cfg[name]['G']
                # do some verification
                assert F == g_cfg[name]['F'] and C == g_cfg[name]['C']
            else:
                G_ = 1  # HACK - we don't want to have G=0 in further processing

        elif MCPG > 0:
            if GroupConv2d.groupable(C, F, max_channels_per_group=MCPG):
                G_ = GroupConv2d.get_num_groups(C,
                                                F,
                                                max_channels_per_group=MCPG)
            else:
                logging.warn(
                    'Module {} is not groupable under MCPG={}, set its G to 1'.
                    format(name, MCPG))
                G_ = 1

        return G_

    return get_num_groups


def main():
    """ Main """
    logging.info('==> Initializing ModelPruner ...')
    model_pruner = ModelPruner(args)

    # load model
    logging.info('==> Loading model ...')

    update_model_fn = create_update_model_fn(args.arch,
                                             args.dataset,
                                             args.pretrained,
                                             args.resume,
                                             apply_mask=args.apply_mask,
                                             condensenet=args.condensenet)
    model = model_pruner.load_model(
        update_model_fn=update_model_fn,
        update_state_dict_fn=create_update_state_dict_fn(
            no_mask=not args.resume, condensenet=args.condensenet),
        fine_tune=args.fine_tune)
    # evaluate the performance of the model in the beginning
    if not args.skip_validation:
        logging.info('==> Validating the loaded model ...')
        loss1, acc1 = model_pruner.validate(model)

    #################################################
    # Pruning                                       #
    #                                               #
    #################################################
    if not args.apply_mask:
        # NOTE: we have not applied mask yet
        # # major pruning function
        logging.info('==> Replacing Conv2d in model by MaskConv2d ...')
        # TODO - duplicated with update_model_fn?
        # not quite, if not resume the model won't be updated
        utils.apply_mask(model)

        if not args.skip_validation:
            logging.info('==> Validating the masked model ...')
            loss2, acc2 = model_pruner.validate(model)
            assert torch.allclose(acc1, acc2)

    # run pruning (update the content of mask)
    logging.info('==> Pruning model ...')
    if not args.skip_prune:
        get_num_groups = create_get_num_groups_fn(G=args.num_groups,
                                                  MCPG=args.mcpg,
                                                  group_cfg=args.group_cfg)

        logging.debug('Pruning configuration:')
        logging.debug('PERM:        {}'.format(args.perm))
        logging.debug('NS:          {}'.format(args.num_sort_iters))
        logging.debug('No weight:   {}'.format(args.no_weight))
        logging.debug('Keep mask:   {}'.format(args.keep_mask))
        logging.debug('')

        model_pruner.prune(model,
                           get_num_groups_fn=get_num_groups,
                           perm=args.perm,
                           no_weight=args.no_weight,
                           num_iters=args.num_sort_iters,
                           keep_mask=args.keep_mask)

        if not args.skip_validation:
            logging.info('==> Validating the pruned model ...')
            loss3, acc3 = model_pruner.validate(model)

    else:
        logging.info('Pruning has been skipped, you have the original model.')

    #################################################
    # Fine-tuning                                   #
    #                                               #
    #################################################
    logging.info('==> Fine-tuning the pruned model ...')
    if args.train_from_scratch:
        logging.info('==> Training the pruned topology from scratch ...')

        # reset weight parameters
        # TODO: refactorize
        for name, mod in model.named_modules():
            if hasattr(
                    mod,
                    'weight') and len(mod.weight.shape) >= 2:  # re-initialize
                torch.nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')
                # if hasattr(mod, 'G'):
                #   mod.weight.data.mul_(mod.G)
            if hasattr(mod, 'bias') and mod.bias is not None:
                mod.bias.data.fill_(0.0)

    if not args.skip_fine_tune:
        model_pruner.fine_tune(model)

        if not args.skip_validation:
            logging.info('==> Validating the fine-tuned pruned model ...')
            loss4, acc4 = model_pruner.validate(model)
            logging.info(
                '==> Final validation accuracy of the pruned model: {:.2f}%'.
                format(acc4))
    else:
        logging.info('Fine-tuning has been skipped.')

    #################################################
    # Export                                        #
    #                                               #
    #################################################
    logging.info('==> Exporting the model ...')
    model = GroupExporter.export(model)
    if use_cuda:
        model.cuda()
    logging.debug('Total params: {:.2f}M FLOPS: {:.2f}M'.format(
        model_utils.get_model_num_params(model),
        utils.get_model_num_ops(model, args.dataset)))

    if not args.skip_validation:
        logging.info('==> Validating the exported pruned model ...')
        loss5, acc5 = model_pruner.validate(model)
        logging.info(
            '==> Final validation accuracy of the exported model: {:.2f}%'.
            format(acc5))


if __name__ == '__main__':
    main()
