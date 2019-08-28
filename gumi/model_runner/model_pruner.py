""" A pruner wrapped upon ModelRunner. """
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
from gumi.pruning import prune_utils
from gumi.model_runner import utils
from gumi.model_runner.model_runner import ModelRunner


class ModelPruner(ModelRunner):
    """ Pruning utilities implemented. """

    def __init__(self, args):
        """ CTOR. """
        super().__init__(args)

    def prune(self, model, G=0, get_num_groups_fn=None, use_cuda=True,
              **kwargs):
        """ The main prune function. 
    
    Now we assume that all the modules that can be pruned
    are replaced by MaskConv2d.

    We need to set:
      G - the number of groups, decided by get_num_groups_fn
      mask - the actual mask value of each module,
        based on prune_utils.create_mbm_mask()
    
    Note:
      model will be pruned in-place.

    Args:
      model(nn.Module)
      get_num_groups_fn(function): a function that can return
        the group size of a given module. Its inputs are 
        mod_name, module.
    """
        assert isinstance(model, nn.Module)
        assert (get_num_groups_fn is not None) or G > 0

        for name, mod in model.named_modules():
            # run prune_module over every MaskConv2d module
            if isinstance(mod, MaskConv2d):
                G_ = get_num_groups_fn(name, mod) if get_num_groups_fn else G
                self.prune_module(name, mod, G_, **kwargs)

        if torch.cuda.is_available() and use_cuda:
            model.cuda()  # update the modules' devices

    def prune_module(self, name, mod, G, **kwargs):
        """ Prune a specific module.
            NOTE: G is known at this moment.
        """
        assert isinstance(mod, MaskConv2d)
        assert G >= 1, '{} has G={} smaller than 1'.format(name, G)

        W = model_utils.get_weight_parameter(mod)
        C_out, C_in = W.shape[:2]

        if G == 1 or (C_out % G != 0 or C_in % G != 0):
            # NOTE: we return if this module cannot be pruned
            return

        prune_utils.prune_module(mod, G=G, **kwargs)

    def collect_gradient(self, model, **kwargs):
        """ Collect gradient """
        if torch.cuda.is_available():
            model.cuda()  # in case the model is not on CUDA yet.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare
        best_acc = 0
        epochs = self.args.epochs
        base_lr = self.args.lr
        optimizer = optim.SGD(model.parameters(),
                              lr=base_lr,
                              momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)

        train_loss, train_acc = utils.train(self.train_loader,
                                            model,
                                            self.criterion,
                                            optimizer,
                                            0,
                                            max_iters=1,
                                            print_freq=self.args.print_freq,
                                            gpu=device,
                                            state=self.state,
                                            schedule=self.args.schedule,
                                            epochs=self.args.epochs,
                                            base_lr=self.args.lr,
                                            gamma=self.args.gamma,
                                            lr_type=self.args.lr_type)

    def fine_tune(self,
                  model,
                  min_val_acc=None,
                  return_init_acc=False,
                  max_iters=None,
                  **kwargs):
        """ Fine tune a pruned model. """
        if torch.cuda.is_available():
            model.cuda()  # in case the model is not on CUDA yet.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare
        best_acc = 0
        epochs = self.args.epochs
        base_lr = self.args.lr
        optimizer = optim.SGD(model.parameters(),
                              lr=base_lr,
                              momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)

        # validation in the beginning
        val_loss, val_acc = utils.validate(self.val_loader,
                                           model,
                                           self.criterion,
                                           gpu=device,
                                           print_freq=self.args.print_freq)
        init_acc = val_acc
        best_acc = val_acc
        best_model = None

        for epoch in range(epochs):
            # TODO(13/02/2019): learning rate adjustment
            # self.adjust_learning_rate(epoch, optimizer)

            logging.debug('Epoch: [%d | %d] LR: %f' %
                          (epoch + 1, epochs, self.state['lr']))

            # Run train and validation for one epoch
            train_loss, train_acc = utils.train(self.train_loader,
                                                model,
                                                self.criterion,
                                                optimizer,
                                                epoch,
                                                max_iters=max_iters,
                                                print_freq=self.args.print_freq,
                                                gpu=device,
                                                state=self.state,
                                                schedule=self.args.schedule,
                                                epochs=self.args.epochs,
                                                base_lr=self.args.lr,
                                                gamma=self.args.gamma,
                                                lr_type=self.args.lr_type)

            val_loss, val_acc = utils.validate(self.val_loader,
                                               model,
                                               self.criterion,
                                               gpu=device,
                                               print_freq=self.args.print_freq)

            # Append message to Logger
            self.logger.append([
                self.state['lr'], train_loss, 0.0, val_loss, train_acc, val_acc
            ])

            # Update best accuracy
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                best_model = copy.deepcopy(model)

            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            utils.save_checkpoint(checkpoint_state, is_best,
                                  self.args.checkpoint)

            if min_val_acc is not None and val_acc >= min_val_acc:
                break
            # for name, mod in model.named_modules():
            #   if isinstance(mod, MaskConv2d):
            #     print(name, torch.nonzero(mod.weight.data).size(0))

        # Finalising
        self.logger.close()
        logging.info(
            'Best accuracy while fine-tuning: {:.2f}%'.format(best_acc))

        if not return_init_acc:
            return best_acc, best_model
        else:
            return best_acc, init_acc, best_model
