""" Optimisation module for group size.

The objective of this module is to dump a group configuration, 
which could be the optimal or the sub-optimal.

Directly copied from early_stage/
"""

import argparse
import copy
import functools
import itertools
import json
import logging
import os
import shutil
import sys
import time
from collections import OrderedDict
from subprocess import PIPE, Popen  # launching pruning processes

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from gumi import model_utils
from gumi.model_runner import utils
from gumi.model_runner.model_runner import ModelRunner
from gumi.model_runner.parser import create_cli_parser
from gumi.ops import *
from gumi.pruning import mask_utils
from pulp import *

logging.getLogger().setLevel(logging.DEBUG)

parser = create_cli_parser(prog="Generate an optimised group configuration.")
parser.add_argument(
    "--dump-only",
    action="store_true",
    default=False,
    help="Whether to simply dump the group configuration.",
)
parser.add_argument(
    "--strategy", type=str, default="ILP", help="How to solve the optimisation problem."
)
parser.add_argument(
    "--min-factor",
    type=float,
    default=0.0,
    help="Minimum channels that should appear in each group.",
)
parser.add_argument(
    "--max-groups", type=int, default=64, help="Maximum number of groups allowed"
)
parser.add_argument(
    "--allow-depthwise", action="store_true", default=False, help="Allow depthwise"
)
parser.add_argument(
    "--absolute", action="store_true", default=False, help="Use absolute cost"
)
parser.add_argument(
    "--max-num-params", type=float, default=100, help="Maximal number of parameters (M)"
)
parser.add_argument(
    "--max-num-ops", type=float, default=1000, help="Maximal number of operations (M)"
)
parser.add_argument(
    "--excludes-for-applying-mask",
    nargs="+",
    help="Excluded module names for applying mask",
)

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


class GOptModelRunner(ModelRunner):
    """ Group OPT model runner.
    We simply override validate_args to avoid validating arguments
  """

    def validate_args(self, args):
        pass


class GOpt(object):
    """ Group size optimizer. """

    def __init__(self, args):
        """ CTOR. """
        self.args = args

    def find_groupable_modules(self, model):
        """ Find modules that can be grouped.

      This groupability is defined by whether its output and 
      input channels can be factorized into lists that have
      intersection. 
    """
        mods = []  # return list

        for name, mod in model.named_modules():

            if isinstance(mod, MaskConv2d):
                F = mod.out_channels
                C = mod.in_channels
                ff, fc = factors(F), factors(C)

                if len(ff.intersection(fc)) <= 1:
                    continue

                mods.append((name, mod))

        return mods

    def find_group_candidates(
        self,
        mod,
        relative=False,
        allow_depthwise=False,
        min_factor=None,
        max_groups=None,
        **kwargs
    ):
        """ Find group number candidates in module.
    
    Note: use kwargs to pass additional requirements.
    """
        assert isinstance(mod, MaskConv2d)
        C = mod.in_channels
        F = mod.out_channels
        W = mod.weight

        # common divisors
        Gs = list(sorted(factors(F).intersection(factors(C))))
        if max_groups:
            while Gs[-1] > max_groups:  # TODO refactorize
                Gs.pop()

        # assert Gs[0] == 1
        # del Gs[0]  # should be 1
        if not allow_depthwise:
            if Gs[-1] == F and Gs[-1] == C:
                del Gs[-1]
        if min_factor is not None:
            # remove those factors that are below a threshold
            while len(Gs) >= 2:
                factor = min(F / Gs[-1], C / Gs[-1])
                if factor < min_factor:
                    Gs.pop()
                else:
                    break

        # get the best cost for all group candidates
        costs = []
        for G in Gs:
            _, _, cost = mask_utils.run_mbm(W, G, **kwargs)
            if relative:
                cost = cost / W.norm(dim=(2, 3)).sum().item()

            costs.append(cost)

        return Gs, costs

    def get_cost(self, state, c_dict):
        """ Collect the sum of cost by iterating every module in state. """
        costs = []
        total = 0.0
        for name, gi in state.items():
            costs.append(c_dict[name][1][gi])
            total += c_dict[name][2]

        # HACK
        if self.args.absolute:
            return np.sum(costs) / total

        return np.mean(costs)

    def assign_state_to_model(self, model, state, c_dict):
        """ Assign G to modules in model. """
        for name, mod in model.named_modules():
            if name in state:
                mod.G = c_dict[name][0][state[name]]
        if use_cuda:  # post-fix model placing
            model.cuda()

    def get_mod_norm(self, mod):
        """ """
        assert isinstance(mod, MaskConv2d)
        return mod.weight.norm(dim=(2, 3)).sum()

    def ilp_solve(self, c_dict, max_num_params, num_num_ops):
        """ Solve the optimisation problem by ILP. """
        logging.info("==> Building an ILP solver ...")
        prob = LpSolver("Maximize cost under params and ops constraints", LpMaximize)

        lp_vars = {}
        lp_bins = {}
        for name, (Gs, costs, norm) in c_dict.items():
            # A list of binary values represents whether we should select to use a G or not
            N = len(Gs)
            lp_bins[name] = LpVariable.dicts("{}_G=%s".format(name), N, 0, 1, LpInteger)
            lp_vars[name] = LpVariable(name)  # the cost

            prob += lp_vars[name]

        return None

    def max_cost_solver(self, model, state, c_dict, max_num_params, max_num_ops):
        num_params = model_utils.get_model_num_params(model)
        num_ops = utils.get_model_num_ops(model, self.args.dataset)
        cost = self.get_cost(state, c_dict)

        if num_params > max_num_params or num_ops > max_num_ops:
            raise ValueError(
                "The maximal constraints are too restrictive: params {:.2f} ops {:.2f}.".format(
                    num_params, num_ops
                )
            )

        step = 0
        while num_params <= max_num_params and num_ops <= max_num_ops:
            # we can have a try
            costs = []
            for name, gi in state.items():
                # try reduce
                if gi == 0:
                    continue

                state_ = state.copy()
                state_[name] = gi - 1  # this will definitely increase the cost
                cost_ = self.get_cost(state_, c_dict)
                costs.append((name, cost_))

            if not costs:
                logging.debug("No more choices for MAX_COST, exiting ...")
                break

            # find the max cost update
            max_cost = max(costs, key=lambda k: k[1])
            # print(costs)
            # print(max_cost)

            # update the state
            state[max_cost[0]] -= 1
            cost = max_cost[1]

            # update
            self.assign_state_to_model(model, state, c_dict)
            if use_cuda:  # post-fix model placing
                model.cuda()
            num_params = model_utils.get_model_num_params(model)
            num_ops = utils.get_model_num_ops(model, self.args.dataset)

            step += 1
            logging.debug(
                "[{:4d}] Current state -  params: {:.2f} M ops: {:6.2f} M cost: {:.2f}%".format(
                    step, num_params, num_ops, (1 - cost) * 100
                )
            )

        return state, cost

    def run_opt(
        self, model, max_num_params, max_num_ops, strategy="MAX_COST", **kwargs
    ):
        """ Run the actual optimization.
    
      Model is already loaded in this case. All the modules that should be grouped 
      are already replaced by MaskConv2d.

      We will update the G value in each module in place.
    """
        logging.info("Finding the optimal group configuration ...")
        logging.info("Max # param: {:.2f} M".format(max_num_params))
        logging.info("Max # ops:   {:.2f} M".format(max_num_ops))

        c_dict = OrderedDict()  # candidate dictionary
        state = OrderedDict()  # G index for each group

        logging.debug("==> Constructing the candidate dictionary ...")
        for name, mod in model.named_modules():
            if isinstance(mod, MaskConv2d):
                Gs, costs = list(self.find_group_candidates(mod, **kwargs))
                c_dict[name] = (Gs, costs, self.get_mod_norm(mod))
                state[name] = len(Gs) - 1  # the last group

        # Initial state
        self.assign_state_to_model(model, state, c_dict)

        logging.debug(
            "Initial state: {:.2f} M params {:.2f} M ops cost: {:.2f}%".format(
                model_utils.get_model_num_params(model),
                utils.get_model_num_ops(model, self.args.dataset),
                (1 - self.get_cost(state, c_dict)) * 100,
            )
        )

        # Now, depends on the strategy, we will update the group setting
        if strategy == "ILP":
            # HACK - need to refactorise the code
            # state = self.ilp_solve(c_dict)
            raise NotImplementedError(
                "ILP as a optimisation strategy is not implemented"
            )
        elif strategy == "MAX_COST":
            state, cost = self.max_cost_solver(
                model, state, c_dict, max_num_params, max_num_ops
            )
        else:
            raise ValueError("Cannot recognise strategy: {}".format(strategy))

        # finalise
        self.assign_state_to_model(model, state, c_dict)
        num_params = model_utils.get_model_num_params(model)
        num_ops = utils.get_model_num_ops(model, self.args.dataset)

        logging.debug(
            "Final state - params: {:.2f} M  ops: {:6.2f} M cost: {:.2f}".format(
                num_params, num_ops, (1 - cost) * 100
            )
        )

    def dump_group_conf(self, model, file_name):
        """ Dump the group configuration of a model to a file. """
        logging.info("Dumping group config to {} ...".format(file_name))

        g_cfg = OrderedDict()
        for idx, (name, mod) in enumerate(model.named_modules()):
            # iterate every module, dump them
            if isinstance(mod, MaskConv2d):
                # logging.info('Dumping data for mod {}:'.format(name))
                # logging.info('\t{}'.format(mod))

                name_ = "module.{}".format(name)
                g_cfg[name_] = {
                    "G": mod.G,
                    "F": mod.out_channels,
                    "C": mod.in_channels,
                    "id": idx,
                }

        dirname = os.path.dirname(file_name)
        os.makedirs(dirname, exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(g_cfg, f, indent=2)


def create_update_state_dict_fn(no_mask=False):
    def update_state_dict(state_dict):
        """ Here are several update rules:
    
      - In this new script, we won't have "module." prefix
      - There won't be any '.conv2d' in the module
    """
        logging.debug("Updating the state_dict to be loaded ...")
        state_dict_ = copy.deepcopy(state_dict)

        for key, val in state_dict.items():
            key_ = key

            if "conv2d" in key_:
                del state_dict_[key_]
                key_ = key_.replace(".conv2d", "")
                state_dict_[key_] = val

            if "module" in key_:
                del state_dict_[key_]
                key_ = key_.replace("module.", "")
                state_dict_[key_] = val

            if no_mask and "mask" in key_:
                del state_dict_[key_]

        return state_dict_

    return update_state_dict


def create_update_model_fn(
    arch, dataset, pretrained, resume, excludes_for_applying_mask=None
):
    """ """

    def update_model(model):
        """ """
        # if not resume:
        #   return model
        if dataset.startswith("cifar"):
            # apply mask right now
            utils.apply_mask(model, excludes=excludes_for_applying_mask)

        return model

    return update_model


def main():
    """ Main runner. """
    gopt = GOpt(args)

    ## Load the model
    runner = GOptModelRunner(args)
    update_model_fn = create_update_model_fn(
        args.arch,
        args.dataset,
        args.pretrained,
        args.resume,
        excludes_for_applying_mask=args.excludes_for_applying_mask,
    )
    model = runner.load_model(
        update_model_fn=update_model_fn,
        update_state_dict_fn=create_update_state_dict_fn(no_mask=not args.resume),
        data_parallel=False,
    )

    if use_cuda:  # post-fix model placing
        model.cuda()
    # utils.apply_mask(model)

    if not args.dump_only:
        gopt.run_opt(
            model,
            args.max_num_params,
            args.max_num_ops,
            min_factor=args.min_factor,
            max_groups=args.max_groups,
            strategy=args.strategy,
            relative=not args.absolute,
            allow_depthwise=args.allow_depthwise,
            perm="GRPS",
            num_iters=10,
        )

    fp = args.group_cfg
    # NOTE - just store the data locally
    # if isinstance(args.checkpoint, str) and os.path.isdir(args.checkpoint):
    #   fp = os.path.join(args.checkpoint, 'group_cfg.json')
    #   logging.info(
    #       '==> Checkpoint directory is provided: {}, writing to {} ...'.format(
    #           args.checkpoint, fp))

    gopt.dump_group_conf(model, fp)


if __name__ == "__main__":
    main()
