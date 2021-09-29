""" This script tries different positions of pruning in the initial step. """

import copy
import functools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel

from gumi import model_utils
from gumi.model_runner import utils
from gumi.model_runner.model_runner import ModelRunner
from gumi.model_runner.model_pruner import ModelPruner
from gumi.model_runner.parser import create_cli_parser
from gumi.ops import MaskConv2d
from gumi.pruning import mask_utils

logging.getLogger().setLevel(logging.DEBUG)

parser = create_cli_parser(prog="Debug the choice of the first pruning step.")
parser.add_argument(
    "--excludes-for-applying-mask",
    nargs="+",
    help="Excluded module names for applying mask",
)
parser.add_argument(
    "--manual-seed", default=None, type=int, help="Manual seed for reproducibility."
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
    "--taylor",
    action="store_true",
    default=False,
    help="Whether to use the Taylor pruning criterion",
)
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True

# TODO:  Move it somewhere else
# Follows: https://pytorch.org/docs/stable/notes/randomness.html
if args.manual_seed is not None:
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomModelRunner(ModelRunner):
    """ Group OPT model runner.
        We simply override validate_args to avoid validating arguments
    """

    def validate_args(self, args):
        pass


class FirstStepDebugger(object):
    """ Debug the pruning performance at the first step. """

    def __init__(self, args):
        self.args = args

    def factors(self, n):
        """
    Copied from - https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
    """
        return set(
            functools.reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0),
            )
        )

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

        # common divisors
        Gs = list(sorted(self.factors(F).intersection(self.factors(C))))
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

        return Gs

    def build_state_map(self, model, min_factor=None, max_groups=None):
        """ Build the state map based on the given model. """
        assert isinstance(model, nn.Module)

        state_map = OrderedDict()
        for name, mod in model.named_modules():
            if isinstance(mod, MaskConv2d):  # only MaskConv2d can be grouped
                Gs = self.find_group_candidates(
                    mod, min_factor=min_factor, max_groups=max_groups
                )
                # current index and all candidates
                state_map[name] = [0, list(Gs)]

        return state_map

    def get_next_cost(self, model, state_map, normalized=False, **kwargs):
        """ Get next cost map """
        assert isinstance(model, nn.Module)

        next_costs = OrderedDict()
        for name, mod in model.named_modules():
            if name in state_map:
                state = state_map[name]
                if len(state[1]) != state[0] + 1:  # already at the end
                    next_G = state[1][state[0] + 1]
                    W = mod.weight
                    # run the heuristic algorithm to calculate cost
                    _, _, crit = mask_utils.run_mbm(
                        W, next_G, normalized=normalized, **kwargs
                    )
                    if not normalized:
                        crit /= W.norm(dim=(2, 3)).sum().item()
                        next_costs[name] = 1 - crit
                    else:
                        next_costs[name] = -crit.item()

        return next_costs

    def prune_to_next_state(
        self, mod_name, model, state_map, min_val_acc=None, freeze=False, **kwargs
    ):
        """ Prune model to the next state of the given mod """
        mod = self.find_module(mod_name, model)
        state = state_map[mod_name]
        next_G = state[1][state[0] + 1]
        # update state_map
        state_map[mod_name][0] += 1

        # prune the specific module by next_G
        self.pruner.prune_module(mod_name, mod, next_G, **kwargs)
        if use_cuda:
            model.cuda()
        if freeze:
            self.freeze_other_modules(mod_name, model)
            # unfreeze the current pruning module
            for name, param in mod.named_parameters():
                if "mask" not in name:
                    param.requires_grad = True

        # now model has been pruned
        return self.pruner.fine_tune(
            model, return_init_acc=True, min_val_acc=min_val_acc
        )

    def get_model_size(self, model):
        """ Return model number of parameters and ops """
        num_params = model_utils.get_model_num_params(model)
        num_ops = utils.get_model_num_ops(model, self.args.dataset)

        return num_params, num_ops

    def find_module(self, name, model):
        """ find module by name """
        for name_, m in model.named_modules():
            if name_ == name:
                return m

    def run(
        self,
        model,
        min_factor=None,
        max_groups=None,
        max_epochs=1,
        crit_type=None,
        **kwargs
    ):
        logging.info("==> Building state map for all layers ...")

        self.pruner = ModelPruner(self.args)  # new model pruner
        if crit_type == "taylor":
            self.pruner.collect_gradient(model)
        state_map = self.build_state_map(
            model, min_factor=min_factor, max_groups=max_groups
        )
        next_costs = self.get_next_cost(model, state_map, crit_type=crit_type, **kwargs)

        logging.info("==> All the states:")
        for key, val in state_map.items():
            print("{}\t{:10.6f} {}".format(key, next_costs[key], val))

        # now iterate every module that can be pruned
        for i, key in enumerate(state_map):
            logging.info(
                "==> Exploring module {} at step {} with cost {} ...".format(
                    key, i, next_costs[key]
                )
            )
            # prepare a copy for pruning
            model_ = copy.deepcopy(model)
            state_map_ = copy.deepcopy(state_map)

            # setup pruner
            args = copy.copy(self.args)
            args.epochs = max_epochs  # maximum number of epochs
            args.checkpoint = os.path.join(self.args.checkpoint, "mod_{}".format(i))
            self.pruner = ModelPruner(args)  # new model pruner
            if crit_type == "taylor":
                self.pruner.collect_gradient(model_)

            # run the pruning
            best_acc, init_acc, best_model = self.prune_to_next_state(
                key, model_, state_map_, crit_type=crit_type, **kwargs
            )
            num_params, num_ops = self.get_model_size(best_model)

            # store some data
            meta_data = {
                "mod_name": key,
                "cost": next_costs[key],
                "val_acc": best_acc.item(),
                "init_acc": init_acc.item(),
                "G": state_map_[key][1][state_map_[key][0]],
                "num_params": num_params,
                "num_ops": num_ops,
            }
            meta_file = os.path.join(args.checkpoint, "meta_data.json")
            with open(meta_file, "w") as f:
                json.dump(meta_data, f)


###############################
# Prepare to load model       #
###############################


def create_update_state_dict_fn(no_mask=False):
    """ Create the state_dict that will be used for loading model """

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
    runner = CustomModelRunner(args)
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

    debugger = FirstStepDebugger(args)
    debugger.run(
        model,
        min_factor=args.min_factor,
        max_groups=args.max_groups,
        perm="GRPS",
        num_iters=10,
        crit_type=None if not args.taylor else "taylor",
        normalized=args.taylor,
    )


if __name__ == "__main__":
    main()
