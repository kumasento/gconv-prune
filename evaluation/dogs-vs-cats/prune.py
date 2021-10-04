#!/usr/bin/env python
import copy
import logging
import os

import torch
from gumi import group_utils, model_utils
from gumi.config import GumiConfig
from gumi.model_runner import utils
from gumi.model_runner.model_pruner import ModelPruner
from gumi.model_runner.parser import create_cli_parser
from gumi.pruning.export import GroupExporter

from utils import get_datasets, set_random_seed

fmt = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s: %(message)s"
logging.basicConfig(format=fmt, level=logging.DEBUG)


def patch_dataset(cfg: GumiConfig) -> GumiConfig:
    if cfg.dataset != 'dogs-vs-cats':
        return cfg
    cfg.dataset = get_datasets(cfg.dataset_dir)
    return cfg


def update_state_dict(state_dict: dict):
    """ Here are several update rules:

    - In this new script, we won't have "module." prefix
    - There won't be any '.conv2d' in the module
    """
    state_dict_ = copy.deepcopy(state_dict)

    for key, val in state_dict.items():
        key_ = key

        if "module" in key_:
            del state_dict_[key_]
            key_ = key_.replace("module.", "")
            state_dict_[key_] = val

        if "conv2d" in key_:
            del state_dict_[key_]
            key_ = key_.replace(".conv2d", "")
            state_dict_[key_] = val

    return state_dict_


def main():
    parser = create_cli_parser()
    args = parser.parse_args()

    set_random_seed(42)

    cfg = GumiConfig(**vars(args))
    cfg = patch_dataset(cfg)

    logging.info('Initializing ModelPruner ...')
    model_pruner = ModelPruner(cfg)

    model = model_pruner.load_model(update_state_dict_fn=update_state_dict)
    model_pruner.validate(model, record_top5=False)

    logging.info("==> Replacing Conv2d in model by MaskConv2d ...")
    utils.apply_mask(model)
    model_pruner.validate(model, record_top5=False)

    logging.info("==> Pruning model ...")
    get_num_groups = group_utils.create_get_num_groups_fn(
        G=cfg.num_groups, MCPG=cfg.mcpg, group_cfg=cfg.group_cfg)

    model_pruner.prune(
        model,
        get_num_groups_fn=get_num_groups,
        perm=cfg.perm,
        no_weight=cfg.no_weight,
        num_iters=cfg.num_sort_iters,
        keep_mask=False,
    )
    model_pruner.validate(model, record_top5=False)

    logging.info("==> Fine-tuning model ...")
    model_pruner.fine_tune(model, record_top5=False)
    model_pruner.validate(model, record_top5=False)

    logging.info("==> Exporting the model ...")
    model = GroupExporter.export(model)
    logging.debug("Total params: {:.2f}M FLOPS: {:.2f}M".format(
        model_utils.get_model_num_params(model),
        utils.get_model_num_ops(model, args.dataset),
    ))

    logging.info("==> Saving exported model ...")
    torch.save({'state_dict': model.state_dict()},
               os.path.join(cfg.checkpoint, 'pruned.pth.tar'))


if __name__ == '__main__':
    main()
