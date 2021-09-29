""" This script tries to calculate the number of parameters
  and operations used in [Peng18].

  Basically, we load a ResNet-34 model, without any parameters.
  Then we iterate the group config file to replace nn.Conv2d
  with nn.Conv2d plus a pointwise convolution.
"""

import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from gumi.ops import *
from gumi.model_runner.parser import create_cli_parser
from gumi.model_runner import utils
from gumi import model_utils

parser = create_cli_parser(prog="Peng18 statistics calculator.")
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Whether to use CUDA"
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
cudnn.benchmark = True


def update_model(model, group_cfg, use_cuda=True):
    """ Update the model based on group_cfg.

  NOTE: model is updated in place.
  """
    for name, mod in model.named_modules():

        mod_map = {}
        for child_name, child in mod.named_children():
            if not isinstance(child, nn.Conv2d):
                continue

            name_ = name + "." + child_name  # actual name
            if not use_cuda:
                name_ = "module." + name_

            if name_ in group_cfg:
                cfg_ = group_cfg[name_]
                if cfg_["G"] == 1:
                    continue

                mod_map[child_name] = GroupPointwiseConv(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    cfg_["G"],
                    stride=child.stride,
                    padding=child.padding,
                    bias=child.bias,
                )

        for child_name, child in mod_map.items():
            mod._modules[child_name] = child

    if torch.cuda.is_available() and use_cuda:
        model = model.cuda()
    return model


def main():
    use_cuda = not args.cpu
    print("==> Loading model {}".format(args.arch))
    model = utils.load_model(args.arch, "imagenet", use_cuda=use_cuda, pretrained=True)

    print("==> Loading group config {}".format(args.group_cfg))
    with open(args.group_cfg, "r") as f:
        group_cfg = json.load(f)

    print("==> Updating model ...")
    model = update_model(model, group_cfg, use_cuda=use_cuda)

    print(model)

    print(
        "==> Model size: {:.2f} M ops: {:.2f} M".format(
            model_utils.get_model_num_params(model),
            utils.get_model_num_ops(model, "imagenet"),
        )
    )

    torch.save(model, os.path.join(os.path.dirname(args.resume), "peng.pth.tar"))


if __name__ == "__main__":
    main()
