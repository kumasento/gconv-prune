""" Argument parser. """

import argparse

from gumi.model_runner.utils import model_names


def create_cli_parser(prog=""):
    """ Generate the argument parser. """
    parser = argparse.ArgumentParser(prog=prog)

    # pruning parameters
    # Group configuration
    parser.add_argument(
        "-g",
        "--num-groups",
        type=int,
        default=0,
        help="Uniform number of groups to be pruned.",
    )
    parser.add_argument("--mcpg", type=int, default=0, help="Max channels per group.")
    parser.add_argument(
        "--group-cfg",
        type=str,
        metavar="PATH",
        default="",
        help="Path to group configuration file to be generated.",
    )
    parser.add_argument(
        "--ind",
        type=str,
        default="none",
        help="How to generate indices around group convolution.",
    )

    # mask
    parser.add_argument(
        "--perm", type=str, default="GRPS", help="Permutation when building MBM mask."
    )
    parser.add_argument(
        "--num-sort-iters", type=int, default=10, help="Number of heuristic iterations."
    )

    # regularization
    parser.add_argument(
        "--reg-scale", type=float, default=0.0, help="Regularization scale, can be 0.0"
    )
    parser.add_argument(
        "--reg-epochs",
        type=int,
        default=0,
        help="Number of epochs for regularization, by default 0",
    )
    parser.add_argument(
        "--reg-lr", type=float, default=1e-3, help="Regularization fixed learning rate."
    )

    # no weight
    parser.add_argument(
        "--no-weight",
        default=False,
        action="store_true",
        help="Simply figure out the mask and don not load weight.",
    )

    # no permute
    parser.add_argument(
        "--no-perm",
        default=False,
        action="store_true",
        help="Whether we should use the permutation or not",
    )

    # no data loader
    parser.add_argument(
        "--no-data-loader",
        default=False,
        action="store_true",
        help="Do not initialise data loaders in CTOR.",
    )

    parser.add_argument(
        "--scratch",
        default=False,
        action="store_true",
        help="Simply figure out the mask and don not load weight.",
    )

    # where to store the new checkpoint file
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="The path to the checkpoint file."
    )

    parser.add_argument(
        "--resume",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--resume-from-best",
        action="store_true",
        default=False,
        help="Whether to restore from the model_best.pth.tar",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Use pretrained models for initialization",
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet110",
        choices=model_names,
        help="model architecture: " + ", ".join(model_names) + "(default: resnet110)",
    )
    parser.add_argument("-d", "--dataset", default="cifar10", type=str)
    parser.add_argument(
        "--dataset-dir", default="data", help="Path to dataset", type=str
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )
    # Optimization options
    parser.add_argument(
        "--epochs",
        default=300,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--train-batch", default=128, type=int, metavar="N", help="train batch size"
    )
    parser.add_argument(
        "--test-batch", default=100, type=int, metavar="N", help="test batch size"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr-type", default="multistep", type=str, help="Type of learning rate decay"
    )
    parser.add_argument(
        "--drop",
        "--dropout",
        default=0,
        type=float,
        metavar="Dropout",
        help="Dropout ratio",
    )
    parser.add_argument(
        "--schedule",
        type=int,
        nargs="+",
        default=[150, 225],
        help="Decrease learning rate at these epochs.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="LR is multiplied by gamma on schedule.",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=5e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--gpu-id", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument(
        "--no-bar",
        default=False,
        action="store_true",
        help="Whether to show bar progress.",
    )
    parser.add_argument(
        "--print-freq", default=50, type=int, help="Log output frequency"
    )

    return parser
