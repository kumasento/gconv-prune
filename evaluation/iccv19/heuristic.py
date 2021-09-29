""" Characteristic the heuristic algorithm. """

import argparse
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from gumi.pruning.mask_utils import group_sort, run_mbm
from gumi.model_runner import utils

# Plotting
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

plt.ioff()
plt.rcParams.update({"font.size": 22})
CMAP = plt.cm.inferno

# Seaborn style
import seaborn as sns

sns.set_style("whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(prog="Characteristic heuristic algorithm.")
    parser.add_argument("-s", "--size", type=int, help="Size of the matrix")
    parser.add_argument(
        "--num-samples", default=1000, type=int, help="Size of the matrix"
    )
    parser.add_argument(
        "-g", "--num-groups", nargs="+", type=int, help="Number of groups"
    )
    parser.add_argument(
        "-a", "--archs", nargs="+", type=str, help="Models to be evaluated"
    )
    parser.add_argument(
        "-i", "--num-iters", nargs="*", type=int, default=1, help="Number of iterations"
    )
    parser.add_argument(
        "-m", "--min-g", type=int, default=0, help="Only move to group min_g"
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=100,
        help="Print frequency when sampling data.",
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="data",
        metavar="PATH",
        help="Directory to place all the data files.",
    )

    # options
    parser.add_argument(
        "--draw-model-stats",
        action="store_true",
        default=False,
        help="Whether to draw the model statistics.",
    )
    parser.add_argument(
        "--draw-rand-stats",
        action="store_true",
        default=False,
        help="Whether to draw the random statistics.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Whether to reload previously computed data.",
    )

    return parser.parse_args()


def generate_test_matrix(size, G, scale=10):
    """ First create the original matrix with diagonal blocks larger,
    then randomly permute it. """

    C = np.zeros((size, size))
    SoG = size // G
    for g in range(G):
        R0 = np.random.rand(SoG, SoG)
        R1 = scale * np.random.rand(SoG, SoG)
        C[g * SoG : (g + 1) * SoG, g * SoG : (g + 1) * SoG] += R0 + R1

    # shuffle
    perm_cols = np.random.permutation(size)
    perm_rows = np.random.permutation(size)

    return C, C[perm_rows, :][:, perm_cols]


def plot_mat_to_file(mat, file_name):
    """ """
    fig, ax = plt.subplots()
    cax = ax.matshow(mat)
    fig.colorbar(cax)
    fig.savefig(file_name)


def plot_mats(mats, file_name):
    """ """
    fig, axes = plt.subplots(ncols=len(mats))
    for i in range(len(mats)):
        cax = axes[i].matshow(mats[i], cmap=CMAP)
    fig.savefig(file_name)


def get_cost(C, G):
    """ Split into groups, collect the values on diagonal. """
    SoG = C.shape[0] // G, C.shape[1] // G
    cost = 0.0
    for g in range(G):
        cost += C[g * SoG[0] : (g + 1) * SoG[0], g * SoG[1] : (g + 1) * SoG[1]].sum()
    return cost


def draw_example(C0, C, fp):
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))

    axes[0].set_title("Original")
    axes[0].matshow(C0, cmap=CMAP)
    axes[0].get_xaxis().set_ticks([])
    axes[0].get_yaxis().set_ticks([])

    axes[1].set_title("Permuted")
    axes[1].matshow(C, cmap=CMAP)
    axes[1].get_xaxis().set_ticks([])
    axes[1].get_yaxis().set_ticks([])

    plt.tight_layout()
    fig.savefig(fp)


def draw_step_by_step(C, G, fp, total_cost, num_iters=100):
    """ The step-by-step algorithm illustration graph"""
    min_gs = [G - 1, G // 2, 0]
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6, 6))

    for idx, min_g in enumerate(min_gs):
        ind_in, ind_out = group_sort(C, G, num_iters=num_iters, min_g=min_g)
        C_ = C[ind_out, :][:, ind_in]
        cost = get_cost(C_, G)

        r, c = idx // 2, idx % 2
        axes[r, c].set_title(
            "$min_g = {} ({:.2f}\%)$".format(min_g + 1, cost / total_cost * 100)
        )
        axes[r, c].matshow(C_, cmap=CMAP)
        axes[r, c].get_xaxis().set_ticks([])
        axes[r, c].get_yaxis().set_ticks([])

    # TODO: don't show these
    # result from run MBM
    gnd_in, gnd_out, cost = run_mbm(C, G, perm="GRPS", num_iters=num_iters)
    ind_in = [i for l in gnd_in for i in l]
    ind_out = [i for l in gnd_out for i in l]
    C_ = C[ind_out, :][:, ind_in]

    axes[1, 1].set_title("$MBM({:.2f}\%)$".format(get_cost(C_, G) / total_cost * 100))
    axes[1, 1].matshow(C_, cmap=CMAP)
    axes[1, 1].get_xaxis().set_ticks([])
    axes[1, 1].get_yaxis().set_ticks([])

    plt.tight_layout()
    fig.savefig(fp)


def draw_stats(size, G, data_dir, num_iters=None, num_samples=500, **kwargs):
    """ Collect the performance result """
    if not num_iters:
        num_iters = [1]

    # sns.set_style('whitegrid')
    fp = os.path.join(
        data_dir,
        "random_stats_NI_{}_NS_{}.pdf".format(
            "-".join([str(i) for i in num_iters]), num_samples
        ),
    )
    fig, ax = plt.subplots(figsize=(5, 4))

    for ni in num_iters:
        print("Figuring out num_iters={}".format(ni))
        ratios = collect_random_stats(
            size, G, data_dir, num_iters=ni, num_samples=num_samples, **kwargs
        )

        # plot the histogram
        sns.distplot(ratios, kde=False, label="$N_S={}$".format(ni), ax=ax)

    ax.set_xlabel("Ratio")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fp)


def collect_random_stats(
    size, G, data_dir, resume=False, num_samples=100, num_iters=100, print_freq=100
):
    """ Collect the statistics from randomly sampled test matrices.
  
    If resume is specified, we will load from the data file. 
  """
    # where the data file is stored
    fp = os.path.join(
        data_dir, "random_stats_NI_{}_NS_{}.npy".format(num_iters, num_samples)
    )

    # Decide how to deal with the stats data
    if resume:
        assert os.path.isfile(fp)
        ratios = np.load(fp)
    else:
        ratios = np.zeros(num_samples)  # where to store result.

        for i in range(num_samples):
            if i % print_freq == 0:
                print("[{}/{}] Sampling ...".format(i, num_samples))

            C0, C = generate_test_matrix(size, G)

            gnd_in, gnd_out, cost = run_mbm(C, G, perm="GRPS", num_iters=num_iters)
            ind_in = [i for l in gnd_in for i in l]
            ind_out = [i for l in gnd_out for i in l]
            C_ = C[ind_out, :][:, ind_in]

            ratios[i] = get_cost(C_, G) / get_cost(C0, G)

    # save to file
    np.save(fp, ratios)

    return ratios


def draw_model_stats(arch, grps, data_dir, num_iters=None):
    """ Draw the statistics of several models """
    if not num_iters:
        num_iters = [1]
    fp = os.path.join(
        data_dir,
        "model_stats_{}_NI_{}_G_{}.pdf".format(
            arch,
            "-".join([str(ni) for ni in num_iters]),
            "-".join([str(g) for g in grps]),
        ),
    )

    print("Plot to file: {}".format(fp))

    fig, ax = plt.subplots(figsize=(5, 4))

    print("Running on model {} ...".format(arch))

    model = utils.load_model(arch, "imagenet", pretrained=True)
    results = {"num_iters": [], "num_groups": [], "ratio": []}

    for ni in num_iters:

        for G in grps:
            print("G = {} NI = {}".format(G, ni))

            mods = {}

            # Collect statistics for a single model
            for name, mod in model.named_modules():
                if not isinstance(mod, nn.Conv2d):
                    continue

                W = mod.weight
                F, C = W.shape[:2]

                if F % G != 0 or C % G != 0:
                    continue

                C = W.norm(dim=(2, 3)).cpu().detach().numpy()
                gnd_in, gnd_out, cost = run_mbm(C, G, perm="GRPS", num_iters=ni)
                mods[name] = (cost, C.sum(), cost / C.sum() * 100)

                # print('{:30s}\t {:.2e}\t {:.2e}\t {:.2f}%'.format(
                #     name, mods[name][0], mods[name][1], mods[name][2]))

            # Summarise results
            sum_cost = sum([val[0] for val in mods.values()])
            total_cost = sum([val[1] for val in mods.values()])

            results["num_iters"].append("$N_S={}$".format(ni))
            results["num_groups"].append("$G={}$".format(G))
            results["ratio"].append(sum_cost / total_cost * 100)

    df = pd.DataFrame(results)
    sns.barplot(x="num_groups", y="ratio", hue="num_iters", data=df)

    ax.legend()
    plt.tight_layout()
    fig.savefig(fp)

    df.to_csv(fp.replace(".pdf", ".csv"))


def main():
    args = parse_args()

    # initialise the directory
    print("===> Initialising data directory ...")
    os.makedirs(args.dir, exist_ok=True)

    # C0, C = generate_test_matrix(args.size, args.num_groups)

    # ind_in, ind_out = group_sort(
    #     C, args.num_groups, num_iters=args.num_iters, min_g=args.min_g)

    # print('===> Plotting the example figure ...')
    # draw_example(C0, C, "original_permuted.pdf")

    # print('===> Plotting the step-by-step figure ...')
    # draw_step_by_step(
    #     C, args.num_groups, "step_by_step.pdf", C0.sum(), num_iters=100)

    if args.draw_rand_stats:
        data_dir = os.path.join(
            args.dir,
            "S_{}_G_{}".format(args.size, "_".join([str(g) for g in args.num_groups])),
        )
        os.makedirs(data_dir, exist_ok=True)

        print("===> Plotting statistics ...")
        draw_stats(
            args.size,
            args.num_groups[0],
            data_dir,
            resume=args.resume,
            print_freq=args.print_freq,
            num_samples=args.num_samples,
            num_iters=args.num_iters,
        )

    if args.draw_model_stats:
        data_dir = os.path.join(args.dir, "model_stats")
        os.makedirs(data_dir, exist_ok=True)

        print("===> Plot model statistics ...")
        draw_model_stats(
            args.archs[0], args.num_groups, data_dir, num_iters=args.num_iters
        )


if __name__ == "__main__":
    main()
