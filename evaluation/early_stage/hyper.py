""" Do the hyperparameter tuning work.
Mainly the hyperparameters for pruning are tuned here.
"""

import os
import sys
import argparse
import copy
import time
import shutil
import json
import itertools
import functools
from subprocess import Popen, PIPE  # launching pruning processes
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

from parser import create_parser
from pruner import Pruner  # pruning runner

parser = create_parser()
# add additional parameters
parser.add_argument(
    "--search", type=str, metavar="PATH", help="Hyperparameter searching configuration."
)
parser.add_argument(
    "--dir", type=str, metavar="PATH", help="Where to store the search results."
)

# parse the input arguments
args = parser.parse_args()


class HyperParamSearcher(object):
    """ Runs hyperparameter searching. """

    def __init__(self, args):
        """ CTOR. Load the configuration file. """
        with open(args.search, "r") as f:  # load the configure file
            self.search_space = json.loads(f.read())

        logging.debug(
            "The provided search space is: {}".format(
                json.dumps(self.search_space, indent=2, sort_keys=True)
            )
        )

        self.init_search_dir(args)
        self.num_gpus = len(args.gpu_id.split(","))
        self.gpu_ids = args.gpu_id.split(",")
        self.args = args
        self.args_list = self.create_args_search_list(self.args, self.search_space)

        # execution specific
        self.sleep_for_secs = 1

    def init_search_dir(self, args):
        """ Initialise search directory. """
        os.makedirs(args.dir, exist_ok=True)

    def create_args_search_list(self, args, search_space):
        """ Create a candidate list of args.
    
    Args:
      args(object): original command line argument
      search_space(dict): search space
    Returns:
      A list of args for future run
    """
        args_list = []

        # parse GPU id to support parallel run
        gpu_ids = args.gpu_id.split(",")  # each ID is a str
        cur_gpu = 0  # current GPU

        # figure out the keys
        keys = self.get_search_space_keys(search_space)

        # make the product between lists of keys in search space
        for prod in itertools.product(*search_space.values()):
            if len(prod) == 0:
                continue

            # flatten the values
            values = self.flatten_search_space_values(prod)
            # create a new argument
            new_args = self.create_new_args(keys, values, args)
            # update the GPU id, should support parallel run
            new_args.gpu_id = gpu_ids[cur_gpu]
            cur_gpu = (cur_gpu + 1) % len(gpu_ids)

            logging.debug("new_args={}".format(new_args.__dict__))

            # update to the returned argument list
            args_list.append(new_args)

        return args_list

    def encode_search_space_value(self, value):
        """ Sometimes a value cannot be encoded into a path. """
        codec = ""
        if isinstance(value, str):
            codec = value.replace(" ", "-")
            codec = codec.replace(",", "-")
        elif isinstance(value, list):
            codec = "-".join([str(x) for x in value])
        else:
            codec = str(value)

        return codec

    def get_checkpoint(self, keys, values, args):
        """ Get the checkpoint path from existing search key and value. """
        checkpoint = args.dir  # that's the prefix
        suffix = "_".join(
            [
                "{}_{}".format(k, self.encode_search_space_value(v))
                for k, v in zip(keys, values)
            ]
        )
        checkpoint += "/" + suffix

        return checkpoint

    def get_search_space_keys(self, search_space):
        """ """
        keys = []
        for rk in search_space.keys():
            keys.extend([k.strip() for k in rk.split(",")]),
        return keys

    def flatten_search_space_values(self, prod):
        """ Flatten values in a product """
        values = []
        for p in prod:
            if isinstance(p, list):
                values.extend(p)
            else:
                values.append(p)
        return values

    def create_new_args(self, keys, values, args):
        """ Create a new_args from existing ones. """
        new_args = copy.deepcopy(args)
        for i in range(len(values)):  # update the args from search space
            new_args.__dict__[keys[i]] = values[i]

        # update checkpoint
        new_args.checkpoint = self.get_checkpoint(keys, values, args)
        return new_args

    def create_prune_process(self, args):
        """ Run the pruning process from a subprocess.
    
      We should convert the arguments from args to a argv list.
    """
        log_file = open(args.checkpoint + ".log", "w")
        argv = ["python", "prune.py"]

        for k, v in args.__dict__.items():
            if k in ["search", "dir"]:
                continue

            key = "--" + k.replace("_", "-")

            value = v
            if isinstance(value, list):
                value = [str(x) for x in value]
            elif isinstance(value, bool):
                if not value:
                    continue  # HACK - we skip the key
                value = []
            else:
                value = [str(value)]

            argv.append(key)
            argv.extend(value)

        logging.debug("==> Processing pruning with arguments: {}".format(argv))
        return Popen(argv, stdout=log_file, stderr=log_file)

    def load_summary_from_checkpoint(self, args):
        """ load summary from checkpoint's summary file. """
        summary_file = args.checkpoint + "/summary.json"
        if not os.path.isfile(summary_file):
            raise RuntimeError(
                "Summary file cannot be found at {}".format(summary_file)
            )

        with open(summary_file, "r") as f:
            dat = json.load(f)  # dat is a dict
        return dat

    def run(self):
        """ Run the searching process. """
        logging.info("Launching subprocesses with GPUs={} ...".format(self.gpu_ids))

        summary = {}  # overall summary

        # split arguments by GPU ID
        # each queue will be scheduled separately
        args_list_queue = [
            [args for args in self.args_list if args.gpu_id == gpu_id]
            for gpu_id in self.gpu_ids
        ]

        # record the current ID of task in each queue
        qid = 0  # current queue ID
        num_queues = len(args_list_queue)
        is_queue_done = [len(q) == 0 for q in args_list_queue]
        task_in_queue = [0] * num_queues
        task_to_proc = [{} for _ in range(num_queues)]

        logging.info("Number of queues for processing: {}".format(num_queues))
        logging.info(
            "Workload in each queue: {}".format(
                list([len(args_list) for args_list in args_list_queue])
            )
        )

        # MAIN LOOP
        while sum(is_queue_done) < num_queues:
            if not is_queue_done[qid]:
                tid = task_in_queue[qid]
                args = args_list_queue[qid][tid]

                # needs to setup the process
                if tid not in task_to_proc[qid]:
                    task_to_proc[qid][tid] = self.create_prune_process(args)
                else:
                    proc = task_to_proc[qid][tid]  # current proc

                    if proc.poll() is not None:  # execution finished
                        logging.debug(
                            "==> Processed process PID={}, summarizing ...".format(
                                proc.pid
                            )
                        )
                        stdout, stderr = proc.communicate()

                        # update the corresponding entry in the global summary
                        key = os.path.basename(args.checkpoint)
                        summary[key] = self.load_summary_from_checkpoint(args)

                        # move to the next process
                        tid += 1
                        task_in_queue[qid] = tid
                        if tid < len(args_list_queue[qid]):
                            # set the new args
                            task_to_proc[qid][tid] = self.create_prune_process(
                                args_list_queue[qid][tid]
                            )
                        else:
                            logging.info(
                                "Queue for GPU={} has bee processed.".format(
                                    self.gpu_ids[qid]
                                )
                            )
                            is_queue_done[qid] = True

            # update the queue ID
            qid = (qid + 1) % num_queues
            time.sleep(self.sleep_for_secs)

        logging.info("Finished all workloads in all queues.")
        # write the final summary to the search directory
        with open(self.args.dir + "/summary.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        logging.info(
            "Summary has been written to {}".format(self.args.dir + "/summary.json")
        )


def main():
    """ iterate every possible hyperparameter combinations """
    searcher = HyperParamSearcher(args)
    searcher.run()


if __name__ == "__main__":
    main()
