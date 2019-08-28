""" A script to generate all CIFAR-10/100 experimental data.

Related to Table 1(a) of the paper.

This serves differently with the cifar.py file under early_stage.

Example usage::

  python cifar.py -a resnet110 -g 2 -d cifar10 [other options]

This command generates one entry of the table.
"""

import argparse
import os
import sys
import shutil
import yaml
from datetime import datetime
from subprocess import Popen, PIPE, STDOUT
import logging
logging.getLogger().setLevel(logging.DEBUG)

# reuse model runner parameters
from gumi.model_runner.parser import create_cli_parser


def parse_args():
    parser = argparse.ArgumentParser(prog='CLI for running CIFAR evaluation.')
    parser.add_argument('-f',
                        '--config',
                        type=str,
                        metavar='PATH',
                        help='Configuration file for running')
    parser.add_argument('--skip-baseline',
                        action='store_true',
                        default=False,
                        help='Whether to skip the baseline command')
    parser.add_argument('--gpu-id',
                        type=str,
                        default='',
                        help='Override the gpu-id provided in configure files.')

    return parser.parse_args()


class ConfigEncoder(object):
    """ Encoded the configuration for evaluation """

    @staticmethod
    def encode(cmd, args_dict):
        """ <arch>/<dataset>/<cmd>/[<group-config>/]<learning-config> """
        strs = []
        strs.append(args_dict['--arch'])
        strs.append(args_dict['--dataset'])
        strs.append(cmd)

        if cmd in ['prune', 'scratch']:
            # group information
            strs.append(ConfigEncoder.encode_group_config(args_dict))

            # permutation information
            perm = ConfigEncoder.encode_perm(args_dict)
            if perm is not None:
                strs.append(perm)

        if cmd in ['gopt']:
            strs.append(ConfigEncoder.encode_gopt(args_dict))

        # don't need train hyperparameters
        if cmd not in ['gopt']:
            strs.append(ConfigEncoder.encode_train_hyper_params(args_dict))

        return os.path.join(*strs)

    @staticmethod
    def encode_train_hyper_params(args_dict):
        """ Encode specified training hyper-parameters. """
        strs = []
        if '--lr' in args_dict:
            strs.append('LR_{}'.format(args_dict['--lr']))
        if '--lr-type' in args_dict:
            strs.append('{}'.format(args_dict['--lr-type']))
        if '--epochs' in args_dict:
            strs.append('N_{}'.format(args_dict['--epochs']))
        if '--schedule' in args_dict:
            strs.append('S_{}'.format(args_dict['--schedule'].replace(' ',
                                                                      '-')))
        if '--wd' in args_dict:
            strs.append('WD_{}'.format(args_dict['--wd']))
        if '--train-batch' in args_dict:
            strs.append('B{}'.format(args_dict['--train-batch']))

        return '_'.join(strs)

    @staticmethod
    def encode_gopt(args_dict):
        """ GOpt configuration """
        strs = []

        # some cfgs
        max_num_params = args_dict.get('--max-num-params', None)
        max_num_ops = args_dict.get('--max-num-ops', None)
        min_factor = args_dict.get('--min-factor', None)
        strategy = args_dict.get('--strategy', None)

        if max_num_params:
            strs.append('MP_{:.2f}M'.format(max_num_params))
        if max_num_ops:
            strs.append('MO_{:.2f}M'.format(max_num_ops))
        if min_factor:
            strs.append('MIN_{}'.format(min_factor))
        if strategy:
            strs.append('ST_{}'.format(strategy))

        return '_'.join(strs)

    @staticmethod
    def encode_perm(args_dict):
        """ Encode permutation. """
        strs = []

        perm_val = args_dict.get('--perm', None)
        if perm_val:
            strs.append(perm_val)

            if perm_val == 'GRPS':
                ns = args_dict.get('--num-sort-iters', None)
                if ns:
                    strs.append('NS_{}'.format(ns))

        no_weights = args_dict.get('--no-weight', False)
        if no_weights:
            strs.append('WoW')  # w/o weights

        ind_type = args_dict.get('--ind', None)
        if ind_type:
            strs.append(ind_type)

        return '_'.join(strs)

    @staticmethod
    def encode_group_config(args_dict):
        """ """
        if '--group-cfg' in args_dict:
            assert os.path.isfile(args_dict['--group-cfg'])
            base_path = os.path.basename(args_dict['--group-cfg'])
            # TODO - this rule may not always be applicable
            return 'CFG_{}'.format(
                os.path.splitext(base_path)[0].replace('_', '-'))

        if '--mcpg' in args_dict:
            return 'MCPG_{}'.format(args_dict['--mcpg'])

        if '-g' in args_dict:
            return 'G_{}'.format(args_dict['-g'])

        raise ValueError('No valid group config can be found.')


class ArgsEncoder(object):
    """ Encode a Namespace back to an arguments list. """

    @staticmethod
    def encode(args, except_keys=None):
        """ Encode. """
        argv = []

        for k, v in args.items():
            if except_keys and k in except_keys:
                continue  # keys to be filtered out

            # special handler
            if k == '--schedule':
                argv.extend([k] + v.split())
            elif k in ['--dataset-dir', '--resume', '--checkpoint']:
                argv.extend([k, os.path.expandvars(v)])
            elif isinstance(v, bool) and v:
                argv.extend([k])
            else:
                # update the list
                argv.extend([k, str(v)])

        return argv


class CIFAREval(object):
    """ CIFAR evaluation class. """

    def __init__(self, args):
        """ CTOR. """
        self.args = args
        assert isinstance(self.args.config, str)

        with open(self.args.config, 'r') as f:
            self.cfg = yaml.load(f)

        self.base_dir = os.path.expandvars(self.cfg['base_dir'])
        os.makedirs(self.base_dir, exist_ok=True)

        # self.dir = self.get_dir(args)
        # os.makedirs(self.dir, exist_ok=True)
        # logging.debug('==> Initialised directory at {}'.format(self.dir))

        # self.timestamp = self.get_timestamp()  # initialise when constructing
        # self.prune_dir = os.path.join(self.dir, 'prune', self.timestamp)
        # os.makedirs(self.prune_dir, exist_ok=True)
        # self.prune_log = open(os.path.join(self.prune_dir, 'prune.log'), 'w')

    def update_args_dict(self, cmd, args_dict):
        """ Insert some new key-value pairs in args_dict, or do some calculation. """
        base = self.cfg['common'].copy()  # common arguments, can be override
        # HACK - we want to override the common
        base.update(args_dict)
        args_dict.update(base)

        # prepend the base directory
        for path_key in ['--resume', '--checkpoint']:
            if path_key in args_dict:
                args_dict[path_key] = os.path.join(self.base_dir,
                                                   args_dict[path_key])

        # the checkpoint dictionary
        if '--checkpoint' not in args_dict:
            # we only overwrite when there is no checkpoint specified
            checkpoint_path = self.get_dir(cmd, args_dict)
            args_dict['--checkpoint'] = checkpoint_path
        else:
            # update if checkpoint is specified as a file
            checkpoint_path = os.path.expandvars(args_dict['--checkpoint'])
            if os.path.isfile(checkpoint_path):
                checkpoint_path = os.path.dirname(checkpoint_path)
                args_dict['--checkpoint'] = checkpoint_path

        # update GPU id
        if self.args.gpu_id:
            args_dict['--gpu-id'] = self.args.gpu_id

    def run_all(self):
        """ Run all runs specified by self.cfg """
        results = []
        for run_args in self.cfg['runs']:
            res = self.run_cmd(run_args)
            results.append(res)

        return results

    def run_cmd(self, run_args):
        """ Run a single argument """
        # extract the CMD and argument dicts
        cmd, args_dict = next(iter(run_args.items()))

        #HACK
        if cmd == 'baseline' and self.args.skip_baseline:
            logging.info(
                'Baseline command is skipped due to the --skip-baseline flag')
            return None  # skipped

        self.update_args_dict(cmd, args_dict)

        # collect ARGV
        argv = self.create_argv(cmd, args_dict)
        # make some preparation
        checkpoint = args_dict['--checkpoint']
        os.makedirs(checkpoint, exist_ok=True)

        log_path = os.path.join(args_dict['--checkpoint'],
                                'stdout.{}.log'.format(self.get_timestamp()))
        log_file = open(log_path, 'w')
        logging.debug('LOG file: {}'.format(log_path))

        proc = Popen(argv, stdout=log_file, stderr=STDOUT)

        logging.debug('==> Started process PID={} for {}:'.format(
            proc.pid, cmd))
        self.print_argv(argv)

        # waiting to be done
        stdout, stderr = proc.communicate()
        logging.debug('DONE')

        log_file.close()

        return stdout

    def get_timestamp(self):
        """ Get the timestamp of the current run. """
        return datetime.now().isoformat()

    def get_dir(self, cmd, args_dict):
        """ Get the directory based on the working_dir.

        Rule:
        <base_dir>/<config_str>/
        """
        cfg_str = ConfigEncoder.encode(cmd, args_dict)
        return os.path.join(self.base_dir, cfg_str)

    def create_argv(self, cmd, args_dict):
        """ Create the argument list from cmd and args_dict """
        prog = '{}.py'.format(cmd)
        assert os.path.isfile(prog)

        argv = ['python', prog] + ArgsEncoder.encode(args_dict)

        return argv

    def print_argv(self, argv):
        """ Print the ARGV to be executed. """
        # the header
        print(' '.join(argv[:2]) + ' \\')

        # print every pair of parameters
        i = 2
        while i < len(argv):
            # check whether a parameter exists
            j = 1
            while i + j < len(argv) and argv[i + j][0] != '-':
                j += 1

            print('\t{} \\'.format(' '.join(argv[i:i + j])))
            i += j


def main():
    """ Main """
    args = parse_args()
    cifar_eval = CIFAREval(args)
    cifar_eval.run_all()
    # cifar_eval.run_prune(args)


if __name__ == '__main__':
    main()
