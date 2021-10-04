""" Base class for model runners. """

import logging
import os
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from gumi.config import GumiConfig
from gumi.model_runner import utils
from gumi.model_runner.logger import Logger


class ModelRunner:
    """ Base class for various utilities that may need to
    train/evaluate a PyTorch model. """
    def __init__(self, args: GumiConfig):
        """ CTOR.

        Args:
            args(Namespace): CLI arguments
        """
        # self.validate_args(args)
        self.args = args

        self.print_args(args)

        # cache data loaders
        self.num_classes = utils.get_num_classes(args.dataset)

        if not args.no_data_loader:
            self.train_loader = utils.get_data_loader(
                args.dataset,
                args.dataset_dir,
                args.train_batch,
                workers=args.workers,
                is_training=True,
            )
            self.val_loader = utils.get_data_loader(
                args.dataset,
                args.dataset_dir,
                args.test_batch,
                workers=args.workers,
                is_training=False,
            )
        else:
            self.train_loader = None
            self.val_loader = None

        # default
        self.criterion = nn.CrossEntropyLoss()
        self.state = asdict(args)
        self.title = ""
        self.checkpoint = args.checkpoint

        # HACK
        if (isinstance(self.checkpoint, str)
                and not os.path.isfile(self.checkpoint)):
            os.makedirs(self.checkpoint, exist_ok=True)
            self.logger = self.get_logger(args)

    def print_args(self, args):
        """ Print given arguments. """
        s = "Arguments:\n"
        s += "---------------------------------------\n"
        s += "ARCH:        {}\n".format(args.arch)
        s += "Dataset:     {}\n".format(args.dataset)
        s += "Dataset DIR: {}\n".format(args.dataset_dir)
        s += "Checkpoint:  {}\n".format(args.checkpoint)
        s += "Resume:      {}\n".format(args.resume)
        s += "Batch:       {} (train) {} (test)\n".format(
            args.train_batch, args.test_batch)
        s += 'Group:       G={} MCPG={} CFG="{}"\n'.format(
            args.num_groups, args.mcpg, args.group_cfg)
        s += 'Ind Type:    "{}"\n'.format(args.ind)
        s += "\n"

        # print out
        print(s)

    def validate_args(self, args):
        """ Do some validation before actual work starts. """
        assert isinstance(args.dataset, str)
        assert (isinstance(args.dataset_dir, str)
                and os.path.isdir(args.dataset_dir))
        assert (args.num_groups >= 1 or args.mcpg >= 1 or os.path.isfile(
            args.group_cfg)), "You should provide at least one group config"

    def get_logger(self, args):
        """ Create logger. """
        if os.path.isdir(args.checkpoint):
            log_file = os.path.join(args.checkpoint, "log.txt")
        else:
            log_file = os.path.join(os.path.dirname(args.checkpoint),
                                    "log.txt")

        logger = Logger(log_file, title=self.title)
        logger.set_names([
            "Learning Rate",
            "Train Loss",
            "Reg Loss",
            "Valid Loss",
            "Train Acc.",
            "Valid Acc.",
        ])
        return logger

    def load_model(self, **kwargs):
        """ Load model and its coefficients. """
        if not self.args.resume_from_best:
            checkpoint_file_name = "checkpoint.pth.tar"
        else:
            checkpoint_file_name = "model_best.pth.tar"

        return utils.load_model(self.args.arch,
                                self.args.dataset,
                                resume=self.args.resume,
                                pretrained=self.args.pretrained,
                                checkpoint_file_name=checkpoint_file_name,
                                **kwargs)

    def print_optimizer(self, optimizer):
        """ Print optimizer state. """
        param_group = optimizer.state_dict()["param_groups"][0]
        s = "Optimizer state:\n"
        s += "--------------------------------\n"
        s += "LR:           {:f}\n".format(param_group["lr"])
        s += "weight decay: {:f}\n".format(param_group["weight_decay"])
        s += "\n"

        print(s)

    def train(self, model, **kwargs):
        """ Simply train the model with provided arguments. """
        if torch.cuda.is_available():
            model.cuda()  # in case the model is not on CUDA yet.

        # prepare
        best_acc = 0
        epochs = self.args.epochs
        base_lr = self.args.lr
        optimizer = optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        # TODO: move this code somewhere else
        if self.args.resume:
            checkpoint = torch.load(self.args.resume)
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])

        logging.info(
            "==> Started training, total epochs {}, start from {}".format(
                epochs, self.args.start_epoch))
        self.print_optimizer(optimizer)

        for epoch in range(self.args.start_epoch, epochs):
            # TODO(13/02/2019): learning rate adjustment
            # self.adjust_learning_rate(epoch, optimizer)
            logging.info("Epoch: [%5d | %5d] LR: %f" %
                         (epoch + 1, epochs, self.state["lr"]))

            # Run train and validation for one epoch
            train_loss, train_acc = utils.train(
                self.train_loader,
                model,
                self.criterion,
                optimizer,
                epoch,
                print_freq=self.args.print_freq,
                state=self.state,
                schedule=self.args.schedule,
                epochs=self.args.epochs,
                base_lr=self.args.lr,
                gamma=self.args.gamma,
                lr_type=self.args.lr_type,
            )

            val_loss, val_acc = utils.validate(self.val_loader,
                                               model,
                                               self.criterion,
                                               print_freq=self.args.print_freq)

            # Append message to Logger
            self.logger.append([
                self.state["lr"], train_loss, 0.0, val_loss, train_acc, val_acc
            ])

            # Update best accuracy
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)

            checkpoint_state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "acc": val_acc,
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            }
            utils.save_checkpoint(checkpoint_state, is_best,
                                  self.args.checkpoint)

        # Finalising
        self.logger.close()
        logging.info("Best accuracy achieved: {:.3f}%".format(best_acc))

        return best_acc

    def validate(self, model, **kwargs):
        """ Validate the performance of a model. """
        return utils.validate(self.val_loader,
                              model,
                              self.criterion,
                              print_freq=self.args.print_freq,
                              **kwargs)

    def adjust_learning_rate(self, epoch, optimizer, batch=None, batches=None):
        """ Adjust learning rate.

            Args:
            epoch(int): current epoch
        """
        args = self.args

        utils.adjust_learning_rate(
            epoch,
            optimizer,
            state=self.state,
            schedule=args.schedule,
            epochs=args.epochs,
            batch=batch,
            batches=batches,
            base_lr=args.lr,
            gamma=args.gamma,
            lr_type=args.lr_type,
        )
