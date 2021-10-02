""" Utility functions for model_runner """

import json
import logging
import math
import os
import random
import shutil
import sys
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as imagenet_models
import torchvision.transforms as transforms
from gumi import model_utils
from gumi import \
    models as cifar_models  # a module contains all supported models
from gumi.models import imagenet as custom_imagenet_models
from gumi.ops import *

#######################################
# Models                              #
#######################################

models = {
    **cifar_models.__dict__,
    **imagenet_models.__dict__,
    **custom_imagenet_models.__dict__,
}
model_names = sorted(
    name for name in models
    if name.islower() and not name.startswith("__") and callable(models[name]))

# datasets can be loaded by imagenet loader
IMAGENET_DATASETS = ["imagenet", "cub200"]


def apply_dense(model):
    """ Turn SparseGroupConv2d to use dense weights """
    for name, mod in model.named_modules():
        if isinstance(mod, SparseGroupConv2d):
            mod.weight.data = mod.weight.to_dense()


def apply_sparse(model):
    """ Turn SparseGroupConv2d to use sparse weights """
    for name, mod in model.named_modules():
        if isinstance(mod, SparseGroupConv2d):
            mod.update_weight(mod.weight)


def apply_mask(model, excludes=None, use_cuda=True):
    """ Updated Conv2d modules in model to MaskConv2d.
  
  NOTE: we need to override a Conv2d module by MaskConv2d from
    its parent's _modules dict.
  
  This module will be updated in-place.
  """
    assert isinstance(model, nn.Module)

    # the name of the module to be excluded.
    if excludes is None:
        excludes = []

    # find the parent module of all Conv2d modules
    # replace them by MaskConv2d
    for name, mod in model.named_modules():
        name_to_mod = {}

        # find all Conv2d children
        for child_name, child in mod.named_children():
            if child_name in excludes:  # skip
                continue

            if isinstance(child, nn.Conv2d) and child.groups == 1:
                # if not, we are losing modules
                assert len(list(child.children())) == 0

                mask_conv = MaskConv2d.create_from_conv2d(child,
                                                          use_cuda=use_cuda)
                mask_conv.G = -1  # de-initialise G
                # update dict
                name_to_mod[child_name] = mask_conv

        # not a direct parent of any Conv2d module
        if not name_to_mod:
            continue

        # update children reference correspondingly
        for child_name, child in name_to_mod.items():
            mod._modules[child_name] = child


def get_model_num_ops(model, dataset):
    """ We map dataset to specific input workload. """
    input_size = None
    if dataset.startswith("cifar"):
        input_size = (1, 3, 32, 32)
    elif dataset in IMAGENET_DATASETS:
        input_size = (1, 3, 224, 224)
    else:
        raise ValueError("Cannot recognise dataset {}".format(dataset))

    return model_utils.get_model_num_ops(model, input_size)


def load_model(arch,
               dataset,
               resume=None,
               pretrained=False,
               update_model_fn=None,
               update_state_dict_fn=None,
               use_cuda=True,
               fine_tune=False,
               data_parallel=True,
               checkpoint_file_name="checkpoint.pth.tar",
               **kwargs):
    """ Load a model.
  
    You can either load a CIFAR model from gumi.models
    or an ImageNet model from torchvision.models

    Additional parameters in kwargs are passed only to cifar models.

    You can use update_model_fn and update_state_dict_fn
    to configure those models undesirable.

    NOTE: fine_tune won't control whether to run replace_classifier.
  """
    # construct the model
    num_classes = get_num_classes(dataset)
    if dataset.startswith("cifar"):
        model = cifar_models.__dict__[arch](num_classes=num_classes, **kwargs)
    elif dataset in IMAGENET_DATASETS:
        # NOTE: when creating this model, all its contents are
        # already initialised. Won't go to the resume branch.
        if arch in imagenet_models.__dict__:
            model = imagenet_models.__dict__[arch](pretrained=pretrained)
        else:
            model = custom_imagenet_models.__dict__[arch](**kwargs)

        replace_classifier(arch, model, dataset, fine_tune=fine_tune)

    logging.debug("Total params: {:.2f}M FLOPS: {:.2f}M".format(
        model_utils.get_model_num_params(model),
        get_model_num_ops(model, dataset)))

    # update model if required
    if update_model_fn:
        logging.debug("update_model_fn is provided.")
        model = update_model_fn(model)

    if resume:  # load from checkpoint
        if pretrained:
            raise ValueError(
                "You cannot specify pretrained to True and resume not None.")

        assert isinstance(resume, str)

        # update the resume if it points to a directory
        if os.path.isdir(resume):
            resume = os.path.join(resume, checkpoint_file_name)
            logging.debug(
                "Resume was given as a directory, updated to: {}".format(
                    resume))

        # now resume should be a valid file.
        assert os.path.isfile(resume)

        checkpoint = torch.load(resume)  # load

        # get the state dict
        state_dict = checkpoint["state_dict"]
        if update_state_dict_fn:
            state_dict = update_state_dict_fn(state_dict)

        # initialize model
        model.load_state_dict(state_dict, strict=not fine_tune)

    if use_cuda:
        if data_parallel:
            model = torch.nn.DataParallel(model)
        model = model.cuda()

    return model


def replace_classifier(arch, model, dataset, fine_tune=False):
    """ Replace the final classifier based on the dataset."""
    # TODO: pass arch?
    if dataset == "imagenet":
        return

    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    num_classes = get_num_classes(dataset)
    if fine_tune:  # in the fine-tune mode, we remove all gradient updates
        set_require_grad(model, False)

    if arch.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "alexnet" or arch.startswith("vgg"):
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif arch.startswith("squeezenet"):
        model.classifier[1] = nn.Conv2d(512,
                                        num_classes,
                                        kernel_size=(1, 1),
                                        stride=(1, 1))
    elif arch.startswith("densenet"):
        model.classifier = nn.Linear(1024, num_classes)
    elif arch.startswith("mobilenet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # HACK
        nn.init.kaiming_normal_(model.fc.weight,
                                mode="fan_out",
                                nonlinearity="relu")
    else:
        raise ValueError("ARCH={} cannot be recognized.".format(arch))


def set_require_grad(model, require_grad):
    for param in model.parameters():
        param.require_grad = require_grad


#######################################
# Dataset                             #
#######################################


def get_cifar_transform(is_training=False, **kwargs):
    """ Get the input data transform object. """
    if is_training:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])


def get_cifar_dataset(dataset,
                      is_training=False,
                      download=False,
                      dataset_dir=None,
                      **kwargs):
    """ create a CIFAR dataset object """
    assert isinstance(dataset_dir, str) and os.path.isdir(dataset_dir)

    # construct the Dataset class
    if dataset == "cifar10":
        CIFARDataset = datasets.CIFAR10
    elif dataset == "cifar100":
        CIFARDataset = datasets.CIFAR100
    else:
        raise ValueError(
            'dataset should be either "cifar10" or "cifar100", got: {}'.format(
                dataset))

    return CIFARDataset(
        root=dataset_dir,
        train=is_training,
        download=download,
        transform=get_cifar_transform(is_training=is_training, **kwargs),
    )


def get_imagenet_dataset(is_training=False, dataset_dir=None, **kwargs):
    """ Create ImageNet dataset object. """
    assert isinstance(dataset_dir, str) and os.path.isdir(dataset_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # convention
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    logging.debug("Creating ImageNet dataset loader for {} ...".format(
        "training" if is_training else "validation"))

    if is_training:
        return datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        )
    else:
        return datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
        )


def get_num_classes(dataset):
    """ NOTE: Need to update this when adding a new dataset. """
    if dataset == "cifar10":
        return 10
    if dataset == "cifar100":
        return 100
    if dataset == "imagenet":
        return 1000
    if dataset == "cub200":
        return 200

    raise ValueError("dataset cannot be recognised: {}".format(dataset))


def get_dataset(dataset, **kwargs):
    """ Get the dataset object based on args.
  
    NOTE: if need more datasets, just add more branches.
  """

    # Get the dataset class
    if dataset.startswith("cifar"):
        return get_cifar_dataset(dataset, **kwargs)
    elif dataset in ["imagenet", "cub200"]:
        return get_imagenet_dataset(**kwargs)
    else:
        raise ValueError(
            'dataset should be one of "cifar10", "cifar100", "imagenet", "cub200", got: {}'
            .format(dataset))


def get_data_loader(dataset,
                    dataset_dir,
                    batch_size,
                    workers=8,
                    is_training=False):
    """ Create data loader. """
    return data.DataLoader(
        get_dataset(dataset, is_training=is_training, dataset_dir=dataset_dir),
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=workers,
        pin_memory=True,
    )


#######################################
# Logging                             #
#######################################


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


#######################################
#  Training and Eval                  #
#######################################


def train(train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          print_freq=100,
          gpu=None,
          max_iters=None,
          no_update=False,
          **kwargs):
    """ Major training function. """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        lr = adjust_learning_rate(epoch,
                                  optimizer,
                                  batch=i,
                                  batches=len(train_loader),
                                  **kwargs)

        if gpu is not None:
            input = input.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if max_iters is not None:  # in this case we don't do update
            if i >= max_iters:
                break
        if not no_update:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print("Epoch: [{0}][{1}/{2}]\t"
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "LR: {lr:.4f}\t"
                  "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                  "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                  "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                      epoch,
                      i,
                      len(train_loader),
                      lr=lr,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                  ))

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, print_freq=100, gpu=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # if gpu is not None:
            input = input.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print("Test: [{0}/{1}]\t"
                      "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                      "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                      "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1,
                          top5=top5,
                      ))

        print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1,
                                                                    top5=top5))

    return losses.avg, top1.avg


def save_checkpoint(state,
                    is_best,
                    checkpoint_dir,
                    file_name="checkpoint.pth.tar"):
    path = os.path.join(checkpoint_dir, file_name)
    torch.save(state, path)

    if is_best:
        shutil.copyfile(path, os.path.join(checkpoint_dir,
                                           "model_best.pth.tar"))


def adjust_learning_rate(
        epoch,
        optimizer,
        state=None,
        schedule=None,
        epochs=None,
        batch=None,
        batches=None,
        base_lr=None,
        gamma=None,
        lr_type=None,
):
    """ Adjust the LR value in state. """
    assert state is not None

    if lr_type == "cosine":
        assert isinstance(epochs, int)
        assert isinstance(batches, int)
        assert isinstance(batch, int)
        assert isinstance(base_lr, float)

        tot = epochs * batches
        cur = (epoch % epochs) * batches + batch  # TODO why mod?
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * cur / tot))

    elif lr_type is None or lr_type == "multistep":
        assert state is not None
        assert schedule is not None
        assert isinstance(gamma, float)
        assert isinstance(batch, int)

        lr = state["lr"]
        if epoch in schedule and batch == 0:
            lr *= gamma

    else:
        raise ValueError("lr_type {} cannot be recognised".format(lr_type))

    state["lr"] = lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def create_get_num_groups_fn(G=0,
                             MCPG=0,
                             group_cfg=None,
                             use_cuda=True,
                             data_parallel=True):
    """ Create the hook function for getting 
    the number of groups for a given module. """

    g_cfg = None
    if isinstance(group_cfg, str) and os.path.isfile(group_cfg):
        with open(group_cfg, "r") as f:
            g_cfg = json.load(f)

    def get_num_groups(name, mod):
        G_ = G  # choose G in the beginning

        W = model_utils.get_weight_parameter(mod)
        F, C = W.shape[:2]
        if not data_parallel:
            name = "module." + name

        # how to override G_
        if g_cfg is not None:
            if name in g_cfg:
                G_ = g_cfg[name]["G"]
                # do some verification
                if G_ != 1:  # HACK
                    assert F == g_cfg[name][
                        "F"], "F={} does not match cfg={}".format(
                            F, g_cfg[name]["F"])
                    assert C == g_cfg[name][
                        "C"], "C={} does not match cfg={}".format(
                            C, g_cfg[name]["C"])
            else:
                G_ = 1  # HACK - we don't want to have G=0 in further processing

        elif MCPG > 0:
            if GroupConv2d.groupable(C, F, max_channels_per_group=MCPG):
                G_ = GroupConv2d.get_num_groups(C,
                                                F,
                                                max_channels_per_group=MCPG)
            else:
                logging.warn(
                    "Module {} is not groupable under MCPG={}, set its G to 1".
                    format(name, MCPG))
                G_ = 1

        return G_

    return get_num_groups
