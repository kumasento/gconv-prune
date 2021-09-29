""" Training and testing utilities """
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *


def train(
    trainloader, model, criterion, optimizer, epoch, use_cuda, reg=None, show_bar=True
):
    """ Train a model.

  Args:
    trainloader(DataLoader): loader for training data
    model(nn.Module): model to be trained
    criterion: a loss function
    optimizer: how to do optimization
    epoch: current epoch
    use_cuda: whether to use CUDA
  """
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar("Processing", max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            # place inputs and targets on GPU
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)

        inputs, targets = (
            torch.autograd.Variable(inputs),
            torch.autograd.Variable(targets),
        )

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        reg_loss = None
        if reg:
            reg_loss = reg.regularize()
            reg_losses.update(reg_loss.item(), inputs.size(0))
            loss += reg_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if show_bar:
            bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | RegLoss: {reg_loss:.3f} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
                batch=batch_idx + 1,
                size=len(trainloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                # eta=bar.eta_td,
                reg_loss=0.0 if reg is None else reg_losses.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()

    if show_bar:
        bar.finish()

    if reg:
        return (losses.avg, reg_losses.avg, top1.avg)

    return (losses.avg, top1.avg)


def test(testloader, model, criterion, epoch, use_cuda, show_bar=True):
    """ Test a model.

  Args:
    testloader(DataLoader): loader for test data
    model(nn.Module): model to be tested
    criterion: a loss function
    epoch: current epoch
    use_cuda: whether to use CUDA
  """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar("Processing", max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = (
            torch.autograd.Variable(inputs, volatile=True),
            torch.autograd.Variable(targets),
        )

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if show_bar:
            bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                # eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()

    if show_bar:
        bar.finish()
    return (losses.avg, top1.avg)
