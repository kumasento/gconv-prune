#!/usr/bin/env python

import argparse
import copy
import logging
import os
import random
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

from utils import DogCatDataset, get_device

fmt = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s: %(message)s"
logging.basicConfig(format=fmt, level=logging.DEBUG)


def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_num_images_per_class(images: Iterable[str]) -> Dict[str, int]:
    count = defaultdict(lambda: 0)
    for img in images:
        count[img.split('.')[0]] += 1
    return dict(count)


def get_dataloaders(dataset_dir: str,
                    val_split: float = 0.1) -> Dict[str, DataLoader]:
    # List all the images and shuffle them.
    images = os.listdir(os.path.join(dataset_dir, 'train'))
    random.shuffle(images)

    N = len(images)
    test_images = images[:int(N * val_split)]
    train_images = images[int(N * val_split):]

    logging.info(f'Classes in train: {get_num_images_per_class(train_images)}')
    logging.info(
        f'Classes in validation: {get_num_images_per_class(test_images)}')

    assert set(test_images).isdisjoint(train_images)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data = DogCatDataset(
        os.path.join(dataset_dir, 'train'), train_images,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float32),
            normalize,
        ]))

    test_data = DogCatDataset(
        os.path.join(dataset_dir, 'train'), test_images,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float32),
            normalize,
        ]))

    train_dataloader = DataLoader(train_data,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=8)
    test_dataloader = DataLoader(test_data,
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=8,
                                 pin_memory=True)

    return {'train': train_dataloader, 'val': test_dataloader}


def train_model(model: nn.Module,
                dataloaders: Dict[str, DataLoader],
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                num_epochs: int = 25) -> Tuple[nn.Module, List[float]]:
    """ Train the provided model. """
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = get_device()

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders[phase].dataset)

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model: nn.Module, feature_extracting: bool):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(feature_extracting: bool) -> nn.Module:
    model = models.resnet50(pretrained=True)
    set_parameter_requires_grad(model, feature_extracting)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.to(get_device())

    return model


def main():
    logging.info(f'Torch version: {torch.__version__}')
    logging.info(f'Torchvision version: {torchvision.__version__}')

    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-n',
                        '--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs.')
    parser.add_argument('-d',
                        '--dataset-dir',
                        type=str,
                        help='Dataset directory.')
    parser.add_argument('--feature-extracting',
                        action='store_true',
                        help='Whether use feature extraction.')
    parser.add_argument('-c',
                        '--checkpoint-dir',
                        type=str,
                        help='Checkpoint directory.')
    args = parser.parse_args()

    dataloaders = get_dataloaders(args.dataset_dir)
    model = initialize_model(args.feature_extracting)

    params_to_update = model.parameters()
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Train and evaluate
    model, hist = train_model(model,
                              dataloaders,
                              criterion,
                              optimizer_ft,
                              num_epochs=args.epochs)

    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    ckpt_path = os.path.join(args.checkpoint_dir, 'resnet50.ckpt')
    logging.info(f"Saving trained model to {ckpt_path}")
    torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    main()
