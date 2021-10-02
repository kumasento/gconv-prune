#!/usr/bin/env python

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from utils import DogCatDataset, get_device

fmt = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s: %(message)s"
logging.basicConfig(format=fmt, level=logging.DEBUG)


def get_dataloader(dataset_dir: str) -> DataLoader:
    path = os.path.join(dataset_dir, 'test1')
    images = os.listdir(path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = DogCatDataset(
        path, images,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float32),
            normalize,
        ]))

    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset-dir',
                        type=str,
                        help='Dataset directory.')
    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        help='Path to checkpoint.')
    args = parser.parse_args()

    device = get_device()

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    model.to(device)

    dataloader = get_dataloader(args.dataset_dir)

    results = []

    for inputs, _ in tqdm(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        results.append(preds)

    labels = torch.cat(results).reshape(-1).cpu().numpy()
    ids = np.arange(1, labels.shape[0] + 1)

    data = {
        'id': ids,
        'label': labels,
    }

    df = pd.DataFrame(data)
    print(df)

    df.to_csv('result.csv', index=False)


if __name__ == '__main__':
    main()
