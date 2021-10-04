import logging
import os
import random
from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DogCatDataset(Dataset):
    """ Dataset for the dogs-vs-cats training images. """

    def __init__(self, img_dir, images, transform=None):
        self.img_dir = img_dir
        self.images = list(images)
        self.transform = transform
        self.num_classes = 2

        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(os.path.join(self.img_dir, self.images[idx]))
        label = int("dog" in os.path.basename(self.images[idx]))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)


def get_num_images_per_class(images: Iterable[str]) -> Dict[str, int]:
    count = defaultdict(lambda: 0)
    for img in images:
        count[img.split(".")[0]] += 1
    return dict(count)


def get_datasets(dataset_dir: str, val_split: float = 0.1) -> Dict[str, Dataset]:
    # List all the images and shuffle them.
    images = os.listdir(os.path.join(dataset_dir, "train"))
    random.shuffle(images)

    N = len(images)
    n_test = int(N * val_split)
    test_images = images[:n_test]
    train_images = images[n_test:]

    logging.info(f"Classes in train: {get_num_images_per_class(train_images)}")
    logging.info(f"Classes in validation: {get_num_images_per_class(test_images)}")

    assert set(test_images).isdisjoint(train_images)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = DogCatDataset(
        os.path.join(dataset_dir, "train"),
        train_images,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float32),
                normalize,
            ]
        ),
    )
    validation_dataset = DogCatDataset(
        os.path.join(dataset_dir, "train"),
        test_images,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ConvertImageDtype(torch.float32),
                normalize,
            ]
        ),
    )
    test_dataset = DogCatDataset(
        os.path.join(dataset_dir, "test1"),
        os.listdir(os.path.join(dataset_dir, "test1")),
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ConvertImageDtype(torch.float32),
                normalize,
            ]
        ),
    )

    return {
        "train": train_dataset,
        "val": validation_dataset,
        "test": test_dataset,
    }
