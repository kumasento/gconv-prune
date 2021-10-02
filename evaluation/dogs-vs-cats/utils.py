import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DogCatDataset(Dataset):
    """ Dataset for the dogs-vs-cats training images. """
    def __init__(self, img_dir, images, transform=None):
        self.img_dir = img_dir
        self.images = list(images)
        self.transform = transform

        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(os.path.join(self.img_dir, self.images[idx]))
        label = int('dog' in os.path.basename(self.images[idx]))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)
