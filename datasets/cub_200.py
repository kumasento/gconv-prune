""" Process CUB-200-2011.

http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
"""

import argparse
import os
from shutil import copyfile
from collections import OrderedDict

NUM_IMAGES = 11788
NUM_TRAIN = 5994


def parse_args():
  """ Parse CLI arguments. """
  parser = argparse.ArgumentParser(prog='CLI for CUB-200-2011')
  parser.add_argument(
      '-d',
      '--dataset-dir',
      type=str,
      metavar='PATH',
      help='Where the raw dataset is located.')

  return parser.parse_args()


def get_image_id_to_path(dataset_dir):
  """ Get the id_to_path dictionary.
  
    NOTE: IDs are strings
  """
  f_path = os.path.join(dataset_dir, 'images.txt')
  images_dir = os.path.join(dataset_dir, 'images')

  assert os.path.isfile(f_path)
  assert os.path.isdir(images_dir)

  with open(f_path, 'r') as f:
    # create the dict
    dict_ = OrderedDict(
        [tuple(line.strip().split(' ')) for line in f.readlines()])
    assert len(dict_) == NUM_IMAGES

    # update the path
    for id_, path in dict_.items():
      dict_[id_] = os.path.join(images_dir, path)
      assert os.path.isfile(dict_[id_])

    return dict_


def get_train_val_split(dataset_dir):
  """ Create two lists of image IDs, one for train,
    one for validation. """

  fp = os.path.join(dataset_dir, 'train_test_split.txt')
  assert os.path.isfile(fp)

  with open(fp, 'r') as f:
    # Image ID -> '0' or '1'
    dict_ = OrderedDict(
        [tuple(line.strip().split(' ')) for line in f.readlines()])
    assert len(dict_) == NUM_IMAGES

    # update is_training
    for k, v in dict_.items():
      dict_[k] = v == '1'

    num_train = len([v for k, v in dict_.items() if v])
    num_val = len([v for k, v in dict_.items() if not v])

    assert num_train == NUM_TRAIN
    assert num_val == NUM_IMAGES - NUM_TRAIN

    return dict_


def create(dataset_dir):
  """ Create the dataset by copying images to
    train and val directories. """
  # Map image ID to path
  id_to_path = get_image_id_to_path(dataset_dir)
  # Map image ID to is_training
  train_val_split = get_train_val_split(dataset_dir)

  # Create train/val directories
  train_dir = os.path.join(dataset_dir, 'train')
  val_dir = os.path.join(dataset_dir, 'val')
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(val_dir, exist_ok=True)

  for img_id, path in id_to_path.items():
    is_train = train_val_split[img_id]
    dst_dir = train_dir if is_train else val_dir
    dst_path = os.path.join(dst_dir, *path.split('/')[-2:])

    # create the parent directory if necessary
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # copy
    copyfile(path, dst_path)


def main():
  """ Main. """
  args = parse_args()

  create(args.dataset_dir)


if __name__ == '__main__':
  main()