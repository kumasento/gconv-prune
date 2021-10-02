#!/usr/bin/env bash

PREFIX=$1
DATASET_DIR="$PREFIX/dogs-vs-cats"

# Create the download target.
mkdir -p "$DATASET_DIR"

# Download the dataset
kaggle competitions download -c dogs-vs-cats -p "$DATASET_DIR"

# Unzip
cd "$DATASET_DIR" || exit
unzip -q dogs-vs-cats.zip
unzip -q train.zip
unzip -q test1.zip

