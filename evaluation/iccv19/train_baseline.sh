#!/bin/sh
# This script summarizes commands for training baseline models

# Revise these environment variables for your own cases
DATASET_DIR=$NAS_HOME/datasets
CHECKPOINT_DIR=$NAS_HOME/train/gumi/evaluation/iccv19/final
GPU_ID=0

# PResNet-164
python baseline.py -a presnet164 -d cifar10 --dataset-dir $DATASET_DIR --epochs 164 --schedule 81 122 --wd 1e-4 --checkpoint ${CHECKPOINT_DIR}/presnet164/cifar10/baseline --gpu-id $GPU_ID
python baseline.py -a presnet164 -d cifar100 --dataset-dir $DATASET_DIR --epochs 164 --schedule 81 122 --wd 1e-4 --checkpoint ${CHECKPOINT_DIR}/presnet164/cifar100/baseline --gpu-id $GPU_ID

# DenseNet-86
python baseline.py -a condensenet86 -d cifar10 --dataset-dir $DATASET_DIR  --epochs 300 --lr-type cosine --train-batch 64 --wd 1e-4 --checkpoint ${CHECKPOINT_DIR}/presnet164/cifar10/baseline --gpu-id $GPU_ID
python baseline.py -a condensenet86 -d cifar100 --dataset-dir $DATASET_DIR  --epochs 300 --lr-type cosine --train-batch 64 --wd 1e-4 --checkpoint ${CHECKPOINT_DIR}/presnet164/cifar100/baseline --gpu-id $GPU_ID
