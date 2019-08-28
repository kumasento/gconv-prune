#!/bin/sh
# Run gopt commands 

DATASET_DIR=${NAS_HOME}/datasets
RESUME_DIR=${NAS_HOME}/train/gumi/evaluation/iccv19/final

# PreResNet-164 on CIFAR-100
python gopt.py \
    -a preresnet164 \
    -d cifar100 \
    --dataset-dir ${DATASET_DIR} \
    --resume ${RESUME_DIR}/preresnet164/cifar100/baseline/model_best.pth.tar \
    --group-cfg config/resnet164_C100_MAX_COST_P0.23_MIN2.json \
    --gpu-id 0 \
    --strategy MAX_COST \
    --min-factor 2 \
    --excludes-for-applying-mask conv1