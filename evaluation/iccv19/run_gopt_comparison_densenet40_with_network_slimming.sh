#!/bin/sh
# Run training to compare model explored from gopt and manually selected

DATASET_DIR=${NAS_HOME}/datasets
RESULT_DIR=${NAS_HOME}/train/gumi/evaluation/iccv19/final
GPU_ID=$1
MIN_FACTOR=$2
MAX_NUM_PARAMS=$3
NUM_CLASSES=$4
MAX_GROUPS=$5
MANUAL_SEED=42

# DenseNet-40 with G=${G}
ARCH=densenet40
DATASET=cifar${NUM_CLASSES}
GROUP_CFG=${RESULT_DIR}/${ARCH}/${DATASET}/prune/network_slimming/gopt/MAX_COST_P${MAX_NUM_PARAMS}_MIN${MIN_FACTOR}_MAX${MAX_GROUPS}.json
RESUME=${RESULT_DIR}/${ARCH}/${DATASET}/baseline/model_best.pth.tar

# Step 1: Run gopt
python gopt.py -a ${ARCH} -d ${DATASET} --dataset-dir ${DATASET_DIR} \
    --resume ${RESUME} \
    --group-cfg ${GROUP_CFG} \
    --max-num-params ${MAX_NUM_PARAMS} \
    --min-factor ${MIN_FACTOR} \
    --max-groups ${MAX_GROUPS} \
    --excludes-for-applying-mask conv1 \
    --strategy MAX_COST \
    --gpu-id ${GPU_ID}
# Step 2: run pruning and training from scratch
python prune.py \
    -a ${ARCH} \
    -d ${DATASET} \
    --dataset-dir ${DATASET_DIR} \
    --group-cfg ${GROUP_CFG} \
    --resume ${RESUME} \
    --checkpoint ${RESULT_DIR}/${ARCH}/${DATASET}/prune/network_slimming/gopt/MAX_COST_P${MAX_NUM_PARAMS}_MIN${MIN_FACTOR}_MAX${MAX_GROUPS} \
    --perm GRPS \
    --num-sort-iters 10 \
    --epochs 100 \
    --schedule 50 75 \
    --lr 0.01 \
    --wd 5e-4 \
    --gpu-id ${GPU_ID} \
    --manual-seed ${MANUAL_SEED}
