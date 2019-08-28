#!/bin/sh
# Run training to compare model explored from gopt and manually selected

DATASET_DIR=${NAS_HOME}/datasets
RESULT_DIR=${NAS_HOME}/train/gumi/evaluation/iccv19/final
GPU_ID=0
MANUAL_SEED=42

# PreResNet-164 with G=8
ARCH=preresnet164
DATASET=cifar100
GROUP_CFG=${RESULT_DIR}/${ARCH}/${DATASET}/prune/G_8/gopt/MAX_COST_P0.26_MIN4.json
RESUME=${RESULT_DIR}/${ARCH}/${DATASET}/baseline/model_best.pth.tar

# Step 1: Run gopt
python gopt.py -a ${ARCH} -d ${DATASET} --dataset-dir ${DATASET_DIR} \
    --resume ${RESUME} \
    --group-cfg ${GROUP_CFG} \
    --max-num-params 0.26 \
    --min-factor 4 \
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
    --checkpoint ${RESULT_DIR}/${ARCH}/${DATASET}/prune/G_8/gopt/MAX_COST_P0.26_MIN4 \
    --perm GRPS \
    --num-sort-iters 10 \
    --train-from-scratch \
    --epochs 164 \
    --schedule 81 122 \
    --wd 1e-4 \
    --gpu-id ${GPU_ID} \
    --manual-seed ${MANUAL_SEED}
