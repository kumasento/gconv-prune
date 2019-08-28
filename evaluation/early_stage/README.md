---
Author: Ruizhe Zhao
Date: 2019/01/07

---


# Early-Stage Evaluation

- [Run train/eval once](#run-traineval-once)

This sub-project evaluates the effect on classification accuracy from number of groups and indexing.

We only explore ResNet-50 on CIFAR-10/100 in this project.

## Run train/eval once

To test the performance of an architecture on CIFAR-10/100:

```shell
python cifar.py -a resnet110g4r --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint $NAS_HOME/train/gumi/evaluation/early_stage/cifar100/resnet110g4r_3 -d cifar100 --dataset-dir $NAS_HOME/datasets --gpu-id 3
```

Supported architectures:

1. `resnet110` - standard ResNet-110
2. `resnet110g[4|8]` - uniformly grouped ResNet-110
3. `resnet110g[4|8]r` - with randomized permutation