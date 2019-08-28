# GConv Prune

This is the codebase for paper [Efficient Structured Pruning and Architecture Searching for Group Convolution](https://arxiv.org/abs/1811.09341v3) that will appear at the [ICCV'19 NEUARCH workshop](https://neuralarchitects.org/).

- [GConv Prune](#gconv-prune)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training baseline models](#training-baseline-models)
    - [Pruning by a fixed group configuration](#pruning-by-a-fixed-group-configuration)
      - [ImageNet Evaluation Results](#imagenet-evaluation-results)
      - [CondenseNet-86 Evaluation Results](#condensenet-86-evaluation-results)
    - [Searching for a group configuration](#searching-for-a-group-configuration)

## Installation

We use Anaconda3 as the package manager. Please run the following script to initialize our required environmental packages.

```shell
conda env create -f environment.yml
```

Then use `pip` to install the core package `gumi`, which implements the algorithm mentioned in our paper.

## Usage

This project is still in its early stage. Scripts are listed under `evaluation/iccv19` mainly.

### Training baseline models

Baseline models serve as starting points for further pruning. They can be treated the same as pre-trained models.

Please move forward to this [script](evaluation/iccv19/train_baselines.sh) to find commands for training baseline CIFAR-10/100 models used in the paper.

ImageNet baseline models are directly download from PyTorch official [releases](https://pytorch.org/docs/stable/torchvision/models.html).

### Pruning by a fixed group configuration

The script for pruning a model by a fixed group configuration is [prune.py](evaluation/iccv19/prune.py).

We basically use this _fixed_ approach to gather our ImageNet training results.

#### ImageNet Evaluation Results

To get results for Table 2., please check out the following commands.

Note that `--perm GRPS` basically sets the optimisation algorithm to use (mentioned in our paper), and `--num-sort-iters` is the `N_S` hyperparameter as mentioned in the paper.

```shell
# Training ResNet-34 A
python prune.py --group-cfg config/resnet34_A.json --perm GRPS --num-sort-iters 10 -a resnet34 -d imagenet --dataset-dir $DATASET_DIR/ILSVRC2012 --gpu-id $GPU_ID --epochs 30 --lr 1e-2 --schedule 10 20 --checkpoint $RESULTS/resnet34_A --pretrained

# Training ResNet-34 B
python prune.py --group-cfg config/resnet34_B.json --perm GRPS --num-sort-iters 10 -a resnet34 -d imagenet --dataset-dir $DATASET_DIR/ILSVRC2012 --gpu-id $GPU_ID --epochs 30 --lr 1e-2 --schedule 10 20 --checkpoint $RESULTS/resnet34_B --pretrained

# Training ResNet-50
python prune.py -g 2 --perm GRPS --num-sort-iters 10 -a resnet50 -d imagenet --dataset-dir $DATASET_DIR/ILSVRC2012 --gpu-id $GPU_ID --epochs 30 --lr 1e-2 --schedule 10 20 --checkpoint $RESULTS/resnet50_G2 --pretrained

# Training ResNet-101
python prune.py -g 2 --perm GRPS --num-sort-iters 10 -a resnet101 -d imagenet --dataset-dir $DATASET_DIR/ILSVRC2012 --gpu-id $GPU_ID --epochs 30 --lr 1e-2 --schedule 10 20 --checkpoint $RESULTS/resnet101_G2 --pretrained
```

Here are links to download these pruned models:

- [ResNet-34 (A)](https://gumi-models.s3.amazonaws.com/resnet34_A.tar.gz)
- [ResNet-34 (B)](https://gumi-models.s3.amazonaws.com/resnet34_B.tar.gz)
- [ResNet-50 (G=2)](https://gumi-models.s3.amazonaws.com/resnet50_G_2.tar.gz)
- [ResNet-101 (G=2)](https://gumi-models.s3.amazonaws.com/resnet101_G_2.tar.gz)

To evaluate these models, you can reuse the `prune.py` script to only validate, instead of pruning, these checkpoint files.

```shell
# ResNet-34 (A)
tar xvf resnet34_A.tar.gz
python prune.py --group-cfg config/resnet34_A.json -a resnet34 -d imagenet --dataset-dir $DATASET_DIR/ILSVRC2012 --resume resnet34_A/model_best.pth.tar --skip-fine-tune --apply-mask --keep-mask --gpu-id $GPU_ID

# ResNet-34 (B)
tar xvf resnet34_B.tar.gz
python prune.py --group-cfg config/resnet34_B.json -a resnet34 -d imagenet --dataset-dir $DATASET_DIR/ILSVRC2012 --resume resnet34_B/model_best.pth.tar --skip-fine-tune --apply-mask --keep-mask --gpu-id $GPU_ID

# ResNet-50 (G=2)
tar xvf resnet50_G_2.tar.gz
python prune.py -g 2 -a resnet50 -d imagenet --dataset-dir $DATASET_DIR/ILSVRC2012 --resume resnet50_G_2/model_best.pth.tar --skip-fine-tune --apply-mask --keep-mask --gpu-id $GPU_ID

# ResNet-101 (G=2)
tar xvf resnet101_G_2.tar.gz
python prune.py -g 2 -a resnet101 -d imagenet --dataset-dir $DATASET_DIR/ILSVRC2012 --resume resnet101_G_2/model_best.pth.tar --skip-fine-tune --apply-mask --keep-mask --gpu-id $GPU_ID
```

Running the script above will first load the model checkpoint and run validation, and then validate again after updating the model for exportation, and finally produce the size of the GConv-based model and the validation accuracy.

#### CondenseNet-86 Evaluation Results

Besides those ResNet variants, we apply our method on CondenseNet-86 as well by taking a fixed group configuration. We have pruned pre-trained CondenseNet-86 into `G=4` (compared with the original CondenseNet paper) and `G=2` (the one labelled by 50% budget).

To produce these models, first you need to download our pre-trained checkpoints ([CIFAR-10](https://gumi-models.s3.amazonaws.com/densenet86_cifar10.pth.tar) and [CIFAR-100](https://gumi-models.s3.amazonaws.com/densenet86_cifar100.pth.tar)), and then run the `prune.py` script with specification on `-g`.

```shell
# Download CIFAR-10 checkpoint
wget https://gumi-models.s3.amazonaws.com/densenet86_cifar10.pth.tar
# G = 2
python prune.py -g 2 --perm GRPS --num-sort-iters 10 -a condensenet86 -d cifar10 --epochs 100 --schedule 50 75 --lr 5e-3 --wd 1e-4 --resume densenet86_cifar10.pth.tar --checkpoint $RESULT_DIR
# G = 4
python prune.py -g 4 --perm GRPS --num-sort-iters 10 -a condensenet86 -d cifar10 --epochs 100 --schedule 50 75 --lr 5e-3 --wd 1e-4 --resume densenet86_cifar10.pth.tar --checkpoint $RESULT_DIR

# Download CIFAR-100 checkpoint
wget https://gumi-models.s3.amazonaws.com/densenet86_cifar100.pth.tar
# G = 2
python prune.py -g 2 --perm GRPS --num-sort-iters 10 -a condensenet86 -d cifar100 --epochs 100 --schedule 50 75 --lr 5e-3 --wd 1e-4 --resume densenet86_cifar100.pth.tar --checkpoint $RESULT_DIR
# G = 4
python prune.py -g 4 --perm GRPS --num-sort-iters 10 -a condensenet86 -d cifar100 --epochs 100 --schedule 50 75 --lr 5e-3 --wd 1e-4 --resume densenet86_cifar100.pth.tar --checkpoint $RESULT_DIR
```

Pruned models can be downloaded from here:

| Model          | G   | Dataset   | Download                                                                            |
| -------------- | --- | --------- | ----------------------------------------------------------------------------------- |
| CondenseNet-86 | 2   | CIFAR-10  | [link](https://gumi-models.s3.amazonaws.com/condensenet86_cifar10_G_2_GRPS.tar.gz)  |
| CondenseNet-86 | 4   | CIFAR-10  | [link](https://gumi-models.s3.amazonaws.com/condensenet86_cifar10_G_4_GRPS.tar.gz)  |
| CondenseNet-86 | 2   | CIFAR-100 | [link](https://gumi-models.s3.amazonaws.com/condensenet86_cifar100_G_2_GRPS.tar.gz) |
| CondenseNet-86 | 4   | CIFAR-100 | [link](https://gumi-models.s3.amazonaws.com/condensenet86_cifar100_G_4_GRPS.tar.gz) |

To evaluate:

```shell
# CIFAR-10 G=2
wget https://gumi-models.s3.amazonaws.com/condensenet86_cifar10_G_2_GRPS.tar.gz
tar xvf condensenet86_cifar10_G_2_GRPS.tar.gz
python prune.py -g 2 -a condensenet86 -d cifar10 --resume condensenet86_cifar10_G_2/model_best.pth.tar --skip-fine-tune --apply-mask --keep-mask --condensenet --gpu-id $GPU_ID
# CIFAR-10 G=4
wget https://gumi-models.s3.amazonaws.com/condensenet86_cifar10_G_4_GRPS.tar.gz
tar xvf condensenet86_cifar10_G_4_GRPS.tar.gz
python prune.py -g 4 -a condensenet86 -d cifar10 --resume condensenet86_cifar10_G_4/model_best.pth.tar --skip-fine-tune --apply-mask --keep-mask --condensenet --gpu-id $GPU_ID
# CIFAR-10 G=2
wget https://gumi-models.s3.amazonaws.com/condensenet86_cifar100_G_2_GRPS.tar.gz
tar xvf condensenet86_cifar100_G_2_GRPS.tar.gz
python prune.py -g 2 -a condensenet86 -d cifar100 --resume condensenet86_cifar100_G_2/model_best.pth.tar --skip-fine-tune --apply-mask --keep-mask --condensenet --gpu-id $GPU_ID
# CIFAR-10 G=2
wget https://gumi-models.s3.amazonaws.com/condensenet86_cifar100_G_4_GRPS.tar.gz
tar xvf condensenet86_cifar100_G_4_GRPS.tar.gz
python prune.py -g 4 -a condensenet86 -d cifar100 --resume condensenet86_cifar100_G_4/model_best.pth.tar --skip-fine-tune --apply-mask --keep-mask --condensenet --gpu-id $GPU_ID
```

### Searching for a group configuration

The other major benefit of using our method is its ability to search efficiently for a group configuration under given constraints.

`gopt.py` under `evaluation/iccv19` provides the corresponding functionality.

```shell
python gopt.py \
    -a presnet164 \
    -d cifar100 \
    --resume ${PATH_TO_PRETRAINED_MODEL} \
    --group-cfg group_cfg.json \
    --gpu-id ${GPU_ID} \
    --max-num-params ${MAX_NUM_PARAMS} \
    --strategy MAX_COST \
    --min-factor ${MIN_FACTOR}
```

As shown above, when using this script, you need to specify the model architecture, the dataset, and the path to the resumable model file pre-trained on the given dataset. Meanwhile, workload constraints should be provided through `--max-num-params` measured in MB.

`--strategy` and `--min-factor` are hyperparameters that specify how the optimisation should perform.
Strategy `MAX_COST` implements the _local search_ method mentioned in the paper (Section 3.3).
`--min-factor` indicates at least how many channels we want to preserve in each layer. We need this number because the estimated cost could be too aggressive.

This [script](evaluation/iccv19/run_gopt.sh) records the commands to run.

And this [script](evaluation/iccv19/run_gopt_comparison_preresnet164_with_network_slimming.sh)
compares the performance between our approach and the result from Network Slimming.

```shell
# Run gopt with 1.44M number of parameter constraints for CIFAR-100 on GPU 0.
# Each group should have at least 8 channels.
./run_gopt_comparison_preresnet164_with_network_slimming.sh 0 8 1.44 100
```
