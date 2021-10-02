# Dogs vs cats

This is an example usage of the `gconv-prune` API on the model trained for [this Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats/data).

## Prepare the dataset

```sh
./prepare-dataset.sh <DATASET_DIR>
```

## Train

```sh
# From the current directory
python3 train.py -d $DATASET_DIR -n $NUM_EPOCHS -c $CHECKPOINT_DIR
```

By default, we use ResNet-50 as the backbone.

## Test

```sh
python3 test.py -d $DATASET_DIR -c $CHECKPOINT_DIR/resnet50.ckpt
```

## Prune
