# ICCV '19 Evaluation

This directory implements all the evaluation can be found in the submission.

Some of them are migrated from the `early_stage` directory, some are new.

A List of evaluation programs:

- `cifar_eval.py`: evaluation on CIFAR-10 and CIFAR-100


## Figures

To reproduce the heuristic algorithm samples:

```shell
python heuristic.py -s 128 -g 16 --num-samples 10000 --draw-rand-stats --num-iters 1 2 10 --resume
```