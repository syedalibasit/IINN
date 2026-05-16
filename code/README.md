# Code

This folder contains scripts used for data preprocessing, visualization, model training, and evaluation.

## Expected Structure

| Folder | Description |
|---|---|
| `preprocessing/` | Scripts for cleaning, merging, binning, and preparing tabular datasets. |
| `visualization/` | Scripts for generating simulation diagnostics and result figures. |
| `experiments/` | Scripts for training baselines, IINN variants, and ablation models. |

## Reproducibility Notes

Each experiment script should document:

- source and target dataset,
- train/test split,
- training fraction,
- model configuration,
- random seed,
- output directory.

This information is needed to reproduce the in-domain, OOD, restricted-training-data, and cross-propagation experiments reported in the paper.
