# Data

This folder contains the processed wireless propagation datasets and metadata used in the IINN experiments.

## Expected Structure

| Path | Description |
|---|---|
| `processed/` | Cleaned tabular datasets used for model training and evaluation. |
| `splits/` | Train/test splits for in-domain, city-shift, frequency-shift, restricted-training-data, and cross-propagation experiments. |
| `metadata/` | Data dictionary, simulation settings, feature descriptions, and label-generation notes. |

## Dataset Description

The datasets were generated using Atoll by Forsk for 5G NR macrocell deployments in Brussels and Chicago. Each deployment uses five macrocell sites, three sectors per site, and a 1 km² urban simulation tile.

The prediction target is RSRP. The main feature groups are:

- network configuration,
- transmitter--receiver geometry,
- antenna parameters,
- carrier frequency,
- height-related variables,
- clutter and environmental descriptors.

## Recommended Metadata Files

The `metadata/` folder should include:

| File | Purpose |
|---|---|
| `simulation_settings.csv` | Radio, geographic, antenna, and propagation settings used for data generation. |
| `data_dictionary.csv` | Column-level description of every feature and target variable. |
| `feature_description.md` | Human-readable explanation of feature groups and physical meaning. |
| `splits_description.md` | Explanation of train/test splits used in each experiment. |

## Notes for Readers

The datasets are intended to support reproducibility of the experiments reported in the paper. Supplementary visual diagnostics for the simulation environment are available in `figures/section03_network_simulation/`.
