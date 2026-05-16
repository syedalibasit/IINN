# IINN: Innately Intelligent Neural Networks for Interpretable and Robust Wireless Propagation Modelling

This repository provides the dataset, code, results, and supplementary visual diagnostics supporting the paper **"IINN: Innately Intelligent Neural Networks for Traceable and Robust Wireless Propagation Modelling"**.

The main paper keeps the network-simulation section concise. Additional figures for Section 03 are provided here so readers can visually inspect the simulation environment, urban morphology shift, terrain effects, and physical consistency of the generated RSRP data.

## Repository Guide

| Folder | Description |
|---|---|
| `data/` | Processed wireless propagation datasets and metadata used for model training and evaluation. |
| `figures/section03_network_simulation/` | Supplementary visual diagnostics for Section 03 of the paper. |
| `code/` | Scripts for preprocessing, visualization, model training, and evaluation. |
| `results/` | Performance tables, ablation outputs, and generated result artifacts. |

## How This Repository Supports Section 03

| Paper topic | Repository location | Purpose |
|---|---|---|
| Simulation settings | `data/metadata/` | Documents city, frequency, antenna, deployment, and propagation settings. |
| Dataset variables | `data/metadata/` | Defines features used for RSRP prediction and their physical meaning. |
| 3D deployment maps | `figures/section03_network_simulation/deployment_maps/` | Shows Brussels and Chicago simulation tiles, gNodeB placement, and UE distribution. |
| Urban morphology shift | `figures/section03_network_simulation/clutter_distributions/` | Shows clutter-class differences between Brussels and Chicago. |
| Terrain and obstruction profiles | `figures/section03_network_simulation/terrain_profiles/` | Shows example link profiles illustrating terrain and clutter effects. |
| RSRP-distance consistency | `figures/section03_network_simulation/rsrp_distance_profiles/` | Shows sector-wise RSRP trends versus transmitter--receiver distance. |

## Dataset Summary

The datasets were generated using Atoll by Forsk for 5G NR macrocell deployments in Brussels, Belgium and Chicago, USA. Each city contains five macrocell sites, three sectors per site, and a 1 km² urban simulation tile. The prediction target is reference signal received power (RSRP).

The input variables include network configuration, transmitter--receiver geometry, antenna parameters, carrier frequency, height-related variables, and environmental descriptors. These variables are selected because they correspond to propagation-relevant factors used in analytical radio models and radio-planning workflows.

The datasets support evaluation under:

- city shift,
- frequency shift,
- restricted training data,
- label noise and imbalance,
- cross-propagation transfer.

## Supplementary Visual Diagnostics

The paper retains the simulation settings and dataset description required for a self-contained evaluation. The following supplementary figures are provided here for visual transparency:

1. `deployment_maps/` shows the 3D urban simulation regions and deployment layout.
2. `clutter_distributions/` shows morphology differences between Brussels and Chicago.
3. `rsrp_distance_profiles/` checks that RSRP follows physically plausible distance-dependent trends.
4. `terrain_profiles/` illustrates local obstruction and clutter effects along selected links.

These figures are not required to reproduce the experiments, but they help readers interpret the simulation scenario and the sources of domain shift.

## Recommended Reading Path

Readers who want to inspect the dataset visually should start with:

1. `figures/section03_network_simulation/README.md`
2. `figures/section03_network_simulation/deployment_maps/`
3. `figures/section03_network_simulation/clutter_distributions/`
4. `figures/section03_network_simulation/rsrp_distance_profiles/`
5. `figures/section03_network_simulation/terrain_profiles/`

Readers who want to reproduce experiments should start with:

1. `data/README.md`
2. `data/metadata/`
3. `code/README.md`
4. `results/README.md`

## Citation

If you use this dataset, code, or supplementary material, please cite the associated paper. A complete BibTeX entry will be added after publication.
