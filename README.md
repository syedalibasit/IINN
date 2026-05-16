# IINN: Innately Intelligent Neural Networks for Traceable and Robust Wireless Propagation Modelling

This repository contains the data, code, figures, and result artifacts supporting the paper **"IINN: Innately Intelligent Neural Networks for Traceable and Robust Wireless Propagation Modelling"**.

The manuscript keeps the network-simulation section concise and self-contained. This repository provides the additional visual diagnostics, raw Atoll exports, implementation files, and ablation outputs needed to inspect the simulation setting and reproduce the reported evidence.

## Repository Structure

| Path | Contents |
|---|---|
| `data/processed/` | Cleaned tabular RSRP dataset used for model training and evaluation. |
| `data/raw/atoll_generated/` | Raw Atoll-generated spreadsheets for Brussels, Chicago, frequency-shift, and propagation-model-transfer settings. |
| `data/raw/transmitter_settings/` | Transmitter configuration spreadsheets for the simulated deployments. |
| `data/metadata/` | Simulation settings, feature dictionary, feature-group description, and split definitions. |
| `code/original_equation_wired_iinn/` | Original equation-wired IINN implementation used to expose the cross-propagation alignment issue. |
| `code/generalized_physics_iinn/` | Updated white-box/generalized Physics-IINN code used for corrected 3GPP-to-SPM transfer analysis. |
| `figures/section03_network_simulation/` | Supplementary figures for the simulation setting: deployment maps, clutter distributions, terrain profiles, and RSRP-distance diagnostics. |
| `figures/model_architecture/` | Full internal IINN computation graph moved from the paper to the repository. |
| `figures/performance_evaluation/` | Spatial ground-truth/prediction comparisons for selected in-domain and OOD experiments. |
| `results/figures/` | Paper-level result figures for performance evaluation and ablation analysis. |
| `results/cross_propagation/` | Detailed metrics, predictions, calibration outputs, and component decompositions for 3GPP-to-SPM transfer. |

## Recommended Reading Path

For readers checking the simulation setup:

1. Start with `data/metadata/simulation_settings.csv`.
2. Read `data/metadata/feature_description.md` and `data/metadata/data_dictionary.csv`.
3. Inspect `figures/section03_network_simulation/README.md`.
4. Use the subfolder guides under `deployment_maps/`, `clutter_distributions/`, `terrain_profiles/`, and `rsrp_distance_profiles/` for figure-by-figure interpretation.

For readers checking the IINN implementation and ablation:

1. Read `code/README.md`.
2. Compare `code/original_equation_wired_iinn/` and `code/generalized_physics_iinn/`.
3. Inspect `figures/model_architecture/` for the full equation-wired computation graph.
4. Review `results/cross_propagation/README.md` for corrected 3GPP-to-SPM evaluation.

## Dataset Summary

The data were generated in Atoll by Forsk for 5G NR macrocell deployments in Brussels, Belgium, and Chicago, USA. Each city uses five macrocell sites, three sectors per site, and a 1 km^2 urban simulation tile. The prediction target is reference signal received power (RSRP).

The processed CSV contains 20,390 labelled RSRP samples and includes network configuration, transmitter-receiver geometry, antenna parameters, carrier frequency, height-related variables, and clutter/environment descriptors. These inputs correspond to propagation-relevant factors used in analytical radio models and practical radio-planning workflows.

The repository supports evaluation under in-domain testing, restricted training data, label noise, imbalance, city shift, frequency shift, joint city-frequency shift, and cross-propagation transfer.

## Result Anchors

The following values are included to help readers connect the repository artifacts to the manuscript:

| Setting | Key result |
|---|---|
| Full-data city shift | IINN reports the lowest OOD degradation of 16.80%, compared with 30.88% for DNN and 37.84-48.68% for tree-based baselines. |
| Joint city-frequency shift | IINN reports RMSE = 11.04 dB, MAE = 8.77 dB, and OOD degradation = 23.20%. |
| Original equation-wired 3GPP-to-SPM transfer | OOD RMSE = 54.21 dB and MBE = +53.08 dB, indicating systematic alignment bias. |
| Updated white-box Physics-IINN 3GPP-to-SPM transfer | Calibrated OOD RMSE = 10.017 dB and MBE = -0.330 dB in the uploaded run artifacts. |

## How to Use This Repository

The raw spreadsheets preserve the Atoll export structure. The processed CSV is intended for direct model training and evaluation. The code files are grouped by purpose so the original equation-wired implementation and the updated cross-propagation implementation can be inspected separately.

Some scripts contain local paths from the original experiment machine. Before rerunning experiments, update dataset paths to the corresponding files under `data/raw/` or `data/processed/`.

## Citation

If you use this dataset, code, or supplementary material, please cite the associated paper. A complete BibTeX entry will be added after publication.
