# Data

This folder contains the processed dataset, raw Atoll exports, transmitter settings, and metadata used in the IINN experiments.

## Folder Layout

| Path | Description |
|---|---|
| `processed/iinn_rsrp_dataset.csv` | Cleaned supervised-learning dataset used for RSRP prediction. |
| `raw/atoll_generated/` | Raw Atoll spreadsheets for Brussels, Chicago, frequency-shift, and 3GPP-to-SPM transfer cases. |
| `raw/transmitter_settings/` | Transmitter configuration files for the simulated gNodeB deployments. |
| `metadata/data_dictionary.csv` | Column-level description of the processed dataset. |
| `metadata/simulation_settings.csv` | Radio, geographic, antenna, and propagation settings used for data generation. |
| `metadata/feature_description.md` | Human-readable description of the physical feature groups. |
| `metadata/splits_description.md` | Definition of in-domain, OOD, restricted-data, and cross-propagation splits. |

## Processed Dataset

`processed/iinn_rsrp_dataset.csv` contains 20,390 labelled samples. The target variable is RSRP. The main feature groups are:

- network configuration,
- transmitter-receiver geometry,
- antenna orientation and beamwidth parameters,
- carrier frequency,
- transmitter and user-equipment height variables,
- clutter and environmental descriptors.

The processed file is suitable for model development because it aligns the radio, geometry, antenna, environment, and RSRP fields in one table.

## Raw Data

The raw spreadsheets are retained for transparency and reproducibility. They include Brussels and Chicago Atoll exports at different carrier frequencies, the alternative propagation-label dataset used for SPM transfer, clutter-distribution summaries, and transmitter-configuration files.

Readers who only want to reproduce the reported model comparisons can start from `processed/iinn_rsrp_dataset.csv`. Readers who want to audit how the processed dataset relates to the original simulation exports should inspect `raw/atoll_generated/` and `raw/transmitter_settings/`.

## Notes

The paper keeps only the essential simulation settings in the main text. Additional visual checks of the data-generation environment are provided under `figures/section03_network_simulation/`.
