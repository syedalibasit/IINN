# Raw Data

This folder keeps the raw Atoll-exported spreadsheets and transmitter-configuration files used to construct the processed RSRP dataset and the cross-propagation ablation.

## `atoll_generated/`

| File | Purpose |
|---|---|
| `3.3Ghz_Bruselles_Data.xlsx` | Brussels 3.3 GHz 3GPP-labelled dataset used as the main training source in several experiments. |
| `3.7Ghz_Bruselles_Data.xlsx` | Brussels 3.7 GHz dataset used for frequency-transfer and joint-shift evaluation. |
| `3.3Ghz_Chicago_Data.xlsx` | Chicago 3.3 GHz dataset used for city-shift evaluation. |
| `3.5Ghz_Chicago_Data.xlsx` | Chicago 3.5 GHz dataset used for frequency-shift and joint city-frequency transfer. |
| `3.3Ghz_Brussels_AlternativePropagation.xlsx` | Brussels 3.3 GHz alternative propagation-label dataset used for 3GPP-to-SPM transfer analysis. |
| `3.3Ghz_Bruselles_Simulations_SPM.xlsx` | SPM simulation export for the Brussels 3.3 GHz cross-propagation setting. |
| `3.3Ghz_Bruselles_Transmitter_SPM.xlsx` | Transmitter export associated with the SPM-labelled Brussels scenario. |
| `Clutter_Distribution_Summary.xlsx` | Clutter summary used to compare Brussels and Chicago morphology. |

## `transmitter_settings/`

| File | Purpose |
|---|---|
| `3.3Ghz_Bruselles_Transmitter.xlsx` | Transmitter configuration for Brussels 3.3 GHz. |
| `3.7Ghz_Bruselles_Transmitter.xlsx` | Transmitter configuration for Brussels 3.7 GHz. |
| `3.3Ghz_Chicago_Transmitter.xlsx` | Transmitter configuration for Chicago 3.3 GHz. |
| `3.5Ghz_Chicago_Transmitter.xlsx` | Transmitter configuration for Chicago 3.5 GHz. |

## Use

The raw files preserve the planning-tool export structure. They are useful for auditing the dataset construction and for regenerating processed features such as distance, angular misalignment, height differences, clutter encodings, and propagation-label transfer splits.
