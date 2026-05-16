# Spatial Consistency Figures

This folder contains ground-truth and predicted RSRP heatmaps for selected experiments.

| File | Experiment | Description |
|---|---|---|
| `id_scarcity_chicago3p3_ground_truth_frac025.png` | In-domain restricted training data | Ground-truth RSRP map for Chicago 3.3 GHz when only 25% of the training data is used. |
| `id_scarcity_chicago3p3_iinn_prediction_frac025.png` | In-domain restricted training data | IINN prediction map for the same restricted-data setting. |
| `frequency_ood_chicago3p3_to_3p5_ground_truth.png` | Frequency OOD | Ground-truth RSRP map for transfer from Chicago 3.3 GHz to Chicago 3.5 GHz. |
| `frequency_ood_chicago3p3_to_3p5_iinn_prediction.png` | Frequency OOD | IINN prediction map under the same frequency shift. |
| `city_ood_chicago3p3_to_brussels3p3_ground_truth.png` | City OOD | Ground-truth RSRP map for transfer from Chicago 3.3 GHz to Brussels 3.3 GHz. |
| `city_ood_chicago3p3_to_brussels3p3_iinn_prediction.png` | City OOD | IINN prediction map under cross-city transfer. |

## Analysis

The heatmaps provide spatial context for the numerical metrics. They allow readers to inspect whether the model preserves broad RSRP structure across space, not only aggregate RMSE or MAE. The paired ground-truth and prediction plots are particularly useful for identifying systematic spatial errors, smoothing, or missed local extrema under OOD shifts.

These figures should be interpreted together with the metric tables in the paper and the result artifacts under `results/figures/performance_summary/`.
