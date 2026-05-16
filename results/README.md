# Results

This folder contains result figures, metric files, prediction tables, and ablation artifacts supporting the performance evaluation and discussion sections of the paper.

## Folder Layout

| Path | Description |
|---|---|
| `figures/performance_summary/` | PDF figures summarizing the main performance-evaluation experiments. |
| `figures/ablation_study/` | PDF figures supporting the cross-propagation ablation and generalized IINN learning behavior. |
| `cross_propagation/generalized_iinn/` | Metrics, predictions, feature weights, and diagnostic plots from the generalized IINN cross-propagation run. |
| `cross_propagation/whitebox_physics_iinn/` | Metrics, predictions, component outputs, and diagnostic plots from the white-box Physics-IINN run. |

## Result Anchors

| Experiment | Key value |
|---|---|
| Full-data city shift | IINN OOD degradation = 16.80%. |
| Joint city-frequency shift | IINN RMSE = 11.04 dB, MAE = 8.77 dB, OOD degradation = 23.20%. |
| Original equation-wired 3GPP-to-SPM transfer | OOD RMSE = 54.21 dB, MBE = +53.08 dB. |
| Updated white-box Physics-IINN 3GPP-to-SPM transfer | Calibrated OOD RMSE = 10.017 dB, MBE = -0.330 dB. |

## Interpretation

The result files support the paper's main narrative. IINN is evaluated not only by in-domain accuracy, but also by OOD degradation, cross-city transfer, frequency transfer, limited training data, and propagation-label transfer. The cross-propagation results are especially important because they show that interpretability should not require rigidly embedding one propagation standard. The updated white-box Physics-IINN preserves named components while substantially reducing the systematic bias observed in the original equation-wired version.
