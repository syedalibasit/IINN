# White-Box Physics-IINN Cross-Propagation Artifacts

This folder contains the uploaded outputs for the white-box Physics-IINN 3GPP-to-SPM transfer run.

## Main Files

| File | Description |
|---|---|
| `results.json` | Main metrics and run configuration in JSON format. |
| `metrics.xlsx` | Metric table for IID, uncalibrated SPM OOD, and calibrated SPM OOD evaluation. |
| `spm_ood_predictions.xlsx` | Per-sample OOD prediction table. |
| `training_history.xlsx` | Training history for the white-box Physics-IINN run. |
| `whitebox_physics_iinn_architecture.png` | Architecture diagram for the white-box Physics-IINN implementation. |
| `SPM_OOD_uncalibrated_true_vs_predicted.png` | True-vs-predicted plot before calibration. |
| `SPM_OOD_calibrated_true_vs_predicted.png` | True-vs-predicted plot after calibration. |
| `SPM_OOD_uncalibrated_error_histogram.png` | OOD error distribution before calibration. |
| `SPM_OOD_calibrated_error_histogram.png` | OOD error distribution after calibration. |
| `SPM_OOD_last_member_component_decomposition.png` | Component decomposition for the last ensemble member. |
| `SPM_OOD_last_member_component_outputs.xlsx` | Per-sample component outputs. |
| `SPM_OOD_last_member_component_summary.xlsx` | Summary statistics for the named components. |

## Key Metrics

The uploaded JSON reports:

- IID 3GPP RMSE = 7.317 dB,
- SPM OOD uncalibrated RMSE = 11.214 dB and MBE = -4.957 dB,
- SPM OOD calibrated RMSE = 10.017 dB and MBE = -0.330 dB.

## Analysis

This run is the clearest repository artifact for the revised ablation narrative. The model computes through named terms:

- transmit-power contribution,
- distance-frequency path-loss contribution,
- antenna-alignment contribution,
- site/environment contribution,
- clutter offset,
- bounded residual correction.

The calibrated 3GPP-to-SPM result reduces the original equation-wired failure from 54.21 dB OOD RMSE and +53.08 dB MBE to approximately 10.02 dB OOD RMSE and -0.33 dB MBE. This supports the manuscript's conclusion that the earlier failure was caused by rigid equation-label alignment rather than by the idea of interpretable NN structure itself.
