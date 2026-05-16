# Generalized IINN Cross-Propagation Artifacts

This folder contains the uploaded outputs for the generalized IINN 3GPP-to-SPM transfer run.

## Main Files

| File | Description |
|---|---|
| `generalized_iinn_results.json` | Main metrics and run configuration in JSON format. |
| `generalized_iinn_metrics.xlsx` | Metric table for IID, OOD, and calibrated OOD evaluation. |
| `generalized_iinn_ood_predictions.xlsx` | OOD prediction table for the SPM-labelled test set. |
| `generalized_iinn_ood_feature_weights.xlsx` | OOD feature weighting output used in the generalized transfer analysis. |
| `generalized_iinn_training_history.xlsx` | Training history for the generalized IINN run. |
| `generalized_iinn_training_curve_rmse.png` | RMSE learning curve across training. |
| `generalized_iinn_training_curve_rmse_last_member.png` | Training curve for the last ensemble member. |
| `OOD_SPM_ensemble_uncalibrated_true_vs_predicted.png` | True-vs-predicted plot before calibration. |
| `OOD_SPM_ensemble_calibrated_true_vs_predicted.png` | True-vs-predicted plot after calibration. |
| `OOD_SPM_ensemble_uncalibrated_error_histogram.png` | OOD error distribution before calibration. |
| `OOD_SPM_ensemble_calibrated_error_histogram.png` | OOD error distribution after calibration. |
| `OOD_SPM_last_ensemble_member_component_contribution_bar.png` | Component contribution summary for the last ensemble member. |
| `OOD_SPM_last_ensemble_member_white_box_contributions.xlsx` | Per-sample white-box contribution outputs. |
| `OOD_SPM_last_ensemble_member_white_box_contribution_summary.xlsx` | Aggregated component contribution statistics. |

## Key Metrics

The run uses 3GPP-labelled Brussels data for training and SPM-labelled Brussels data for OOD testing. The uploaded JSON reports:

- IID 3GPP test RMSE = 7.084 dB,
- OOD SPM uncalibrated RMSE = 10.762 dB and MBE = -4.085 dB,
- OOD SPM calibrated RMSE = 9.957 dB and MBE = -0.314 dB.

## Analysis

The uncalibrated result already removes the large positive offset seen in the original equation-wired IINN. The calibrated result further reduces mean bias while retaining the component-level structure needed for traceability.
