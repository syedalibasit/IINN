# Cross-Propagation Results

This folder contains detailed outputs for the 3GPP-to-SPM transfer study.

## Purpose

The cross-propagation ablation tests whether an interpretable NN structure remains reliable when the label-generating propagation model changes. This is distinct from city or frequency shift: the input domain may remain related, but the mapping from physical variables to labels changes because the propagation-label model changes.

## Result Summary

| Model variant | OOD RMSE | OOD MAE | OOD MBE | Interpretation |
|---|---:|---:|---:|---|
| Original equation-wired IINN | 54.21 dB | Not listed here | +53.08 dB | Failure dominated by systematic positive offset, indicating equation-label alignment bias. |
| Generalized IINN, uncalibrated | 10.762 dB | 8.524 dB | -4.085 dB | Large error reduction after replacing rigid equation wiring with physically grouped trainable components. |
| Generalized IINN, calibrated | 9.957 dB | 7.545 dB | -0.314 dB | Calibration removes most remaining mean offset while preserving interpretable components. |
| White-box Physics-IINN, uncalibrated | 11.214 dB | 8.985 dB | -4.957 dB | Component-level white-box design without calibration still avoids the original failure mode. |
| White-box Physics-IINN, calibrated | 10.017 dB | 7.565 dB | -0.330 dB | Manuscript-reported rounded result: approximately 10.02 dB RMSE and -0.33 dB MBE. |

## Interpretation

The original equation-wired IINN is structurally interpretable, but the ablation shows that interpretability alone is not enough. If a model is wired too tightly to one propagation-label generator, it can transfer poorly to another label regime.

The updated generalized and white-box Physics-IINN variants retain component-level traceability while reducing dependence on simulator-specific constants. The calibrated results show that most of the original cross-propagation error was systematic bias rather than unavoidable random error.
