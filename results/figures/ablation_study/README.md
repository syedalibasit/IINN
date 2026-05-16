# Ablation Study Figures

This folder contains figures supporting the ablation analysis of equation-wired IINN and generalized/white-box Physics-IINN.

| File | Description | Interpretation |
|---|---|---|
| `original_iinn_cross_propagation_failure.pdf` | Cross-propagation failure plot for the original equation-wired IINN. | Visualizes the systematic offset observed when transferring from 3GPP-labelled training data to SPM-labelled test data. |
| `ablation_progression_rmse_mbe.pdf` | RMSE and MBE progression across ablation variants. | Shows how changes to the model structure and calibration reduce both error magnitude and mean bias. |
| `generalized_iinn_learning_curve_after_epoch1.pdf` | Learning curve for the generalized IINN after the first epoch. | Shows training behavior after the initial transient and supports convergence analysis. |

## Key Reading

The most important figure is `ablation_progression_rmse_mbe.pdf`, because it connects the architectural revision to the reduction in systematic bias.
