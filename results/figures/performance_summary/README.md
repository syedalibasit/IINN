# Performance Summary Figures

This folder contains paper-level performance figures for the main evaluation experiments.

| File | Description | Interpretation |
|---|---|---|
| `experiment1_city_shift_all_metrics.pdf` | Full metric comparison for the city-shift experiment. | Shows model behavior when transferring between urban environments. |
| `experiment1_city_shift_ood_degradation.pdf` | OOD degradation comparison for city shift. | Highlights the stability gap between IINN and conventional data-driven baselines under city transfer. |
| `experiment2_derived_robustness_metrics.pdf` | Derived robustness metrics for restricted training and robustness settings. | Supports analysis beyond raw RMSE by considering degradation and stability. |
| `experiment2_training_fraction_ood_curve.pdf` | OOD behavior as the available training fraction changes. | Shows how model behavior changes when labelled training data are restricted. |
| `experiment3_joint_shift_all_metrics.pdf` | Metric comparison for joint city-frequency shift. | Evaluates transfer when both geography and carrier frequency change. |
| `experiment3_joint_shift_all_metrics_v1.pdf` | Alternate rendering of the joint-shift metric comparison. | Retained for transparency and comparison with earlier manuscript figures. |
| `experiment3_joint_shift_rmse_ood.pdf` | RMSE and OOD degradation bars for joint city-frequency shift. | Highlights the main joint-shift result reported in the manuscript. |
| `overall_winner_matrix.pdf` | Summary matrix of best-performing models across metrics and scenarios. | Provides a compact view of where each model performs best. |

## Key Reading

For the manuscript's headline performance claims, the most important files are `experiment1_city_shift_ood_degradation.pdf` and `experiment3_joint_shift_rmse_ood.pdf`.
