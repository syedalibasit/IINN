# Split Description

The experiments use several train/test settings to separate in-domain accuracy from deployment robustness.

## Split Types

| Split | Description | Purpose |
|---|---|---|
| In-domain | Train and test samples are drawn from the same city/frequency setting. | Measures baseline predictive accuracy when deployment conditions are matched. |
| Limited training data | The training set is restricted to a smaller fraction while the test setting remains fixed. | Tests how models behave when labelled samples are limited. |
| Label noise | Training labels are perturbed while the test labels remain clean. | Tests robustness to imperfect measurements or simulation noise. |
| Imbalance | Training data distribution is skewed across relevant regions or classes. | Tests sensitivity to nonuniform sample coverage. |
| City shift | Train and test cities differ, for example Chicago-to-Brussels or Brussels-to-Chicago. | Tests transfer across urban morphology, clutter composition, and site geometry. |
| Frequency shift | Train and test carrier frequencies differ within a related deployment setting. | Tests whether learned structure transfers across carrier frequency. |
| Joint city-frequency shift | Both city and carrier frequency change between training and testing. | Tests a stronger OOD deployment shift. |
| Cross-propagation transfer | Training labels and test labels are generated using different propagation-label models, for example 3GPP-to-SPM. | Tests whether the model over-aligns with one label generator. |

## OOD Degradation

OOD degradation is used to compare how much performance worsens when moving from an in-domain test setting to an OOD setting. The metric is interpreted alongside RMSE, MAE, and MBE because a model may have acceptable average error under matched conditions but degrade sharply when city, frequency, or label-generation assumptions change.

## Cross-Propagation Note

The cross-propagation split is central to the ablation. The original equation-wired IINN shows a large systematic offset under 3GPP-to-SPM transfer. The updated generalized/white-box Physics-IINN keeps interpretable components while reducing this alignment bias.
