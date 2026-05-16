# Train/Test Split Description

This file documents the split types used in the paper.

## Split Types

| Split type | Description | Purpose |
|---|---|---|
| In-domain | Training and test samples come from the same city/frequency domain. | Measures baseline predictive accuracy. |
| City shift | Training and test samples come from different cities. | Evaluates robustness to morphology and clutter shift. |
| Frequency shift | Training and test samples differ in carrier frequency. | Evaluates transfer across operating bands. |
| Joint city--frequency shift | Training and test samples differ in both city and frequency. | Evaluates the most challenging deployment shift. |
| Restricted training data | Training fraction is reduced while the test domain is fixed. | Evaluates behavior under limited labelled samples. |
| Cross-propagation transfer | Training and test labels come from different propagation-label models. | Evaluates model--label alignment bias and propagation-label robustness. |

## Notes

Each split should be stored with enough information to reproduce the corresponding experiment, including source domain, target domain, training fraction, random seed, and sample identifiers where available.
