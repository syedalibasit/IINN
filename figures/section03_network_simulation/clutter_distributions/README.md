# Clutter Distributions

This folder contains clutter-class distribution plots for the Brussels and Chicago datasets.

## Purpose

These figures support the paper's claim that the cross-city setting involves a meaningful morphology shift. Brussels contains a higher concentration of dense urban and high-rise clutter, whereas Chicago has a more balanced distribution across open, suburban, and urban classes.

## Suggested Files

| File | Description |
|---|---|
| `brussels_clutter_distribution.png` | Clutter-class distribution for Brussels. |
| `chicago_clutter_distribution.png` | Clutter-class distribution for Chicago. |
| `brussels_vs_chicago_clutter_distribution.png` | Direct comparison used to interpret the city-shift experiment. |

## Interpretation

The clutter mismatch contributes to cross-city covariate shift. This supports the OOD evaluation setting used in the paper, where models must transfer from one urban morphology to another while predicting the same physical target, RSRP.
