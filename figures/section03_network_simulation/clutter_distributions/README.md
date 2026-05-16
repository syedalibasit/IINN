# Clutter Distributions

These figures describe the clutter and land-use composition of the Brussels and Chicago simulation tiles.

| File | Description | Interpretation |
|---|---|---|
| `brussels_clutter_distribution.png` | Clutter-class distribution for the Brussels tile. | Highlights the higher presence of dense urban and high-building classes, which are associated with stronger blockage, shadowing, and multipath variability. |
| `chicago_clutter_distribution.png` | Clutter-class distribution for the Chicago tile. | Shows a different morphology mix with a more balanced distribution across open, suburban, and urban classes. |
| `brussels_vs_chicago_clutter_distribution.png` | Direct comparison of Brussels and Chicago clutter percentages. | This is the most important figure in this folder because it provides visual evidence for the cross-city covariate shift used in the OOD experiments. |

## Analysis

Clutter distribution is a practical source of domain shift in radio planning. A model trained on one city can overfit city-specific correlations if it does not learn transferable propagation structure. The Brussels-Chicago mismatch therefore supports the paper's city-shift evaluation: the model must transfer across a different clutter composition while predicting the same physical target, RSRP.

These figures also clarify why errors may not be explained by distance alone. Samples at similar transmitter-receiver distances can experience different RSRP values because the surrounding clutter, obstruction, and building morphology differ.
