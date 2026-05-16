# Code

This folder contains the implementation files used for the IINN experiments and the cross-propagation ablation.

## Folder Layout

| Path | Description |
|---|---|
| `experiments_full_e1_e17.py` | Full experiment script covering the E1-E17 study sequence used during rebuttal analysis. |
| `original_equation_wired_iinn/` | Original IINN code in which analytical propagation equations are wired directly into the NN computation. |
| `generalized_physics_iinn/` | Updated code for the generalized/white-box Physics-IINN used to correct the 3GPP-to-SPM OOD analysis. |

## Recommended Reading

Readers interested in the ablation should inspect the code in this order:

1. `original_equation_wired_iinn/` to understand the initial equation-wired design.
2. `generalized_physics_iinn/` to see how the updated model keeps physically named components while reducing dependence on one propagation-label generator.
3. `results/cross_propagation/` to compare the original failure mode with the corrected calibrated transfer results.

## Reproducibility Notes

Some scripts contain absolute paths from the original experiment workstation. Before rerunning, replace those paths with the corresponding files under `data/raw/` or `data/processed/`.

The main Python dependencies are listed in `requirements.txt`. TensorFlow version compatibility may depend on the local CUDA/CPU environment.
