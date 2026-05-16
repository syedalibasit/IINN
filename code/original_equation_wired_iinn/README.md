# Original Equation-Wired IINN

This folder contains the original IINN implementation supplied for the paper experiments.

| File | Description |
|---|---|
| `iinn_generalised_original.py` | Original generalized IINN model code based on equation-wired propagation components. |
| `iinn_original_training_code.py` | Original training and experiment code for the equation-wired IINN workflow. |

## Purpose

The original implementation maps analytical propagation relationships directly into the NN computation. This provides structural interpretability because intermediate components remain associated with physically meaningful terms such as path loss, antenna alignment, and propagation corrections.

The cross-propagation ablation shows the limitation of this design. When a specific propagation formulation is wired too rigidly, the model can align strongly with the label generator used during training. In the uploaded 3GPP-to-SPM transfer results, the original equation-wired design reports OOD RMSE = 54.21 dB and MBE = +53.08 dB, indicating a systematic offset rather than only random prediction error.

This folder is kept to make the ablation transparent: readers can inspect the original implementation and compare it with the updated code in `code/generalized_physics_iinn/`.
