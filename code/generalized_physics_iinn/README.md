# Generalized / White-Box Physics-IINN

This folder contains the updated implementation used for the corrected 3GPP-to-SPM cross-propagation analysis.

| File | Description |
|---|---|
| `generalized_iinn_model.py` | Generalized IINN model with physically grouped components and trainable correction terms. |
| `whitebox_physics_iinn_model.py` | White-box Physics-IINN model using named components for transmit power, distance-frequency loss, antenna alignment, site/environment effects, clutter offset, and bounded residual correction. |
| `run_cross_propagation_generalized_iinn.py` | Experiment runner for 3GPP-to-SPM transfer, including feature preparation, training, OOD evaluation, and calibration. |

## What Was Corrected

The updated implementation addresses the cross-propagation OOD degradation issue by separating interpretable structure from rigid simulator-specific equation wiring. The revised code:

- removes fixed LoS/NLoS propagation-model constants from the default learned structure,
- replaces absolute coordinates with relative geometry such as link distance, horizontal displacement, antenna misalignment, height difference, and building clearance,
- uses physically named trainable components rather than a single unconstrained hidden representation,
- reports both uncalibrated and calibrated 3GPP-to-SPM OOD metrics,
- applies a small SPM calibration split for the calibrated transfer result,
- exports predictions, component summaries, training curves, and error diagnostics.

## Key Uploaded Result

The uploaded white-box Physics-IINN run reports calibrated 3GPP-to-SPM transfer performance of RMSE = 10.017 dB and MBE = -0.330 dB. This corresponds to the manuscript's rounded discussion value of approximately 10.02 dB and -0.33 dB.

## Running the Code

Before running, update local dataset paths inside the scripts so they point to the spreadsheets under `data/raw/atoll_generated/` or the processed CSV under `data/processed/`.

Install the main dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Then run the cross-propagation script after path updates:

```bash
python code/generalized_physics_iinn/run_cross_propagation_generalized_iinn.py
```
