# Figures

This folder contains supplementary visual material that supports the paper without overloading the main manuscript.

## Folder Layout

| Path | Contents |
|---|---|
| `section03_network_simulation/` | Simulation-setting diagnostics for deployment maps, clutter distributions, terrain profiles, and RSRP-distance trends. |
| `model_architecture/` | Full IINN internal computation graph and architecture-level visual material. |
| `performance_evaluation/spatial_consistency/` | Ground-truth and predicted RSRP heatmaps for representative in-domain and OOD experiments. |

## Why These Figures Are Here

The main paper remains self-contained through the simulation settings table, dataset description, model equations, and reported metrics. The figures here provide additional visual evidence for readers who want to inspect:

- the physical deployment layout used in Atoll,
- the urban morphology differences between Brussels and Chicago,
- terrain and obstruction effects along selected links,
- distance-dependent RSRP behavior,
- spatial consistency of selected model predictions,
- the full equation-wired IINN graph that is too dense for the main paper.

Each subfolder contains its own README with figure-level descriptions and interpretation notes.
