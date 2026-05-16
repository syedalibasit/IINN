# Section 03: Network Simulation Visual Diagnostics

This folder contains supplementary figures supporting Section 03 of the paper.

The main manuscript reports the simulation settings and dataset construction. The figures here provide additional visual evidence for the simulation environment, domain shift, and physical consistency of the generated RSRP data.

## Folder Guide

| Folder | What it shows | Why it matters |
|---|---|---|
| `deployment_maps/` | 3D map samples, gNodeB locations, and UE distribution. | Confirms that the datasets come from realistic urban simulation tiles. |
| `clutter_distributions/` | Clutter-class histograms for Brussels and Chicago. | Supports the cross-city OOD shift argument. |
| `rsrp_distance_profiles/` | Sector-wise RSRP trends versus distance. | Checks physical consistency of generated RSRP labels. |
| `terrain_profiles/` | Example terrain and obstruction profiles. | Shows why local propagation deviations occur beyond distance loss alone. |

## Recommended Use

Readers interested in dataset construction should review the folders in this order:

1. `deployment_maps/`
2. `clutter_distributions/`
3. `rsrp_distance_profiles/`
4. `terrain_profiles/`

These figures are supplementary and are not required to reproduce the experiments, but they help visually interpret the simulation environment and domain shifts.
