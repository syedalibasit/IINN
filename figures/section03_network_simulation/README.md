# Section 03: Network Simulation Diagnostics

This folder provides visual diagnostics for the data-generation setting described in Section 03 of the paper. These figures are supplementary: they help readers inspect the simulation environment and the source of deployment shifts, while the main manuscript retains only the essential table and dataset description.

## Subfolders

| Path | Purpose |
|---|---|
| `deployment_maps/` | Shows the 1 km^2 3D urban tiles and simulated gNodeB/UE deployments. |
| `clutter_distributions/` | Compares urban morphology and clutter composition between Brussels and Chicago. |
| `terrain_profiles/` | Shows representative terrain and obstruction profiles along selected links. |
| `rsrp_distance_profiles/` | Checks whether simulated RSRP follows physically plausible distance-dependent behavior. |

## How These Diagnostics Support the Paper

The paper evaluates IINN under city shift, frequency shift, joint city-frequency shift, limited training data, and cross-propagation transfer. The figures here support that evaluation by showing that the datasets are not arbitrary tabular samples: they originate from realistic radio-planning scenes with sectorized macrocell deployments, map-dependent clutter, terrain variations, antenna geometry, and distance-dependent signal decay.

The most important diagnostics for Section 03 are:

1. `clutter_distributions/brussels_vs_chicago_clutter_distribution.png`, which explains why Brussels-to-Chicago transfer is a meaningful morphology shift.
2. `deployment_maps/`, which shows the physical layout and UE distribution behind the tabular data.
3. `rsrp_distance_profiles/`, which checks that the generated labels preserve expected propagation trends.
4. `terrain_profiles/`, which illustrates why local deviations from smooth path-loss curves occur.
