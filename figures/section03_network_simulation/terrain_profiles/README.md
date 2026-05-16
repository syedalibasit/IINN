# Terrain Profiles

These figures show selected terrain and obstruction profiles from the Brussels simulation tile.

| File | Description | Interpretation |
|---|---|---|
| `brussels_site0_site3_terrain_profile_928m.png` | Terrain and clutter profile between Site 0 and Site 3 over a 928 m path. | Shows how elevation, buildings, and clutter can vary along a longer urban path. This explains why local signal behavior can deviate from a smooth distance-only path-loss curve. |
| `brussels_site1_sector2_ue_profile_245m.png` | Example terrain profile from Site 1 sector 2 to a UE over a 245 m path. | A shorter link profile used to inspect local obstruction and near-site propagation conditions. |
| `brussels_site3_sector2_ue_profile_557m.png` | Example terrain profile from Site 3 sector 2 to a UE over a 557 m path. | A medium-length link where terrain and clutter transitions can contribute to RSRP variation. |

## Analysis

The terrain profiles support the physical interpretation of the dataset. RSRP is affected by distance and frequency, but also by site geometry, building clearance, clutter, and local obstruction. These profiles help readers understand why the IINN framework includes physically named components beyond a single distance term.

The profiles are not required to reproduce the experiments. They are provided as supplementary diagnostics for readers who want to visually inspect the local propagation context behind selected samples.
