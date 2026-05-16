# RSRP-Distance Profiles

These figures summarize RSRP variation with transmitter-receiver distance for Brussels sites and sectors.

| File | Description | Interpretation |
|---|---|---|
| `brussels_site0_rsrp_distance.png` | Sector-wise RSRP-distance trend for Brussels Site 0. | Median RSRP generally decreases with distance, while spread reflects local clutter, antenna orientation, and obstruction effects. |
| `brussels_site1_rsrp_distance.png` | Sector-wise RSRP-distance trend for Brussels Site 1. | Shows site-specific coverage behavior and sector-dependent variation. |
| `brussels_site2_rsrp_distance.png` | Sector-wise RSRP-distance trend for Brussels Site 2. | Helps verify that the simulated labels retain large-scale attenuation structure. |
| `brussels_site3_rsrp_distance.png` | Sector-wise RSRP-distance trend for Brussels Site 3. | Shows how sector alignment and local morphology affect RSRP spread around the median trend. |
| `brussels_site4_rsrp_distance.png` | Sector-wise RSRP-distance trend for Brussels Site 4. | Provides another site-level check of distance-dependent received-power behavior. |
| `brussels_all_sites_rsrp_distance.png` | Combined multi-site RSRP-distance diagnostic. | Summarizes the global trend across sites while preserving site and sector variability. |

## Analysis

These plots are a physical sanity check for the generated labels. The main expected trend is decreasing RSRP with distance. The deviations around the trend are also important: they indicate that the dataset contains more than a simple distance-response relationship. Antenna alignment, clutter transitions, terrain obstruction, and sector geometry introduce local variability.

This behavior is central to the IINN evaluation. A useful model should learn distance-frequency loss while also accounting for antenna and environment effects that explain the spread around the distance trend.
