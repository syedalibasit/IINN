# RSRP-Distance Profiles

This folder contains sector-wise RSRP trends versus transmitter--receiver distance.

## Purpose

These plots check whether the generated RSRP labels follow expected propagation behavior. Median RSRP should generally decrease with distance, while local deviations may occur due to antenna orientation, clutter transitions, terrain obstruction, and morphology.

## Suggested Files

| File | Description |
|---|---|
| `brussels_site0_rsrp_distance.png` | Sector-wise RSRP-distance trend for Brussels Site 0. |
| `brussels_site1_rsrp_distance.png` | Sector-wise RSRP-distance trend for Brussels Site 1. |
| `brussels_all_sites_rsrp_distance.png` | Consolidated RSRP-distance trends across Brussels sites. |

## Interpretation

These plots support the paper's claim that the dataset contains both large-scale propagation structure and local environment-dependent deviations. This is important because IINN is evaluated not only on prediction accuracy, but also on traceability to propagation-relevant factors.
