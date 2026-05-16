# Deployment Maps

These figures show the 3D urban map samples and the simulated radio deployment used for data generation.

| File | Description | Interpretation |
|---|---|---|
| `brussels_3p7ghz_deployment_map.png` | Brussels 3.7 GHz 3D map sample with simulated gNodeB deployment and UE distribution. | Shows a dense urban tile with building structures, sectorized sites, and UE placement. This supports the paper's claim that the dataset reflects a realistic urban propagation scenario rather than a synthetic point-cloud exercise. |
| `chicago_3p3ghz_deployment_map.png` | Chicago 3.3 GHz 3D map sample with simulated gNodeB deployment and UE distribution. | Provides the target-city deployment used for city-shift evaluation. The morphology differs from Brussels, making cross-city transfer a meaningful OOD setting. |

## Analysis

The maps are useful for visually checking site placement, sector coverage, and UE distribution before interpreting the tabular learning results. They also explain why city transfer is difficult: the model is not only seeing different coordinates, but also a different built environment, clutter mix, antenna geometry, and local obstruction pattern.

These figures are kept in the repository rather than the main paper because the simulation settings table already provides the reproducible parameters, while the maps mainly provide visual auditability.
