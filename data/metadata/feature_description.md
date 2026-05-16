# Feature Description

This file describes the main feature groups used for RSRP prediction.

## Feature Groups

| Feature group | Examples | Physical relevance |
|---|---|---|
| Network configuration | transmit power, carrier frequency, site identifier | Defines radio operating conditions. |
| Geometry | transmitter--receiver distance, relative displacement, UE/site coordinates | Captures distance-dependent attenuation and spatial deployment structure. |
| Antenna configuration | azimuth, downtilt, horizontal/vertical beamwidth, antenna gain | Captures directional gain and alignment effects. |
| Height-related variables | gNodeB height, UE height, height difference, building height | Affects LoS/NLoS conditions and vertical geometry. |
| Environment | clutter class, street width, local morphology descriptors | Captures shadowing, blockage, and morphology-dependent propagation effects. |
| Target | RSRP | Received signal strength used for supervised prediction. |

## Interpretation

These features were selected because they correspond to propagation-relevant factors used in analytical radio models and professional radio-planning workflows. They also support the IINN objective of decomposing predictions into physically meaningful components.
