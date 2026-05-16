# Feature Description

The processed dataset is organized around propagation-relevant feature groups rather than anonymous tabular columns. This is important for IINN because the model is designed to preserve traceability between inputs, intermediate components, and physical propagation factors.

## Feature Groups

| Group | Columns | Physical meaning |
|---|---|---|
| Network configuration | `Max_Power`, `Frequency` | Defines transmit-power and carrier-frequency conditions. Frequency affects path loss and is central to frequency-shift evaluation. |
| Transmitter geometry | `Transmitter_X`, `Transmitter_Y`, `Transmitter_Height` | Defines the serving-site location and height used to compute link geometry and height-related effects. |
| UE geometry | `User_X`, `User_Y`, `User_Height`, `Distance_euclid` | Defines receiver location, receiver height, and transmitter-receiver distance. |
| Antenna orientation | `Transmitter_Azimuth`, `Transmitter_Mechanical_Downtilt`, `User_Azimuth`, `User_Downtilt` | Supports horizontal and vertical alignment analysis. |
| Antenna pattern | `Antenna_Gain`, `Antenna_Horizontal_Half-power_Beamwidth`, `Antenna_Vertical_Half-power_Beamwidth`, `Horizontal_Attenuation`, `Vertical_Attenuation` | Captures directional gain and attenuation effects caused by angular mismatch. |
| Environment | `Clutter Class` | Encodes land-use and morphology categories associated with shadowing, blockage, and local propagation variation. |
| Propagation labels | `Path Loss (DL) (dB)`, `RSRP` | Path loss and received-power labels exported from the planning tool. RSRP is the prediction target. |
| Serving-cell assignment | `Best Server` | Identifies the serving transmitter/sector selected by the radio-planning simulation. |

## Relation to IINN Components

The features support the interpretable components used in the model:

- distance-frequency loss uses distance and carrier frequency,
- antenna alignment uses azimuth, downtilt, beamwidth, and attenuation terms,
- site/environment effects use height-related and clutter variables,
- residual correction captures effects not explained by the named components.

This grouping allows the model to be audited at the level of physical propagation factors rather than only at the level of global feature importance.
