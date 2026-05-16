# Model Architecture Figures

This folder contains architecture-level visual material for IINN.

| File | Description | Interpretation |
|---|---|---|
| `equation_wired_iinn_full_graph.png` | Full internal graph of the original equation-wired IINN, including scalar streams, PLO layers, and mathematical operator blocks. | The figure shows how input variables are kept as physically named streams before being combined through propagation-relevant operations. It is useful for auditing the implementation, but it is too dense for the main paper. |

## Analysis

The full graph documents the one-to-one mapping between the analytical propagation scaffold and the original NN computation. It supports the paper's structural-interpretability claim by showing that the model computation is not formed by immediately concatenating all inputs into an unconstrained hidden representation.

The ablation results should be read alongside this figure. The equation-wired graph is interpretable, but the cross-propagation study shows that overly rigid equation wiring can over-align the model with one label generator. The updated white-box Physics-IINN keeps named components while replacing brittle simulator-specific constants with trainable, calibrated terms.
