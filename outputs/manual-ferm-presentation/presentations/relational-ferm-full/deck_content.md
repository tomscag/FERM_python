# Relational FERM Full Presentation

## Core Claim

Relational FERM keeps the Radiation Model mechanism fixed and changes only the attractiveness matrix Sigma. In the relational version, off-diagonal entries Sigma_ij encode corridor-specific attractiveness or repulsion.

## Key Model Idea

For origin i and destination j:

- Threshold side: T_i is centered at Sigma_ii.
- Offer side: B_ij is centered at Sigma_ij.
- Destinations are scanned in increasing distance order.
- A particle is absorbed by the first destination whose offer exceeds its threshold.

## Model Variants

| Variant | Sigma structure | Interpretation |
|---|---|---|
| RM | Sigma_ii = 0, Sigma_ij = 0 | no feature-based attractiveness |
| Traditional FERM | Sigma_ii = mu_i, Sigma_ij = mu_j | destination-specific attractiveness |
| Relational FERM | Sigma_ii = 0, Sigma_ij = delta_ij | corridor-specific attractiveness |
| Combined model | Sigma_ii = mu_i, Sigma_ij combines mu_j and delta_ij | destination and corridor information |

## Empirical Result

GDP is the strongest global feature. Social connectedness is the strongest relational feature. Abel and diplomatic disagreement are weak in the current specification.

## Final Interpretation

The relational model is viable but feature-dependent. The next specification should combine destination attractiveness and corridor-specific relational structure.
