---
title: Abel Sigma Normalization Options for FERM
tags:
  - FERM
  - Abel
  - Sigma
  - normalization
  - migration-networks
---

# Abel Sigma Normalization Options for FERM

## Core Issue

The Abel bilateral migrant-stock matrix is nonnegative:

$$
A_{ij} \geq 0
$$

where \(A_{ij}\) is the migrant stock from origin \(i\) to destination \(j\).

However, in FERM the feature matrix \(\Sigma\) acts as a signed shift in the latent absorption mechanism. Therefore we need:

$$
\Sigma_{ij} > 0
$$

to mean that destination \(j\) is more attractive than baseline for origin \(i\), and:

$$
\Sigma_{ij} < 0
$$

to mean that destination \(j\) is less attractive than baseline for origin \(i\).

So the normalization is not a harmless preprocessing step. It defines the meaning of the Abel feature.

## Why Results Change So Much

FERM is sensitive to \(\Sigma\) because \(\Sigma_{ij}\) shifts the latent offer distribution for route \(i \to j\). Changing the normalization changes:

1. The baseline against which a corridor is judged.
2. Whether large countries or small countries are favored.
3. Whether the comparison is made within origins, within destinations, or globally.
4. The scale of the feature relative to the model parameter \(\sigma\).

Therefore two normalizations can produce completely different predictions even when they use the same Abel stock data.

## Absorption Interpretation

In the FERM structure, a migrant from origin \(i\) compares an origin-specific threshold with destination-specific offers:

$$
T_i \quad \text{vs.} \quad O_{ij}
$$

The relevant comparison is therefore conditional on the origin:

> For migrants from origin \(i\), is destination \(j\) unusually attractive relative to other possible destinations?

This suggests that the most model-consistent Abel normalization should be row-based, because rows correspond to origin-specific choice environments.

Destination-based normalizations answer a different question:

> Within destination \(j\), how strongly represented is origin \(i\)?

That can be substantively meaningful, but it is less directly aligned with the absorption structure if \(\Sigma_{ij}\) is interpreted as an origin-to-destination offer shift.

## Raw Abel Stock

### Formula

$$
x_{ij} = A_{ij}
$$

### Interpretation

Routes with larger historical migrant stocks receive larger feature values.

### Problem

This mostly captures country size and historical migration scale. Large origins and large destinations dominate.

### Assessment

Not recommended as the main specification.

## Origin-Population Normalization

### Formula

$$
x_{ij} = \frac{A_{ij}}{P_i}
$$

where \(P_i\) is the population of origin \(i\).

### Interpretation

This measures the size of the stock from \(i\) to \(j\) relative to the origin population.

### What It Emphasizes

Origin-side migration exposure.

### Strength

Controls for the fact that large origins mechanically produce larger migrant stocks.

### Weakness

It does not control for the fact that some destinations are global immigration hubs.

### Model Fit

Moderately aligned with FERM because the denominator is origin-specific.

## Destination-Population Normalization

### Formula

$$
x_{ij} = \frac{A_{ij}}{P_j}
$$

where \(P_j\) is the population of destination \(j\).

### Interpretation

This measures how visible origin \(i\)'s community is inside destination \(j\).

### What It Emphasizes

Destination-side diaspora concentration.

### Strength

Useful if the mechanism is interpreted as destination-side absorption through existing communities.

### Weakness

Can explode for small destinations. A modest migrant stock in a small country may become extremely large.

### Model Fit

Less directly aligned with the FERM absorption structure, because it compares origins within the same destination rather than comparing destinations within the same origin.

## Row-Share Normalization

### Formula

$$
x_{ij} = \frac{A_{ij}}{\sum_k A_{ik}}
$$

### Interpretation

This measures the share of origin \(i\)'s historical migrant stock located in destination \(j\).

### What It Emphasizes

The historical destination distribution of each origin.

### Strength

This is strongly aligned with FERM because it compares destinations within each origin's choice set.

### Weakness

It removes differences in total emigration intensity across origins. An origin with very little migration history receives the same row total as a highly migration-intensive origin.

### Model Fit

Very strong. This is probably the most natural normalization if \(\Sigma_{ij}\) is interpreted as an origin-specific destination preference.

## Column-Share Normalization

### Formula

$$
x_{ij} = \frac{A_{ij}}{\sum_k A_{kj}}
$$

### Interpretation

This measures the share of destination \(j\)'s migrant stock coming from origin \(i\).

### What It Emphasizes

The composition of each destination's immigrant population.

### Strength

Useful if the theoretical mechanism is destination-side community composition.

### Weakness

Less natural for an origin-conditioned route-choice model.

### Model Fit

Potentially useful as a robustness check, but not my preferred main specification.

## Expected-Stock Residual

### Formula

First define the expected stock under independence:

$$
E_{ij} = \frac{\left(\sum_k A_{ik}\right)\left(\sum_k A_{kj}\right)}{\sum_{i,j} A_{ij}}
$$

Then define:

$$
r_{ij} = \log \left( \frac{A_{ij} + \epsilon}{E_{ij} + \epsilon} \right)
$$

### Interpretation

This asks whether corridor \(i \to j\) is over- or under-represented relative to what would be expected from:

1. How migration-intensive origin \(i\) is.
2. How important destination \(j\) is globally.

### Positive and Negative Values

If:

$$
A_{ij} > E_{ij}
$$

then:

$$
r_{ij} > 0
$$

If:

$$
A_{ij} < E_{ij}
$$

then:

$$
r_{ij} < 0
$$

### Strength

This gives negative values naturally and controls for both origin and destination margins.

### Weakness

It is less directly row-choice based than row-share normalization, because it also adjusts for destination popularity.

### Model Fit

Very strong if \(\Sigma_{ij}\) is interpreted as dyadic affinity beyond marginal migration size.

## Log Transform

### Formula

$$
z_{ij} = \log(1 + x_{ij})
$$

### Purpose

Abel stocks are extremely skewed. The log transform reduces the dominance of very large corridors.

### Limitation

The log transform alone does not create negative values. It must be combined with centering.

## Global Centering

### Formula

$$
\Sigma_{ij} = z_{ij} - \bar{z}
$$

### Interpretation

A corridor is positive if it is above the global Abel baseline.

### Strength

Simple and transparent.

### Weakness

Can be unfair when origins have very different migration profiles.

## Row Centering

### Formula

$$
\Sigma_{ij} = z_{ij} - \bar{z}_{i \cdot}
$$

### Interpretation

A destination is positive if it is unusually strong for origin \(i\).

### Strength

This is highly aligned with FERM's origin-conditioned absorption structure.

### Weakness

Each origin is centered separately, so cross-origin differences in overall migration network strength are removed.

## Column Centering

### Formula

$$
\Sigma_{ij} = z_{ij} - \bar{z}_{\cdot j}
$$

### Interpretation

An origin is positive if it is unusually represented within destination \(j\).

### Strength

Useful for destination-composition mechanisms.

### Weakness

Less aligned with an origin-conditioned offer comparison.

## Z-Score Scaling

### Formula

$$
\Sigma_{ij} = \frac{z_{ij} - \mu}{s}
$$

### Purpose

Z-score scaling makes the feature dimensionless and creates both positive and negative values.

### Problem

Plain z-score scaling is sensitive to outliers. For Abel data, this can create very large positive values for a few corridors, which can dominate FERM predictions.

### Assessment

Useful, but should usually be clipped or made robust.

## Robust Z-Score Scaling

### Formula

$$
\Sigma_{ij} = \frac{z_{ij} - \mathrm{median}(z)}{\mathrm{IQR}(z)}
$$

or:

$$
\Sigma_{ij} = \frac{z_{ij} - \mathrm{median}(z)}{\mathrm{MAD}(z)}
$$

### Purpose

This reduces sensitivity to extreme Abel corridors.

### Assessment

Better than plain z-score for sparse migration-stock data.

## Clipping

### Formula

$$
\Sigma_{ij} = \min\left(c, \max\left(-c, \Sigma_{ij}\right)\right)
$$

Common choices:

$$
c = 1
$$

or:

$$
c = 2
$$

### Purpose

Clipping prevents a small number of extreme corridors from dominating the absorption probabilities.

### Interpretation

Clipping is a regularization choice. It says that Abel can shift attractiveness, but only up to a bounded amount.

### Assessment

Recommended when using any z-score-like transformation.

## Min-Max Centering

### Formula

$$
\Sigma_{ij} = 2 \frac{x_{ij} - \min(x)}{\max(x) - \min(x)} - 1
$$

### Strength

Guarantees values in \([-1, 1]\).

### Problem

With sparse Abel data, many corridors can collapse to the lower bound \(-1\). This makes the feature almost binary.

### Assessment

Not recommended as the main specification.

## Rank-Based Scaling

### Formula

$$
\Sigma_{ij} = 2 \cdot \mathrm{rankpct}(x_{ij}) - 1
$$

### Interpretation

Only the ordering of Abel corridors matters, not the magnitude.

### Strength

Very robust to outliers.

### Weakness

Throws away magnitude information.

### Assessment

Good robustness check.

## Recommended Main Specifications

### Option 1: Row-Share Log Centered

Use:

$$
x_{ij} = \frac{A_{ij}}{\sum_k A_{ik}}
$$

then:

$$
\Sigma_{ij} = \log(x_{ij} + \epsilon) - \frac{1}{J_i}\sum_j \log(x_{ij} + \epsilon)
$$

This is the most directly aligned with FERM's absorption structure.

### Option 2: Row-Share Robust Z-Score Clipped

Use:

$$
x_{ij} = \frac{A_{ij}}{\sum_k A_{ik}}
$$

then:

$$
\Sigma_{ij} = \mathrm{clip}\left(
\frac{\log(x_{ij} + \epsilon) - \mathrm{median}_j}{\mathrm{IQR}_j},
-c,
c
\right)
$$

This is a more regularized version of Option 1.

### Option 3: Expected-Stock Log Ratio Clipped

Use:

$$
\Sigma_{ij} =
\mathrm{clip}\left(
\log \left( \frac{A_{ij} + \epsilon}{E_{ij} + \epsilon} \right),
-c,
c
\right)
$$

This is the best option if the goal is to estimate dyadic affinity net of origin and destination migration size.

## Preferred Interpretation

For the current FERM setup, the strongest theoretical argument is for a row-based normalization:

> Since absorption is evaluated for migrants from a given origin, Abel should be normalized within origin rows. This makes \(\Sigma_{ij}\) represent whether destination \(j\) is unusually attractive in origin \(i\)'s historical migration network.

The expected-stock residual is also defensible:

> It treats Abel as a dyadic affinity matrix and removes both origin and destination marginal effects.

Therefore, I would use row-share normalization as the main model-consistent specification and expected-stock residual as the main robustness check.

## Empirical Strategy

Do not choose the normalization only because it performs best on the test set.

Suggested workflow:

1. Choose the main normalization from the model structure.
2. Tune \(\sigma\) only on validation data.
3. Freeze the normalization and \(\sigma\).
4. Report test performance.
5. Add a robustness table with alternative normalizations.

Recommended robustness table:

| Specification | Normalization | Transformation | Interpretation |
|---|---|---|---|
| Main | Row share | Row log centered or robust clipped | Origin-specific destination preference |
| Robustness 1 | Origin population | Row centered | Origin exposure |
| Robustness 2 | Destination population | Column centered | Destination community visibility |
| Robustness 3 | Expected stock | Log ratio clipped | Dyadic affinity net of margins |
| Robustness 4 | Rank | Rank centered | Robust ordering only |

## Short Paper Explanation

Because FERM evaluates destination offers relative to an origin-specific threshold, Abel migrant stocks are normalized within origin rows. This makes the feature \(\Sigma_{ij}\) capture whether destination \(j\) is historically over- or under-represented in origin \(i\)'s migration network. Since Abel stocks are highly skewed, the normalized values are log-transformed, centered, and optionally clipped to prevent a small number of very large corridors from dominating the absorption probabilities. As a robustness check, we also consider a dyadic residual specification based on the ratio between observed Abel stocks and expected stocks under independence of origin and destination margins.

