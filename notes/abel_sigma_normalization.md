---
title: Abel Sigma Normalization for Relational FERM
tags:
  - FERM
  - Abel
  - Sigma
  - normalization
  - relational-FERM
  - migration-networks
---

# Abel Sigma Normalization for Relational FERM

## Purpose

This note rethinks how to transform Abel bilateral migrant stocks into a relational FERM attractiveness matrix.

The goal is not simply to find a transformation that performs well. The goal is to construct a meaningful signed matrix:

$$
\Sigma = (\Sigma_{ij})
$$

that is consistent with the model note:

$$
T_i \sim \max_{1,\dots,m_i} N(\Sigma_{ii}, \sigma^2),
$$

$$
B_{ij} \sim \max_{1,\dots,n_j} N(\Sigma_{ij}, \sigma^2).
$$

In the pure relational FERM:

$$
\Sigma_{ii} = 0,
$$

$$
\Sigma_{ij} = \delta_{ij}, \quad i \neq j.
$$

So the Abel problem is:

> How should the nonnegative Abel migrant-stock matrix \(A_{ij}\) be transformed into a signed relational score \(\delta_{ij}\)?

## Model Anchor

The radiation mechanism is unchanged across RM, traditional FERM, and relational FERM.

Particles:

1. leave origin \(i\),
2. carry a threshold \(T_i\),
3. scan destinations in increasing distance from \(i\),
4. are absorbed by the first destination whose offer \(B_{ij}\) exceeds the threshold.

Only the centers of the latent distributions change.

For RM:

$$
\Sigma_{ii} = 0, \quad \Sigma_{ij} = 0.
$$

All thresholds and offers are centered at zero. Prediction differences come only from population and distance ordering.

For relational FERM:

$$
\Sigma_{ii} = 0, \quad \Sigma_{ij} = \delta_{ij}.
$$

The threshold remains neutral. The off-diagonal entries shift corridor-specific offers.

Therefore:

$$
\delta_{ij} > 0
$$

means corridor \(i \to j\) is more attractive than neutral.

$$
\delta_{ij} = 0
$$

means the corridor is neutral.

$$
\delta_{ij} < 0
$$

means the corridor is less attractive than neutral.

## Important Consequence for Sigma

The scale of \(\Sigma\) only matters relative to \(\sigma\).

The relevant contrast is roughly:

$$
\frac{\Sigma_{ij} - \Sigma_{ii}}{\sigma}.
$$

In pure relational FERM:

$$
\Sigma_{ii} = 0,
$$

so the active contrast is approximately:

$$
\frac{\delta_{ij}}{\sigma}.
$$

This means:

- small \(\sigma\): the signed Abel feature has strong influence;
- large \(\sigma\): the feature centers are washed out;
- very large \(\sigma\): the model moves toward the common-distribution/RM-like limit.

Therefore, if the best validation \(\sigma\) is always very large, the correct interpretation is not that Abel is stronger. It is that the model prefers to dilute Abel and return toward the neutral radiation mechanism.

## Why Abel Is Difficult

The Abel matrix is:

$$
A_{ij} \geq 0.
$$

It is not already a signed attractiveness variable.

It is also:

- highly sparse,
- heavy-tailed,
- mechanically affected by origin size,
- mechanically affected by destination size,
- partly an outcome of distance, population, and historical migration frictions.

So using Abel directly risks confusing several things:

1. origin emigration scale,
2. destination immigration scale,
3. corridor-specific diaspora affinity,
4. global migration volume,
5. missing or zero historical links.

The normalization must decide what the neutral baseline is.

## What A Good Abel Sigma Must Do

A usable Abel-based \(\delta_{ij}\) should satisfy five requirements.

1. It must be signed:

$$
\delta_{ij} \in \mathbb{R}.
$$

2. Zero must mean neutral:

$$
\delta_{ij} = 0
$$

should mean neither especially attractive nor especially unattractive.

3. Positive values must have the correct orientation:

$$
\delta_{ij} > 0
$$

should mean stronger historical corridor affinity.

4. Scale must be controlled, because \(\delta_{ij}/\sigma\) determines strength.

5. The transformation must be fixed consistently across validation and test splits.

## The Central Choice: Baseline

Every Abel normalization answers:

> Compared to what is \(A_{ij}\) large or small?

The baseline defines the meaning of negative values.

Different baselines produce different models.

## Option 1: Raw Stock

### Formula

$$
x_{ij} = A_{ij}.
$$

Then center and scale:

$$
\delta_{ij} = g(x_{ij}).
$$

### Interpretation

Routes with larger historical migrant stocks become more attractive.

### Problem

Raw stock mainly captures size. Large origins and large destinations dominate.

### Assessment

Not recommended. It is not a clean relational feature.

## Option 2: Origin-Population Rate

### Formula

$$
x_{ij} = \frac{A_{ij}}{P_i},
$$

where \(P_i\) is origin population.

### Interpretation

This asks:

> How large is the stock from origin \(i\) to destination \(j\) relative to origin population?

### Strength

Controls for the fact that large origins mechanically produce more migrants.

### Weakness

It does not control for destination size or destination popularity.

### Model Meaning

This is an origin exposure measure. It is reasonable if the Abel stock is interpreted as the fraction of origin society historically connected to destination \(j\).

### Assessment

Useful robustness check. Not the cleanest main relational normalization.

## Option 3: Destination-Population Rate

### Formula

$$
x_{ij} = \frac{A_{ij}}{P_j},
$$

where \(P_j\) is destination population.

### Interpretation

This asks:

> How visible is origin \(i\)'s community inside destination \(j\)?

### Strength

Captures destination-side diaspora concentration.

### Weakness

Can strongly inflate small destinations. It compares origins within a destination, not destinations within an origin.

### Model Meaning

This is defensible only if the mechanism is destination-side community visibility.

### Assessment

Not the preferred main specification for an origin-scanning absorption model.

## Option 4: Row Share

### Formula

$$
x_{ij} = \frac{A_{ij}}{\sum_{k \neq i} A_{ik}}.
$$

### Interpretation

This asks:

> Among origin \(i\)'s historical migrant stock, what share is located in destination \(j\)?

### Why It Fits The Absorption Structure

In the model, particles are emitted from one origin and scan possible destinations.

So an origin-conditioned Abel variable is natural:

> For origin \(i\), is destination \(j\) historically over- or under-represented relative to other destinations?

### How To Make It Signed

A simple signed version is row-centered log share:

$$
z_{ij} = \log(x_{ij} + \epsilon),
$$

$$
\delta_{ij} = z_{ij} - \frac{1}{J_i}\sum_{k \neq i} z_{ik}.
$$

Then:

- \(\delta_{ij} > 0\): destination \(j\) is above origin \(i\)'s average historical destination share;
- \(\delta_{ij} < 0\): destination \(j\) is below origin \(i\)'s average historical destination share.

### Major Risk

Zeros are dangerous.

If \(x_{ij}=0\), then:

$$
\log(x_{ij}+\epsilon)
$$

can become extremely negative.

This can turn absence of historical stock into strong repulsion.

That may be too strong because:

- Abel stock is historical and incomplete;
- no stock does not necessarily mean impossible or unattractive;
- routes may exist in current flows even without old Abel stock.

### Fix

Use smoothing before the log:

$$
\tilde{x}_{ij}
=
\frac{x_{ij} + \alpha / J_i}{1+\alpha}.
$$

Then:

$$
z_{ij} = \log(\tilde{x}_{ij}),
$$

$$
\delta_{ij} = z_{ij} - \bar{z}_{i\cdot}.
$$

Optionally clip:

$$
\delta_{ij} =
\mathrm{clip}(\delta_{ij}, -c, c).
$$

### Strength

This is the most aligned with the origin-conditioned scanning structure.

### Weakness

It removes total emigration intensity. Every origin row is made comparable even if one origin has a huge historical diaspora and another has almost none.

### Assessment

Best main specification if Abel is interpreted as origin-specific destination preference.

## Option 5: Column Share

### Formula

$$
x_{ij} = \frac{A_{ij}}{\sum_{k \neq j} A_{kj}}.
$$

### Interpretation

This asks:

> Among destination \(j\)'s immigrant stock, what share comes from origin \(i\)?

### Strength

Captures destination immigrant composition.

### Weakness

Less aligned with the model's origin-side scanning process.

### Assessment

Useful only if the theory is destination-side community composition. Otherwise, use as robustness check.

## Option 6: Expected-Stock Residual

### Formula

Define the expected stock under independence of origin and destination margins:

$$
E_{ij}
=
\frac{\left(\sum_k A_{ik}\right)
\left(\sum_k A_{kj}\right)}
{\sum_{i,j} A_{ij}}.
$$

Then:

$$
\delta_{ij}
=
\log
\left(
\frac{A_{ij}+\epsilon}
{E_{ij}+\epsilon}
\right).
$$

### Interpretation

This asks:

> Is corridor \(i \to j\) over- or under-represented relative to what would be expected from origin \(i\)'s total emigration stock and destination \(j\)'s total immigration stock?

### Why It Is Attractive

It naturally creates signed values:

$$
A_{ij} > E_{ij} \Rightarrow \delta_{ij} > 0,
$$

$$
A_{ij} < E_{ij} \Rightarrow \delta_{ij} < 0.
$$

It controls for both margins:

- origin migration scale,
- destination migration scale.

### Weakness

It is less purely origin-choice based than row share. It removes destination popularity, which may or may not be part of the desired network effect.

### Assessment

Best specification if Abel is interpreted as dyadic affinity net of marginal migration size.

This is a very strong robustness specification, and possibly a main specification if the paper wants Abel to represent corridor-specific over-representation rather than origin-specific choice shares.

## Option 7: Rank-Based Abel

### Formula

Within each origin:

$$
\delta_{ij}
=
2 \cdot \mathrm{rankpct}_j(A_{ij}) - 1.
$$

Or globally:

$$
\delta_{ij}
=
2 \cdot \mathrm{rankpct}(A_{ij}) - 1.
$$

### Interpretation

Only the ordering of Abel links matters, not their magnitude.

### Strength

Very robust to heavy tails and extreme historical corridors.

### Weakness

Throws away magnitude information.

### Assessment

Good robustness check, not preferred as main unless magnitudes are considered unreliable.

## Transformation Choices

Normalization chooses \(x_{ij}\).

Transformation chooses how \(x_{ij}\) becomes signed \(\delta_{ij}\).

## Log Compression

For heavy-tailed Abel stocks, use a log or log-ratio before centering:

$$
z_{ij} = \log(x_{ij}+\epsilon).
$$

Log compression prevents a few huge corridors from dominating the Gaussian mean shifts.

## Centering

Centering defines neutrality.

### Global Centering

$$
\delta_{ij}=z_{ij}-\bar{z}.
$$

Neutral means average across all corridors.

### Row Centering

$$
\delta_{ij}=z_{ij}-\bar{z}_{i\cdot}.
$$

Neutral means average for origin \(i\).

This is natural for origin-conditioned destination comparison.

### Column Centering

$$
\delta_{ij}=z_{ij}-\bar{z}_{\cdot j}.
$$

Neutral means average for destination \(j\).

This is natural for destination-composition comparison.

## Scaling

Scaling determines how large \(\delta_{ij}\) is relative to \(\sigma\).

This is crucial.

If \(\delta_{ij}\) is tiny, the feature does almost nothing unless \(\sigma\) is tiny.

If \(\delta_{ij}\) is huge, the feature dominates unless \(\sigma\) is huge.

Therefore, \(\Sigma\) scale and \(\sigma\) cannot be interpreted separately.

## Z-Score Scaling

Formula:

$$
\delta_{ij}
=
\frac{z_{ij}-\mu_z}{s_z}.
$$

Useful for comparability, but sensitive to outliers.

For Abel, ordinary z-score can be unstable because Abel is sparse and heavy-tailed.

## Robust Scaling

Formula:

$$
\delta_{ij}
=
\frac{z_{ij}-\mathrm{median}(z)}
{\mathrm{IQR}(z)}.
$$

or:

$$
\delta_{ij}
=
\frac{z_{ij}-\mathrm{median}(z)}
{\mathrm{MAD}(z)}.
$$

Better for heavy-tailed Abel data.

## Clipping

Formula:

$$
\delta_{ij}
=
\mathrm{clip}(\delta_{ij}, -c, c).
$$

Clipping is not just technical. It is a substantive regularization:

> Abel can shift the offer distribution, but only up to a bounded amount.

This may be necessary because a few old high-stock corridors should not necessarily dominate current absorption probabilities.

## Min-Max Scaling

Formula:

$$
\delta_{ij}
=
2\frac{x_{ij}-\min(x)}
{\max(x)-\min(x)}
-1.
$$

This guarantees \([-1,1]\), but with sparse Abel data it can collapse many routes to \(-1\).

Not recommended as main specification.

## Missing And Zero Values

Zero and missing Abel values are not the same.

Possible meanings:

1. true absence of historical stock,
2. measurement missingness,
3. stock below reporting threshold,
4. historical absence but current route possible.

Treating zero as strongly negative is a modeling assumption.

Treating zero as neutral is also a modeling assumption.

For Abel, smoothing is usually preferable:

$$
x_{ij} \leftarrow x_{ij} + \epsilon
$$

or, for row shares:

$$
\tilde{x}_{ij}
=
\frac{x_{ij}+\alpha/J_i}{1+\alpha}.
$$

The parameter \(\alpha\) controls how strongly zeros are pulled toward a uniform baseline.

## Split Comparability

The transformation should not be fit separately on validation and test.

A cleaner procedure:

1. choose a reference Abel matrix and country set,
2. estimate transformation parameters on that reference,
3. apply the same transformation to validation and test.

Otherwise the same raw Abel corridor can receive different \(\delta_{ij}\) in different splits.

## Recommended Main Choices

There is no single mathematically forced normalization. The right choice depends on what Abel is supposed to mean.

But there are two defensible main candidates.

## Main Candidate A: Origin-Conditioned Network Preference

Use this if the interpretation is:

> Migrants from origin \(i\) are more likely to be absorbed by destinations that historically received a larger share of origin \(i\)'s migrants.

Pipeline:

1. compute row shares:

$$
x_{ij}=\frac{A_{ij}}{\sum_k A_{ik}};
$$

2. smooth zeros:

$$
\tilde{x}_{ij}
=
\frac{x_{ij}+\alpha/J_i}{1+\alpha};
$$

3. log:

$$
z_{ij}=\log(\tilde{x}_{ij});
$$

4. row-center:

$$
\delta_{ij}=z_{ij}-\bar{z}_{i\cdot};
$$

5. optionally robust-scale or clip:

$$
\delta_{ij}=\mathrm{clip}(\delta_{ij},-c,c).
$$

Set:

$$
\Sigma_{ii}=0,\quad \Sigma_{ij}=\delta_{ij}.
$$

This is most aligned with the origin-conditioned scanning mechanism.

## Main Candidate B: Dyadic Over-Representation

Use this if the interpretation is:

> Abel captures corridor-specific historical affinity beyond origin and destination migration scale.

Pipeline:

1. compute expected stock:

$$
E_{ij}
=
\frac{
\left(\sum_k A_{ik}\right)
\left(\sum_k A_{kj}\right)}
{\sum_{i,j}A_{ij}};
$$

2. compute log ratio:

$$
\delta_{ij}
=
\log\left(
\frac{A_{ij}+\epsilon}
{E_{ij}+\epsilon}
\right);
$$

3. robust-scale or clip:

$$
\delta_{ij}=\mathrm{clip}(\delta_{ij},-c,c).
$$

Set:

$$
\Sigma_{ii}=0,\quad \Sigma_{ij}=\delta_{ij}.
$$

This is the cleanest way to remove origin and destination marginal size effects.

## How To Interpret High Best Sigma

Because \(\sigma\) controls the common variance of threshold and offer distributions, large \(\sigma\) makes mean shifts less important.

Therefore:

> If validation prefers very large \(\sigma\), the model is likely diluting the Abel feature and moving back toward RM-like behavior.

This should not be interpreted as Abel becoming stronger.

It should be interpreted as evidence that the active Abel \(\Sigma\) may be harmful, too noisy, too sparse, or badly scaled.

## Empirical Testing Strategy

Do not choose the normalization only from test performance.

Recommended workflow:

1. Choose the main normalization from theory.
2. Fix the construction of \(\Sigma\).
3. Tune \(\sigma\), smoothing, and clipping only on validation data.
4. Evaluate once on test data.
5. Report robustness across alternative theoretically meaningful normalizations.

Recommended robustness table:

| Specification | Abel Baseline | Transform | Interpretation |
|---|---|---|---|
| Main A | Origin row share | smoothed log, row-centered, clipped | origin-specific destination preference |
| Main B | Expected stock | log observed/expected, clipped | dyadic affinity net of margins |
| Robustness 1 | Origin population | log, centered | origin exposure |
| Robustness 2 | Destination population | log, centered | destination community visibility |
| Robustness 3 | Rank within origin | centered rank | robust origin-specific ordering |
| Robustness 4 | Raw/log global | global centered | size-dominated benchmark |

## Paper-Style Explanation For Row-Share Specification

In the relational FERM, the diagonal of \(\Sigma\) is kept neutral and the off-diagonal entries shift corridor-specific offer distributions. Since particles are emitted from an origin and scan possible destinations, Abel migrant stocks are normalized within origin rows. This transforms Abel into an origin-conditioned historical destination preference. The row shares are smoothed to avoid treating zero historical stock as an absorbing impossibility, log-transformed to reduce skewness, centered within each origin so that zero represents a neutral destination for that origin, and clipped to prevent a small number of historical corridors from dominating the latent offer means.

## Paper-Style Explanation For Expected-Stock Specification

In the relational FERM, Abel migrant stocks are used to construct a signed corridor-specific offer shift. Because raw stocks reflect both origin emigration scale and destination immigration scale, we compare observed stocks to an independence benchmark based on the origin and destination margins. The resulting log observed-to-expected ratio is positive when a corridor is historically over-represented and negative when it is under-represented. This produces a signed dyadic affinity measure suitable for the off-diagonal entries of \(\Sigma\), with the diagonal kept neutral.

## Takeaway

The Abel normalization must answer one substantive question:

> What does neutral mean for a historical migrant-stock corridor?

If neutral means average destination preference within an origin, use the row-share specification.

If neutral means expected stock after accounting for origin and destination margins, use the expected-stock residual.

Both are defensible. What is not defensible is inserting raw Abel stock or choosing a transformation only because it makes test-set plots look better.

