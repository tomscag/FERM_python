# MDL Model Cost for RM vs FERM

The MDL comparison asks whether the extra structure in FERM is justified by the improvement in predictive compression.

The total description length is:

```math
L(M, D) = L(M) + L(D \mid M)
```

where:

- `L(M)` is the cost of describing the model,
- `L(D | M)` is the cost of describing the observed migration counts given the model predictions.

For the data term, RM and FERM both produce origin-specific destination probabilities:

```math
p_{ij}^{M}
```

For each origin `i`, the observed destination counts are treated as a multinomial allocation of the fixed observed outflow:

```math
(y_{i1}, \ldots, y_{iJ}) \sim \mathrm{Multinomial}(N_i, p_{i1}^{M}, \ldots, p_{iJ}^{M})
```

with:

```math
N_i = \sum_j y_{ij}
```

Thus:

```math
L(D \mid M) = -\log_2 P(D \mid M)
```

Ignoring constants common to all models, the relevant part is:

```math
L(D \mid M) \approx - \sum_i \sum_j y_{ij} \log_2 p_{ij}^{M}
```

This is flow-weighted: improving probability on high-flow routes saves more bits than improving probability on low-flow routes.

## Current Model-Cost Convention

In the current notebook, the minimal model cost is counted as a discrete selection cost.

For RM:

```math
L(\mathrm{RM}) = 0
```

because RM is the baseline benchmark.

For FERM:

```math
L(\mathrm{FERM}) =
\log_2(K) + \log_2(S)
```

where:

- `K` is the number of candidate model variants,
- `S` is the number of sigma values in the fixed grid.

So this charges the cost of saying:

> I selected this FERM specification and this sigma value.

For example, if there are 6 candidate models and 5 sigma values:

```math
L(\mathrm{FERM}) = \log_2(6) + \log_2(5) \approx 4.9 \text{ bits}
```

This is intentionally minimal. It treats GDP, SCI, and other features as external covariates already available to the researcher.

## Strict Sigma-Matrix Sensitivity Check

The notebook also reports a stricter sensitivity version that charges for transmitting the full feature matrix:

```math
L(\Sigma) = n^2 b
```

where:

- `n` is the number of countries,
- `b` is the number of bits per matrix entry.

With 46 countries and 16 bits per entry:

```math
L(\Sigma) = 46^2 \times 16 = 33{,}856 \text{ bits}
```

This is a conservative upper-bound check if the feature matrix itself must be encoded.

## What Is Not Yet Charged

The current implementation does not fully charge for the exploratory research process. In particular, it does not automatically include:

- feature choice,
- normalization choice,
- combination rule choice,
- construction of the sigma grid,
- researcher degrees of freedom,
- the full cost of external datasets such as GDP or SCI.

A stricter version could add:

```math
L(\mathrm{FERM}) =
L(\mathrm{model\ family})
+ L(\mathrm{feature})
+ L(\mathrm{normalization})
+ L(\sigma)
+ L(\mathrm{combination\ rule})
+ L(\Sigma)
```

For discrete choices, this usually means adding terms like:

```math
\log_2(\#\mathrm{features})
+ \log_2(\#\mathrm{normalizations})
+ \log_2(\#\sigma\mathrm{\ values})
+ \log_2(\#\mathrm{combination\ rules})
```

These discrete penalties are usually small in bits. The larger cost would come from transmitting the full covariate data or matrix values.

## Interpretation

The current MDL result should be read as:

> Conditional on the candidate features, transformations, model variants, and sigma grid being fixed, does FERM compress the observed migration flows better than RM?

It should not be read as:

> FERM is absolutely simpler than RM.

FERM is more complex than RM. The question is whether the gain in `L(D | M)` is large enough to pay for the extra `L(M)`.

If FERM saves more bits in the data term than it costs to describe the model, then the extra feature structure is justified under the MDL criterion.
