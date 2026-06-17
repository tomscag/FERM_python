# FERM Results Presentation

This is the editable story text used to generate the PowerPoint deck.

## Thesis

GDP is the strongest single feature for FERM in this test, but social connectedness is the only relational feature with a meaningful positive signal. The relational model is not falsified; the evidence says bilateral feature quality and normalization determine whether the extension helps.

## Selected Normalizations

| Feature group | Selected specification | Sigma |
|---|---|---:|
| GDP | log zscore | 8 |
| Abel stock | expected-stock log-ratio min-max | 8 |
| Social connectedness | min-max | 5 |
| Common religion | min-max | 8 |
| Diplomatic disagreement | log rank | 8 |

## Test Results

| Model | Median improvement | Share better than RM | Pearson log | Zero prediction share |
|---|---:|---:|---:|---:|
| Traditional FERM: GDP | 0.037 | 0.568 | 0.689 | 0.152 |
| Relational FERM: Social connectedness | 0.026 | 0.562 | 0.658 | 0.157 |
| Relational FERM: Common religion | 0.002 | 0.502 | 0.654 | 0.197 |
| Abel stock | -0.003 | 0.446 | 0.643 | 0.207 |
| Relational FERM: Diplomatic disagreement | -0.008 | 0.413 | 0.626 | 0.228 |
| RM | baseline | baseline | 0.646 | 0.000 |

## Interpretation

GDP wins globally because it captures destination absorption: high-opportunity destinations attract flows from many origins.

SCI is the strongest relational feature. It does not beat GDP overall, but it wins a large subset of routes and improves many medium/high-flow OD pairs.

Common religion is marginal. Abel is empirically unstable in this specification because it zero-predicts many nontrivial routes. Diplomatic disagreement underperforms RM globally.

## Main Conclusion

The relational extension is informative but feature-dependent. The next step is not GDP versus relational features; it is a combined destination-plus-relational specification tested out of sample.
