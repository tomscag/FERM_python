# FERM 10-06-2026 Speaker Notes

## Slide 1 — Title
Today I present an extension of the Radiation Model that allows mobility predictions to use features beyond population. The application is international migration, where economic attractiveness, social ties, political relations, and historical corridors may all matter. The goal is to see whether these features improve route-level prediction while preserving the core RM mechanism.

## Slide 2 — From RM to FERM
The standard Radiation Model is elegant because it only needs population and distance ordering, but that is also its limitation. Traditional FERM keeps the same absorption process but lets each location have its own benefit distribution through a feature. Relational FERM goes one step further: attractiveness can depend on the specific origin-destination corridor.

## Slide 3 — Conceptual Move
The key conceptual move is from a universal destination attractiveness to a corridor-specific attractiveness. In traditional FERM, destination \(j\) has the same niche for every origin. In relational FERM, the same destination can be attractive from one origin and unattractive from another.

## Slide 4 — Corridor-Specific Attractiveness
I encode this through a matrix \(\Sigma_{ij}\). Positive entries mean attraction, negative entries mean repulsion, and zero means neutral. Traditional FERM is a special case where each row repeats the destination feature, while relational FERM allows every corridor to have its own value.

## Slide 5 — Simulation Mechanics
The simulation keeps the Radiation Model logic: migrants draw a threshold, destinations are visited in distance order, and the first destination whose offer exceeds the threshold absorbs the migrant. The only change is that thresholds and offers are shifted by feature values. Distance is still used for ordering destinations; it is not replaced by the features.

## Slide 6 — Features Used
The feature, or niche, shifts the Gaussian benefit distribution. I test GDP per capita as an economic signal, SCI as social connectedness, common religion, diplomatic disagreement, and Abel historical migrant stock. These variables live on very different scales, so normalization is a central part of the analysis.

## Slide 7 — Normalization
Normalization is not cosmetic here: it controls the effective contrast between countries or corridors. I compare min-max, log min-max, log rank, z-score, and for Abel an expected-stock log-ratio. The goal is to find transformations that are interpretable and stable, not just transformations that mechanically improve one metric.

## Slide 8 — Validation by Feature
Here I tune \(\sigma\) and normalization on the validation period. The left panels show median route-level improvement over RM, while the right panels show the share of routes that improve. Large \(\sigma\) weakens the feature effect and moves the model closer to RM-like behavior.

## Slide 9 — Best Single Features
At the best validation choices, GDP is the strongest single feature overall. This suggests that a destination-level economic attractiveness signal is very powerful for this migration setting. Some relational features still improve many routes, but their median gains are usually smaller.

## Slide 10 — Route-by-Route Winners
This plot asks a stricter question: for each OD pair, which model has the lowest absolute log error? GDP wins the largest share, but relational models do win non-trivial subsets of routes. So the relational signal is not useless; it is more localized.

## Slide 11 — Motivation for Combination
Since GDP performs best overall and SCI appears to capture corridor-specific social ties, the natural next question is whether they are complementary. In other words: can GDP provide the destination baseline while SCI corrects specific corridors?

## Slide 12 — Combining GDP and SCI
For the combined model, the diagonal remains the GDP-based origin threshold, while the off-diagonal offer combines destination GDP and corridor SCI. I test direct addition, off-diagonal rescaling after addition, and a signed Euclidean combination. This checks whether GDP and SCI should act linearly or as two dimensions of attractiveness.

## Slide 13 — Combined Validation
The additive GDP + SCI specification performs best in validation. This means the SCI correction adds information on top of GDP rather than simply duplicating it. The gain is visible both in median improvement and in the share of improved routes.

## Slide 14 — Combined Scatter vs RM
In this scatter, points below the diagonal are routes where the model improves over RM. The combined model moves many routes below the diagonal, especially where RM has larger errors. This suggests the combined feature is correcting meaningful route-level mistakes, not only tiny residuals.

## Slide 15 — Route Changes / Map
The map shows where the combined model changes route errors most strongly relative to the benchmark. Blue routes are improvements where the benchmark underestimates; red routes are cases where the combined model worsens by overestimating. This helps interpret the model spatially rather than only through aggregate scores.

## Slide 16 — Combined Route-by-Route Winners
This summarizes the route-level competition after adding the combined models. The additive GDP + SCI model wins the largest share of OD pairs, which supports the idea that destination-level and corridor-level information are complementary. The combined model is not just better on average; it wins route by route.

## Slide 17 — Summary and Next Steps
The main result is that FERM can incorporate exogenous and relational features while preserving the RM structure. GDP is the strongest single feature, SCI is the most promising relational signal, and the additive GDP + SCI model improves further. Next, I want to test robustness across periods, refine normalization choices, and better diagnose which corridors are helped or hurt.
