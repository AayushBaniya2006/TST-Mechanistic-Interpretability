Published my AI safety project from BlueDot Impact.

I added statistical validation to an activation patching framework for Time Series Transformers - checking whether the attention heads it identifies as "important" actually are.

On simple tasks (9-10 classes), the method works. 21-24 out of 24 heads survive FDR correction, effect sizes are large, head rankings stay stable under input perturbations.

On a 96-class astronomy dataset, every significant head was a false positive. The naive analysis flagged 8 heads. After correcting for multiple comparisons, zero survived. Effect sizes were negligible. Head rankings reshuffled under minimal noise. The heatmaps looked fine the whole time - you'd only catch the problem by running the stats.

With 96 output classes, probability mass per class is ~1%. Single-head patching can't produce shifts large enough to distinguish from noise at that scale.

Attention weights also predict almost nothing about causal importance (rho ~ 0.2). Where the model looks is not what it uses.

~3,840 lines of statistical infrastructure, 174 tests. Bootstrap CIs, FDR correction, Cohen's d, stability testing, baseline comparisons.

Code and all results: github.com/AayushBaniya2006/TST-Mechanistic-Interpretability

Built on Matiss Kalnare's patching framework.

#AISafety #MechanisticInterpretability #MachineLearning #BlueDotImpact
