Tweet 1 (Hook - attach fig1_head_importance.png)

I added statistical testing to activation patching on time series Transformers.

8 out of 24 "important" attention heads turned out to be false positives. On one dataset, all of them were.

The heatmaps looked clean the whole time. Thread:


Tweet 2 (attach pipeline_diagram.png)

The setup: a 3-layer Transformer (24 attention heads), trained on time series classification. Swap each head's output between a correct and incorrect input. Measure the probability shift.

I added bootstrap CIs (10k resamples), FDR correction, stability testing under noise, and baseline comparisons. ~3,840 lines.


Tweet 3 (attach fig3_stability.png)

Simple tasks (9-10 classes): method works.

JapaneseVowels: 21/24 heads significant, d=3.35, stability rho=0.884
PenDigits: 24/24 significant, d=1.22, rho=0.894

Head rankings stay consistent under input perturbations. Real signal.


Tweet 4 (attach fig4_effect_sizes.png)

LSST (96 classes): 0/24 significant after FDR correction. The naive analysis said 8. All false positives.

Effect sizes cluster around zero. Confidence intervals cross the zero line for almost every head. No detectable signal.


Tweet 5

96 classes means ~1% probability mass per class. Single-head patching produces 0.1-0.5% shifts - below the noise floor. The method lacks power once the output space dilutes per-head effects below detectability.

This is a complexity ceiling on single-head causal attribution.


Tweet 6 (attach fig2_baseline_comparison.png)

Attention weights predict almost nothing about causal importance. Correlation with patching effects: rho ~ 0.2 everywhere.

Where the model looks is not what the model uses. Known for LLMs, now confirmed for time series.


Tweet 7

For AI safety: if interpretability methods give confident results that are wrong, the evidence standards need to include uncertainty quantification and stability checks as defaults.

Code + all results: github.com/AayushBaniya2006/TST-Mechanistic-Interpretability

Built on Matiss Kalnare's framework. Done through @BlueDotImpact.
