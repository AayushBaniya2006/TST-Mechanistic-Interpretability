Statistical Validation of Activation Patching in Time Series Transformers: False Positives, Stability Bounds, and Complexity Ceilings

Crossposted from GitHub (github.com/AayushBaniya2006/TST-Mechanistic-Interpretability). Completed as part of the BlueDot Impact AI Safety Fundamentals course.


TL;DR

I applied rigorous statistical testing to activation patching results on Time Series Transformers. On simple classification tasks (9-10 classes), activation patching reliably identifies causally important attention heads with large effect sizes and high stability. On a complex task (96 classes), every single "significant" head was a false positive — the naive analysis reported 8/24 significant heads, FDR-corrected analysis found 0/24. Attention weight magnitudes are poor predictors of causal importance (rho = 0.2). The method has a complexity ceiling that practitioners should be aware of.


MOTIVATION

Activation patching (Vig et al., 2020; Meng et al., 2022) is a core tool in mechanistic interpretability. The technique swaps internal activations between a "clean" input (correctly classified) and a "corrupt" input (misclassified), measuring how each component's activation causally affects the output.

Most activation patching work targets language models. I wanted to know: does activation patching produce reliable results on Time Series Transformers? And specifically: how do we distinguish genuine causal importance from statistical noise when testing many components simultaneously?

This question matters because time series Transformers are deployed in domains where trust in model internals has real consequences — medical monitoring, financial systems, scientific instruments.


SETUP

Model: 3-layer Transformer encoder, 8 attention heads per layer (24 heads total), trained on multivariate time series classification tasks from the UEA archive.

Datasets:
JapaneseVowels: 9 classes, 12 input dimensions, 29 time steps
PenDigits: 10 classes, 2 input dimensions, 8 time steps
LSST: 96 classes (rare event categories), 6 input dimensions, variable length

Original framework (Matiss Kalnare): Core patching functions, model training, visualization. This provided sweep_heads() which patches each of the 24 heads individually and returns probability shifts.

My extensions (~3,840 lines): Statistical validation, stability testing, baseline comparisons, automation.


METHOD

Statistical Filtering

For each head, I collected patching effects across multiple sample pairs per dataset. To determine significance:

1. Bootstrap confidence intervals (n = 10,000): Resampled the head importance scores to estimate 95% CIs. This tells us how uncertain we are about each head's importance.

2. Benjamini-Hochberg FDR correction (alpha = 0.05): With 24 simultaneous tests, the expected number of false positives at p < 0.05 is 1.2 heads. FDR correction controls the false discovery rate across all heads.

3. Cohen's d effect sizes: Standardized measure of the difference between patching effects (denoise vs. noise direction). Provides magnitude context beyond binary significance.

Stability Testing

I applied three label-preserving perturbations at multiple strengths:
Gaussian noise: sigma in {0.05, 0.10, 0.20} — simulates sensor noise
Time warp: factor in {0.05, 0.10, 0.20} — simulates temporal distortion
Phase shift: shift in {1, 2, 3} steps — simulates alignment errors

For each perturbation, I re-ran the full patching sweep and compared head importance rankings to the unperturbed baseline using:
Spearman rank correlation (rho): Do the same heads stay in the same relative positions?
Top-K Jaccard overlap: Are the top-5 most important heads the same?

Baseline Comparisons

I compared activation patching against two simpler attribution methods:
Integrated Gradients (via Captum): Gradient-based attribution accumulated along a straight-line path from baseline to input
Attention weight analysis: Entropy, max, and variance of attention distributions

Correlation between each baseline method and patching results tells us whether simpler methods capture the same signal.


RESULTS

Significance After FDR Correction

JapaneseVowels: Naive ~24/24, FDR-corrected 21/24, effect size d = 3.35 (Large)
PenDigits: Naive ~24/24, FDR-corrected 24/24, effect size d = 1.22 (Large)
LSST: Naive ~8/24, FDR-corrected 0/24, effect size d = 0.50 (Small)

The LSST result is the most striking. The uncorrected analysis suggested one-third of heads had meaningful effects. After controlling for multiple comparisons, none survived. The per-head effect sizes for LSST were "negligible" (|d| < 0.25) for 19 out of 24 heads.

Stability

JapaneseVowels: Mean rho = 0.884 [0.834, 0.915], mean top-5 overlap = 0.77. Verdict: Stable.
PenDigits: Mean rho = 0.894 [0.775, 0.953], mean top-5 overlap = 0.73. Verdict: Stable.
LSST: Mean rho = 0.484 [0.450, 0.509], mean top-5 overlap = 0.42. Verdict: Unstable.

For JapaneseVowels and PenDigits, perturbing the input barely changes which heads are ranked as important. For LSST, the rankings are essentially random — consistent with the significance results showing no real signal.

Baseline Correlations

Integrated Gradients: JapaneseVowels rho = 0.575, PenDigits rho = 0.372, LSST rho = -0.122
Attention entropy: JapaneseVowels rho = 0.207, PenDigits rho = 0.269, LSST rho = 0.184
Attention max: JapaneseVowels rho = 0.483, PenDigits rho = 0.282, LSST rho = 0.170
Attention variance: JapaneseVowels rho = 0.217, PenDigits rho = 0.272, LSST rho = 0.181

Integrated Gradients captures some of the same signal as activation patching (moderate correlation on simpler tasks) but fails on LSST. Raw attention metrics are poor predictors across the board (rho = 0.2). This confirms the "attention is not explanation" finding in a time series context.

Layer-wise Patterns

On datasets where the method works (JapaneseVowels, PenDigits), Layer 0 heads are consistently 2-4x more important than Layer 2 heads. This suggests early layers perform more of the causal work in these classification tasks — potentially handling feature extraction while later layers refine representations.


DISCUSSION

Why Does LSST Fail?

LSST has 96 classes. When probability mass is distributed across 96 categories, swapping a single attention head's output produces tiny shifts that are indistinguishable from noise. The effect size is below the detection threshold of single-head patching.

This suggests activation patching has a complexity ceiling — it works when individual heads carry enough causal weight to produce detectable probability shifts, which requires tasks where each head's contribution meaningfully moves probability mass. With 96 classes, no single head moves enough mass.

Implications for Interpretability Practice

1. Always correct for multiple comparisons. Testing N components means expecting N x alpha false positives. This is basic statistics but is routinely ignored in interpretability papers.

2. Report effect sizes, not just p-values. LSST's per-head effects were "negligible" — knowing this is more useful than knowing they failed an FDR threshold.

3. Test stability. A finding that disappears under small perturbations was never robust. Stability testing should be standard.

4. Don't assume attention = importance. The rho = 0.2 correlation is damning. Interpretability methods based solely on attention patterns will miss most of the causal structure.

5. Be honest about limitations. Activation patching is a powerful tool, but it has a complexity ceiling. Claiming otherwise would be dishonest.

Limitations of This Work

3 datasets: Cannot draw strong conclusions about the exact complexity ceiling. More datasets between 10 and 96 classes would help.
Single training seed: Results may be specific to the trained model's weight configuration.
Head-level granularity only: Position-level patching (which heads matter at which time steps) was explored but not statistically validated at the same depth.


REPRODUCIBILITY

All code, data, and results are available at github.com/AayushBaniya2006/TST-Mechanistic-Interpretability.

Random seed: 42
Bootstrap iterations: 10,000
FDR alpha: 0.05
174 automated tests
Full pipeline runnable with: python Scripts/run_complete_analysis.py


ACKNOWLEDGMENTS

The original activation patching framework for Time Series Transformers was built by Matiss Kalnare. I built the statistical validation, stability testing, baseline comparisons, and automation pipeline. This project was completed as part of the BlueDot Impact AI Safety Fundamentals course.
