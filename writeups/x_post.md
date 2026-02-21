Most "Important" Attention Heads in Time Series Transformers Are Statistical Noise

Aayush Baniya | BlueDot Impact AI Safety Fundamentals


Activation patching on a Time Series Transformer identified 8 out of 24 attention heads as causally important on a dataset with 96 output classes. After FDR correction, zero survived. Every one was a false positive. The heatmaps looked clean the entire time.

On simpler tasks (9-10 classes), the method works — 21-24 heads survive statistical correction with large effect sizes and stable rankings under perturbation. On complex tasks, it collapses. Without statistical validation, you can't tell the difference.


Background

I found Matiss Kalnare's activation patching framework for Time Series Transformers while looking for a project for the BlueDot Impact AI Safety course. Most mech interp work targets LLMs — time series models are barely touched, even though they show up in healthcare, finance, and astronomy.

His framework takes a 3-layer Transformer encoder with 8 heads per layer (24 total), trained on UEA archive classification tasks, and patches individual head outputs between correctly and incorrectly classified inputs to measure the probability shift. You take a correctly classified input and a misclassified one, swap one head's output at a time, and see which swaps change the prediction.

What it lacked: confidence intervals, multiple comparison correction, any stability check. The importance scores were point estimates — one number per head, no uncertainty quantification, no control for 24 simultaneous tests.


What I Built

https://raw.githubusercontent.com/AayushBaniya2006/TST-Mechanistic-Interpretability/main/Results/Summary/figures/pipeline_diagram.png

About 3,840 lines of code, 174 automated tests. Three main additions:

Bootstrap resampling — Instead of trusting one importance score per head, I resampled the data 10,000 times to get 95% confidence intervals. This tells you how much the score would change if you collected slightly different data.

FDR correction — When you test 24 heads at once, you'd expect about 1.2 false positives by chance. Benjamini-Hochberg adjusts the significance threshold to control for that.

Stability testing — I added three types of noise to the inputs (Gaussian noise like sensor error, time warping, phase shifting), re-ran all the patching on the noisy inputs, and checked whether the same heads came out on top. If the rankings change when you barely touch the input, those rankings weren't real.


Results

https://raw.githubusercontent.com/AayushBaniya2006/TST-Mechanistic-Interpretability/main/Results/Summary/figures/fig1_head_importance.png

JapaneseVowels (9 classes) — 21/24 heads significant after FDR correction. Large effect size (Cohen's d = 3.35). Stability correlation of 0.884. Layer 0 heads carried roughly 4x more causal weight than Layer 2. Signal large enough that statistical corrections barely changed the picture.

PenDigits (10 classes) — 24/24 significant. Large effect size (d = 1.22). Stability correlation of 0.894. Every head matters and the rankings hold under perturbation.

LSST (96-way output: 14 astronomical object types, label indices spanning 0-95) — 0/24 significant. Small effect size (d = 0.50). Stability correlation of 0.484. Head rankings reshuffled under minimal noise. The naive analysis flagged 8 heads. After FDR correction, all were false positives.

https://raw.githubusercontent.com/AayushBaniya2006/TST-Mechanistic-Interpretability/main/Results/Summary/figures/fig4_effect_sizes.png

The effect sizes tell the story clearly. JapaneseVowels and PenDigits heads sit in the medium-to-large range. LSST heads cluster around zero — most confidence intervals cross the zero line. You can't even tell which direction the effect goes.

https://raw.githubusercontent.com/AayushBaniya2006/TST-Mechanistic-Interpretability/main/Results/Summary/figures/fig3_stability.png

For stability: JapaneseVowels and PenDigits stay well above the 0.7 threshold — the same heads matter regardless of how you perturb the input. LSST is at 0.48, below coin-flip territory. Different "important" heads every time.

https://raw.githubusercontent.com/AayushBaniya2006/TST-Mechanistic-Interpretability/main/Results/Summary/figures/fig2_baseline_comparison.png

I also compared activation patching against simpler attribution methods. Attention entropy and variance correlate poorly with patching effects (correlation around 0.2 across all datasets). Maximum attention weight does better on simpler tasks (0.48 on JapaneseVowels) but not on LSST. Integrated Gradients captures moderate signal on JapaneseVowels (0.575) but goes negative on LSST. Simpler methods don't reliably track causal importance.


Why LSST Fails

LSST has 14 distinct astronomical object types, but because the label indices are sparse (ranging from 0 to 95), the model uses a 96-way softmax. Probability mass is distributed across 96 output slots, meaning each class gets roughly 1% of the mass. Single-head patching produces shifts that are below the noise floor at that scale.

The method lacks statistical power once the output space dilutes per-head effects below detectability. This is a complexity-dependent ceiling on head-level causal attribution.


What This Means for Interpretability

If single-head activation patching breaks down with large output spaces, that constrains how far head-level causal claims generalize. Language models with vocabulary sizes in the tens of thousands face a similar dilution problem in principle — one hypothesis for why patching works in that setting is that researchers typically measure logit differences on specific tokens rather than full-distribution probability shifts. The output space is effectively narrowed by the metric choice, and that carries assumptions worth stating explicitly.

The LSST heatmaps looked just as structured as the JapaneseVowels heatmaps. Same color gradients, same visual patterns. A reviewer would not have caught the problem without running the corrections. If mechanistic interpretability is going to underwrite safety claims, the evidence standards need to include uncertainty quantification and robustness checks as defaults. Point estimates and heatmaps are hypotheses, not evidence.

Four takeaways:

1. Always correct for multiple comparisons. Testing N components means expecting false positives by chance. This is basic statistics but is routinely ignored in interpretability papers.

2. Report effect sizes, not just p-values. LSST's per-head effects were negligible — knowing this is more useful than knowing they failed a significance threshold.

3. Test stability. A finding that disappears under small perturbations was never robust. Stability testing should be standard.

4. Don't assume attention equals importance. Attention entropy and variance show correlations around 0.2 against causal importance. Even maximum attention weight only reaches moderate correlation on simpler tasks.


Limitations

Only 3 datasets — can't pin down the exact complexity ceiling. More datasets between 10 and 96 output classes would help. Single training seed, so results may be specific to this model's weight configuration. Position-level patching was explored but not statistically validated at the same depth.


Open Questions

Where exactly between 10 and 96 output classes the complexity ceiling sits. Whether group-level patching (multiple heads at once) recovers signal on complex tasks. Whether head importance rankings change across training seeds. How this extends to other Transformer architectures.


All code, data, and results are open-source: github.com/AayushBaniya2006/TST-Mechanistic-Interpretability

Seed 42, 10,000 bootstrap resamples, FDR alpha = 0.05. Fully reproducible.

This project was completed through BlueDot Impact AI Safety Fundamentals. The original activation patching framework was built by Matiss Kalnare. I built the statistical validation, stability testing, baseline comparisons, and automation pipeline.
