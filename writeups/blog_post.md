Most "Important" Attention Heads in Time Series Transformers Are Statistical Noise

By Aayush Baniya


Activation patching on a Time Series Transformer identified 8 out of 24 attention heads as causally important on a 96-class dataset. After FDR correction, zero survived. Every one was a false positive. The heatmaps looked clean the entire time.

The patching pipeline produced confident, interpretable-looking results that were indistinguishable from noise, and there was nothing in the original framework to flag that.


I found Matiss Kalnare's activation patching framework for Time Series Transformers while looking for a project for the BlueDot Impact AI Safety course. Most mech interp work targets LLMs - time series models are barely touched, even though they show up in healthcare, finance, and astronomy.

His framework takes a 3-layer encoder with 8 heads per layer (24 total), trained on UEA archive classification tasks, and patches individual head outputs between correctly and incorrectly classified inputs to measure the probability shift. You take a correctly classified input and a misclassified one, swap one head's output at a time, and see which swaps change the prediction.

What it lacked: confidence intervals, multiple comparison correction, any stability check. The importance scores were point estimates - one number per head, no uncertainty quantification, no control for 24 simultaneous tests.


Here's how the full pipeline works:

[INSERT IMAGE: pipeline_diagram.png]

The top half is the original framework. The bottom half is what I added. About 3,840 lines of code (with the help of Claude Code), 174 tests.

The three main additions:

Bootstrap resampling - instead of trusting one importance score per head, I resampled the data 10,000 times to get 95% confidence intervals. This tells you how much the score would change if you collected slightly different data.

FDR correction - when you test 24 heads at p < 0.05, you'd expect about 1.2 false positives by chance. Benjamini-Hochberg adjusts the significance threshold to control for that.

Stability testing - I added three types of noise to the inputs (Gaussian noise like sensor error, time warping, phase shifting), re-ran all the patching on the noisy inputs, and checked whether the same heads came out on top. If the rankings change when you barely touch the input, those rankings weren't real.


Here are the head importance heatmaps across all three datasets:

[INSERT IMAGE: fig1_head_importance.png]

Left to right: JapaneseVowels, PenDigits, LSST. Stars mark heads that survived FDR correction. JapaneseVowels has deep reds and 21 stars. LSST has near-zero values everywhere and no stars.

The numbers:

JapaneseVowels (9 classes): 21/24 heads significant after FDR. Cohen's d = 3.35. Stability rho = 0.884. Layer 0 heads carried 2-4x more causal weight than Layer 2. Signal large enough that statistical corrections barely changed the picture.

PenDigits (10 classes): 24/24 significant. d = 1.22. Stability rho = 0.894.

LSST (96 classes): 0/24 significant. d = 0.50. Stability rho = 0.484. Head rankings reshuffled under minimal Gaussian noise. The naive analysis reported 8 significant heads. All false positives.


The effect sizes tell the story per-head:

[INSERT IMAGE: fig4_effect_sizes.png]

JapaneseVowels and PenDigits heads sit in the medium-to-large range. LSST heads cluster around zero - most confidence intervals cross the zero line. You can't even tell which direction the effect goes.


Stability is where LSST falls apart most visibly:

[INSERT IMAGE: fig3_stability.png]

The green dashed line is the 0.7 stability threshold. JapaneseVowels and PenDigits stay well above it - the same heads matter regardless of how you perturb the input. LSST is at 0.48, below coin-flip territory. Different "important" heads every time.


Attention weights are near-useless as a proxy for causal importance:

[INSERT IMAGE: fig2_baseline_comparison.png]

Correlation between attention metrics and actual patching effects: rho around 0.2 across all three datasets. Integrated Gradients does better on JapaneseVowels (rho = 0.575) but goes negative on LSST. None of the cheaper methods track the causal signal reliably.


With 96 output classes, probability mass per class is ~1%. Single-head patching produces shifts on the order of 0.1-0.5% - below the noise floor. The method lacks statistical power once the output space is large enough to dilute per-head effects below detectability. Individual head contributions are too small relative to the output entropy for single-head patching to resolve them.

If single-head activation patching breaks down at ~100 output classes, that constrains how far head-level causal claims generalize. Language models with vocabulary sizes in the tens of thousands face the same dilution problem in principle - the reason patching works there is usually because researchers measure logit differences on specific tokens rather than full-distribution probability shifts. The output space is effectively narrowed by the metric, and that methodological choice carries assumptions that should be stated.

The LSST heatmaps looked just as structured as the JapaneseVowels heatmaps. Same color gradients, same visual patterns. A reviewer would not have caught the problem without running the corrections. If mechanistic interpretability is going to underwrite safety claims, the evidence standards need to include uncertainty quantification and robustness checks as defaults. Point estimates and heatmaps are hypotheses, not evidence.

Open questions: where exactly between 10 and 96 classes the ceiling is, whether group-level patching recovers signal on complex tasks, and whether head importance rankings are training-run-dependent or architecture-dependent.


Technical details

Model: 3-layer Transformer encoder, 8 heads/layer, 24 total
Patching: head-level output swap between correct/incorrect pairs
Stats: bootstrap CIs (10k), Benjamini-Hochberg FDR (alpha 0.05), Cohen's d
Stability: Gaussian noise (sigma 0.05-0.20), time warp (0.05-0.20), phase shift (1-3 steps)
Baselines: Integrated Gradients, attention weights
Seed 42, fully reproducible

github.com/AayushBaniya2006/TST-Mechanistic-Interpretability
Original framework: github.com/mathiisk/TST-Mechanistic-Interpretability

Done through BlueDot Impact AI Safety Fundamentals. Matiss built the patching framework, I built the statistical validation and stability analysis.
