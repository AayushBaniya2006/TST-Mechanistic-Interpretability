# Mechanistic Stability Analysis: Complete Findings

## Executive Summary

We tested whether the internal components identified as "important" by activation patching remain consistent when inputs are slightly modified. Results vary dramatically by dataset.

## Key Results

### JapaneseVowels (12-dim, 25 timesteps, 9 classes)
- **High stability**: Rank correlation 0.78-0.95
- **Perturbations that work**: Gaussian noise, time warp
- **Perturbations that fail**: Phase shift (breaks accuracy)
- **Interpretation**: Mechanisms are relatively robust to noise and local time distortions

### PenDigits (2-dim, 8 timesteps, 10 classes)  
- **High stability**: Rank correlation 0.69-0.99
- **Perturbations that work**: Gaussian noise, time warp
- **Perturbations that fail**: Phase shift (breaks accuracy severely)
- **Interpretation**: Short sequences with low dimensionality show stable mechanisms

### LSST (6-dim, 36 timesteps, 14 classes)
- **Low stability**: Rank correlation 0.42-0.53
- **Perturbations that work**: Time warp, phase shift
- **Perturbations that fail**: Gaussian noise (breaks accuracy)
- **Interpretation**: Mechanisms are UNSTABLE - different heads become important under perturbation

## Main Finding

**Mechanism stability is dataset-dependent.** 

- JapaneseVowels and PenDigits show stable mechanisms (rank ρ > 0.7)
- LSST shows unstable mechanisms (rank ρ ≈ 0.5)

This suggests that interpretability explanations from activation patching may be more reliable for some time series types than others.

## Perturbation Sensitivity

| Dataset | Gaussian | Time Warp | Phase Shift |
|---------|----------|-----------|-------------|
| JapaneseVowels | Works | Works | Breaks accuracy |
| PenDigits | Works | Works | Breaks accuracy |
| LSST | Breaks accuracy | Works | Works |

Different datasets are sensitive to different perturbation types, which affects what stability tests are valid.

## Conclusions (Using Allowed Statements Only)

1. "Mechanistic explanations appear **stable** under Gaussian noise and time warp for JapaneseVowels and PenDigits"

2. "Mechanistic explanations appear **unstable** under time warp and phase shift for LSST"

3. "Rank correlation of 0.42-0.53 for LSST suggests **low consistency** in which heads are identified as important"

4. "Top-5 overlap of 0.38-0.44 for LSST indicates **limited mechanism preservation** under perturbation"

5. "Accuracy-mechanism decoupling observed in LSST: accuracy stable while mechanisms shift"

## Files Generated

- `Results/Stability/JapaneseVowels/summary_table.md`
- `Results/Stability/PenDigits/summary_table.md`
- `Results/Stability/LSST/summary_table.md`
- `Results/Stability/Summary/aggregate_results.csv`
- `Results/Stability/Summary/findings.md`

## Methodology Notes

- 10 sample pairs per dataset
- Random seed: 42
- Perturbations validated to preserve accuracy within 5%
- Metrics: Spearman rank correlation, Jaccard top-K overlap, composite stability score
