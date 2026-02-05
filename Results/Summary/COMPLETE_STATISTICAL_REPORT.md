# Complete Statistical Analysis Report

## Executive Summary

**Total pairs analyzed:** 162
**Datasets:** JapaneseVowels, PenDigits, LSST
**Bootstrap iterations:** 10000
**Random seed:** 42

## Key Findings

### JapaneseVowels
- Mean ΔP: 0.0865 [95% CI: 0.0738, 0.1022]
- Significant heads (FDR-corrected): 21/24

### PenDigits
- Mean ΔP: 0.0188 [95% CI: 0.0167, 0.0210]
- Significant heads (FDR-corrected): 24/24

### LSST
- Mean ΔP: -0.0018 [95% CI: -0.0051, 0.0003]
- Significant heads (FDR-corrected): 0/24

## Stability Analysis

### JapaneseVowels
- Mean rank correlation: 0.884 [0.834, 0.915]
- Verdict: Stable

### PenDigits
- Mean rank correlation: 0.894 [0.775, 0.953]
- Verdict: Stable

### LSST
- Mean rank correlation: 0.484 [0.450, 0.509]
- Verdict: Unstable

## Baseline Method Comparisons

| Dataset | Method | ρ | 95% CI |
|---------|--------|---|--------|
| JapaneseVowels | integrated_gradients | 0.575 | [0.461, 0.676] |
| PenDigits | integrated_gradients | 0.372 | [0.299, 0.451] |
| LSST | integrated_gradients | -0.122 | [-0.237, 0.011] |
| JapaneseVowels | attention_entropy | 0.207 | [0.099, 0.283] |
| JapaneseVowels | attention_max | 0.483 | [0.388, 0.574] |
| JapaneseVowels | attention_variance | 0.217 | [0.111, 0.300] |
| PenDigits | attention_entropy | 0.269 | [0.149, 0.360] |
| PenDigits | attention_max | 0.282 | [0.178, 0.369] |
| PenDigits | attention_variance | 0.272 | [0.160, 0.357] |
| LSST | attention_entropy | 0.184 | [0.057, 0.310] |
| LSST | attention_max | 0.170 | [0.043, 0.290] |
| LSST | attention_variance | 0.181 | [0.049, 0.303] |

## Statistical Limitations

1. **Sample size (n=3 datasets)**: Cannot establish cross-dataset correlations
2. **Single training run**: Results may be specific to training seeds
3. **Multiple comparisons**: FDR correction applied but some false positives possible

## Files Generated

### Data Files
- `data/confidence_intervals.csv`
- `data/fdr_corrected_pvalues.csv`
- `data/effect_sizes.csv`
- `data/baseline_comparisons.csv`
- `data/stability_with_statistics.csv`
- `data/null_distribution.csv`

### Figures
- `figures/fig1_head_importance.pdf`
- `figures/fig2_baseline_comparison.pdf`
- `figures/fig3_stability.pdf`
- `figures/fig4_effect_sizes.pdf`
- `figures/fig5_null_distribution.pdf`

### Tables
- `tables/table1_summary.csv`
- `tables/table2_significant_heads.csv`
- `tables/table3_method_comparison.csv`
- `tables/table4_stability.csv`
- `tables/table5_power_analysis.csv`
