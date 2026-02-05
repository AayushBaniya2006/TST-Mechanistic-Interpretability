# Stability Experiment Findings Summary

## Overview

- **Total experiments**: 26
- **Datasets analyzed**: 3
- **Perturbation types**: 3

## Key Findings

### Most Stable Configuration
- Dataset: JapaneseVowels
- Perturbation: gaussian_sigma_0.05
- Stability Score: 0.999

### Least Stable Configuration
- Dataset: PenDigits
- Perturbation: phase_shift_max_2
- Stability Score: 0.791

## Per-Dataset Summary

### JapaneseVowels
- Mean rank correlation: 0.934
- Mean stability score: 0.921
- Valid experiments: 77

### PenDigits
- Mean rank correlation: 0.935
- Mean stability score: 0.928
- Valid experiments: 238

### LSST
- Mean rank correlation: 0.894
- Mean stability score: 0.886
- Valid experiments: 321

## Aggregate Metrics Table

| Perturbation | Rank ρ | Top-3 | Top-5 | Top-10 | Stability |
|--------------|--------|-------|-------|--------|-----------|
| gaussian_sigma_0.05 | 0.969 | 0.878 | 0.875 | 0.920 | 0.955 |
| gaussian_sigma_0.10 | 0.943 | 0.849 | 0.828 | 0.869 | 0.934 |
| gaussian_sigma_0.20 | 0.902 | 0.741 | 0.773 | 0.824 | 0.907 |
| time_warp_factor_0.05 | 0.913 | 0.709 | 0.716 | 0.780 | 0.892 |
| time_warp_factor_0.10 | 0.901 | 0.671 | 0.671 | 0.783 | 0.876 |
| time_warp_factor_0.20 | 0.864 | 0.588 | 0.664 | 0.741 | 0.864 |
| phase_shift_max_1 | 0.864 | 0.611 | 0.644 | 0.750 | 0.858 |
| phase_shift_max_2 | 0.865 | 0.641 | 0.645 | 0.759 | 0.859 |
| phase_shift_max_3 | 0.821 | 0.505 | 0.575 | 0.709 | 0.827 |

---
*Generated automatically by run_stability_experiments.py*