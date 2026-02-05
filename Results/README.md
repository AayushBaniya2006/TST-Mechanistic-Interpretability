# Results Directory

This directory contains all experimental outputs from activation patching analysis.

## Directory Structure

```
Results/
├── Summary/                    Publication-ready outputs (616 KB)
├── JapaneseVowels/            18 sample pair results
├── PenDigits/                 70 sample pair results
├── LSST/                      74 sample pair results (56 MB)
└── Stability/                 Perturbation experiments (79 MB)
```

---

## Summary/ (Publication Materials)

### Executive Summary
- `COMPLETE_STATISTICAL_REPORT.md` - Key findings and methodology

### Data Files (CSV)
| File | Description | Columns |
|------|-------------|---------|
| `all_pairs_summary.csv` | Results for all 162 sample pairs | dataset, pair_id, delta_p, significant |
| `dataset_overview.csv` | Per-dataset statistics | dataset, n_pairs, mean_delta_p, significant_heads |
| `confidence_intervals.csv` | 95% CIs for all heads | dataset, layer, head, mean, ci_lower, ci_upper |
| `fdr_corrected_pvalues.csv` | Multiple comparison correction | dataset, layer, head, p_raw, p_fdr, significant |
| `effect_sizes.csv` | Cohen's d with CIs | dataset, layer, head, cohens_d, ci_lower, ci_upper |
| `baseline_comparisons.csv` | Method correlations | dataset, method, spearman_rho, ci_lower, ci_upper |
| `stability_with_statistics.csv` | Perturbation results | dataset, perturbation, mean_rho, topk_overlap |

### Figures (PDF + PNG)
| File | Description |
|------|-------------|
| `fig1_head_importance.pdf` | Heatmap of ΔP per head across datasets |
| `fig2_baseline_comparison.pdf` | Scatter: patching vs IG/attention |
| `fig3_stability.pdf` | Rank correlation under perturbations |
| `fig4_effect_sizes.pdf` | Forest plot of Cohen's d with CIs |
| `fig5_null_distribution.pdf` | Random patching baseline distribution |

### Tables (Markdown)
| File | Description |
|------|-------------|
| `table1_summary.md` | Dataset overview and key metrics |
| `table2_significant_heads.md` | Ranked list of significant heads |
| `table3_method_comparison.md` | Patching vs baselines correlation |
| `table4_stability.md` | Stability metrics per dataset |
| `table5_power_analysis.md` | Statistical power calculations |

---

## Dataset Results (JapaneseVowels/, PenDigits/, LSST/)

### Structure
```
{Dataset}/
├── denoise/
│   └── class_{N}/
│       ├── aggregated_deltas.npz       Class-level averages
│       ├── class_avg_head_heatmap.png  Visualization
│       └── pair_{clean}_{corrupt}/
│           ├── raw_results.npz         All patching arrays
│           ├── summary.csv             Key metrics
│           ├── patch_each_head_heatmap.png
│           └── causal_graph.png
└── noise/
    └── [same structure]
```

### File Formats

#### raw_results.npz
```python
import numpy as np
data = np.load('raw_results.npz')

# Available keys:
data['baseline']        # (num_classes,) - baseline predictions
data['head_patch']      # (n_layers, n_heads, num_classes) - after patching each head
data['layer_patch']     # (n_layers, num_classes) - after patching each layer
data['head_pos_patch']  # (n_layers, n_heads, seq_len, num_classes) - position-level
data['mlp_patch']       # (n_layers, num_classes) - MLP patching
data['delta_p']         # (n_layers, n_heads) - probability change for true class
```

#### summary.csv
| Column | Description |
|--------|-------------|
| `clean_idx` | Index of correctly classified sample |
| `corrupt_idx` | Index of misclassified sample |
| `true_label` | Ground truth class |
| `baseline_prob` | P(true_label) before patching |
| `max_delta_p` | Maximum probability improvement |
| `best_head` | Head with largest effect (e.g., "L0H3") |

---

## Stability/ (Perturbation Results)

### Structure
```
Stability/
└── {Dataset}/
    └── class_{N}/
        └── pair_{clean}_{corrupt}/
            ├── baseline/
            │   ├── accuracy.txt
            │   └── head_importance_ranking.csv
            ├── gaussian/
            │   ├── sigma_0.05/
            │   ├── sigma_0.10/
            │   └── sigma_0.20/
            ├── time_warp/
            │   ├── factor_0.05/
            │   ├── factor_0.10/
            │   └── factor_0.20/
            └── phase_shift/
                ├── shift_1/
                ├── shift_2/
                └── shift_3/
```

### Perturbation Subdirectory Contents
```
{perturbation}/{parameter}/
├── raw_results.npz              Patching on perturbed input
├── stability_metrics.json       Stability scores
└── head_comparison_heatmap.png  Before/after visualization
```

### stability_metrics.json
```json
{
  "rank_correlation": 0.912,
  "stability_score": 0.95,
  "topk_overlap_k3": 1.0,
  "topk_overlap_k5": 0.8,
  "topk_overlap_k10": 0.7,
  "valid": true,
  "orig_prob": 0.95,
  "pert_prob": 0.92,
  "recovery_delta": {
    "mean_abs_diff": 0.02,
    "max_abs_diff": 0.08,
    "std_diff": 0.015
  }
}
```

---

## Loading Results

### Python Example
```python
import numpy as np
import pandas as pd

# Load summary statistics
ci_df = pd.read_csv('Results/Summary/data/confidence_intervals.csv')
significant_heads = ci_df[ci_df['significant'] == True]

# Load raw patching results
results = np.load('Results/JapaneseVowels/denoise/class_1/pair_5_1/raw_results.npz')
delta_p = results['delta_p']  # Shape: (3, 8) for 3 layers × 8 heads

# Load stability metrics
import json
with open('Results/Stability/JapaneseVowels/class_1/pair_5_1/gaussian/sigma_0.10/stability_metrics.json') as f:
    stability = json.load(f)
print(f"Rank correlation: {stability['rank_correlation']:.3f}")
```

---

## Regenerating Results

```bash
# Full statistical analysis (2.5-3.5 hours)
python Scripts/run_complete_analysis.py

# Stability experiments only
python Scripts/run_stability_experiments.py --dataset all
```

All experiments use seed=42 for reproducibility.
