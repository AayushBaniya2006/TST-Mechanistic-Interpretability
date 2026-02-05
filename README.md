# Transformer Time Series Interpretability Toolkit

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-174%20passed-brightgreen.svg)

Mechanistic interpretability for Transformer-based time series classifiers via activation patching.

---

## Key Findings

| Dataset | Significant Heads | Effect Size | Stability | Verdict |
|---------|-------------------|-------------|-----------|---------|
| **JapaneseVowels** | 21/24 (87.5%) | d = 3.35 (Large) | ρ = 0.884 | Reliable |
| **PenDigits** | 24/24 (100%) | d = 1.22 (Large) | ρ = 0.894 | Reliable |
| **LSST** | 0/24 (0%) | d = 0.50 (Small) | ρ = 0.484 | Unreliable |

**Main conclusions:**
1. Activation patching identifies causally important attention heads in time series Transformers
2. Layer 0 heads are 2-4x more important than Layer 2 heads
3. Findings are stable under input perturbations (for simpler datasets)
4. Attention weights poorly predict causal importance (ρ ~ 0.2)
5. Method fails on complex datasets (96 classes)

---

## Project Background

### Original Work (Matiss Kalnare)

The core activation patching framework:

- **Core Patching Functions**: `sweep_heads()`, `sweep_layerwise_patch()`, `sweep_attention_head_positions()`
- **TST Model Architecture**: 3-layer Transformer encoder with 8 attention heads per layer
- **Visualization**: Heatmaps, causal graphs, attention overlays
- **Interactive Exploration**: Jupyter notebook with widgets and sliders
- **Initial Results**: JapaneseVowels and PenDigits patching experiments

### My Extensions (Aayush Baniya)

I added rigorous statistical validation and stability testing to transform exploratory results into publishable findings.

#### New Modules (~3,840 lines of code)

| Module | Lines | Purpose |
|--------|-------|---------|
| `statistics.py` | 913 | Bootstrap CIs (10k iterations), FDR correction, Cohen's d effect sizes |
| `baselines.py` | 876 | Integrated Gradients, attention weight analysis, random patching baseline |
| `stability_metrics.py` | 370 | Spearman rank correlation, top-K overlap, mechanism stability scores |
| `metrics.py` | 628 | Logit difference, KL divergence, cross-entropy change |
| `perturbations.py` | 247 | Gaussian noise, time warp, phase shift (label-preserving) |
| `config.py` | 478 | Seed management, experiment configuration, reproducibility |

#### What I Built

| Contribution | Description |
|--------------|-------------|
| **Statistical Filtering** | Separates real findings from false positives using bootstrap CIs and FDR correction |
| **Stability Testing** | Validates that discovered mechanisms are robust under realistic input perturbations |
| **Baseline Comparisons** | Proves activation patching outperforms simpler methods (attention weights, Integrated Gradients) |
| **LSST Analysis** | 74 new sample pairs revealing method limitations on complex datasets |
| **Test Suite** | 174 automated tests across 9 files ensuring code correctness |
| **Automation Scripts** | Full statistical pipeline executable with one command |

#### Key Findings from My Analysis

| Finding | Before My Analysis | After My Analysis |
|---------|-------------------|-------------------|
| LSST significant heads | 8/24 "significant" | **0/24** (all were false positives) |
| Confidence in results | Point estimates only | 95% CIs with uncertainty quantification |
| Stability validation | None | ρ > 0.88 for JV/PD, ρ = 0.48 for LSST |
| Baseline comparison | None | Attention ≠ causation (ρ ~ 0.2) |

---

## Installation

```bash
git clone https://github.com/mathiisk/TSTpatching.git
cd TSTpatching
pip install -e ".[dev]"
```

Verify installation:
```bash
python -c "from Utilities import sweep_heads; print('OK')"
pytest tests/ -v
```

## Quick Start

```python
import torch
from Utilities import (
    TimeSeriesTransformer, load_dataset, sweep_heads, get_probs,
    get_head_importance, plot_influence, compute_confidence_interval,
    apply_fdr_correction
)

# Load pre-trained model
train_loader, test_loader = load_dataset("JapaneseVowels")
model = TimeSeriesTransformer(input_dim=12, num_classes=9, seq_len=29)
model.load_state_dict(torch.load("TST_models/TST_japanesevowels.pth"))

# Get a sample pair (clean prediction vs misclassified)
clean_x = ...  # shape: (1, seq_len, input_dim)
corrupt_x = ...

# Run activation patching across all heads
patch_probs = sweep_heads(model, clean_x, corrupt_x, num_classes=9)
baseline = get_probs(model, corrupt_x)

# Analyze which heads matter
importance = get_head_importance(patch_probs, baseline, true_label=0)
plot_influence(patch_probs, baseline, true_label=0)

# Add statistical rigor (Aayush's contribution)
ci = compute_confidence_interval(importance, confidence=0.95, n_bootstrap=10000)
print(f"Mean importance: {ci.mean:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]")
```

---

## Repository Structure

```
Utilities/                 Core library
├── utils.py               Activation patching, sweeps, visualization (Matiss)
├── TST_trainer.py         Model architecture, training (Matiss)
├── statistics.py          Bootstrap CIs, FDR correction, effect sizes (Aayush)
├── baselines.py           Integrated Gradients, attention baselines (Aayush)
├── stability_metrics.py   Rank correlation, top-K overlap (Aayush)
├── metrics.py             Logit diff, KL divergence, cross-entropy (Aayush)
├── perturbations.py       Gaussian noise, time warp, phase shift (Aayush)
└── config.py              Seed management, experiment configs (Aayush)

Scripts/                   Automation (Aayush)
├── run_stability_experiments.py   Stability stress-testing pipeline
└── run_complete_analysis.py       Full statistical pipeline

Notebooks/
├── Patching.ipynb                 Interactive exploration (Matiss)
├── Patching_Stability.ipynb       Stability validation (Aayush)
├── Statistical_Reanalysis.ipynb   Statistical analysis (Aayush)
└── SAE.ipynb                      Sparse autoencoder (Matiss)

tests/                     Test suite - 174 tests (Aayush)
TST_models/                Pre-trained weights
Results/                   Experimental outputs
└── Summary/               Publication-ready tables & figures
```

---

## Core API

### Patching (Original - Matiss)

| Function | Description |
|----------|-------------|
| `sweep_heads(model, clean, corrupt, num_classes)` | Patch each attention head individually |
| `sweep_layerwise_patch(model, clean, corrupt, num_classes)` | Patch entire layers |
| `sweep_attention_head_positions(...)` | Patch at (layer, head, position) granularity |
| `find_critical_patches(patch_probs, baseline, label, threshold)` | Find important head-position pairs |
| `build_causal_graph(critical_patches)` | Build influence graph |

### Statistics (Extension - Aayush)

| Function | Description |
|----------|-------------|
| `compute_confidence_interval(data, confidence, n_bootstrap)` | Bootstrap CI with 10,000 iterations |
| `compute_effect_size(treatment, control)` | Cohen's d with CI |
| `apply_fdr_correction(p_values, method)` | Benjamini-Hochberg correction |

### Stability (Extension - Aayush)

| Function | Description |
|----------|-------------|
| `gaussian_noise(X, sigma)` | Add scaled Gaussian noise |
| `time_warp(X, factor)` | Local time stretching |
| `phase_shift(X, shift)` | Circular time shift |
| `head_rank_correlation(baseline, perturbed)` | Spearman ρ of head rankings |
| `topk_overlap(baseline, perturbed, k)` | Jaccard overlap of top-K heads |

### Baselines (Extension - Aayush)

| Function | Description |
|----------|-------------|
| `integrated_gradients_importance(model, x, target)` | IG attribution via Captum |
| `attention_weight_importance(model, x)` | Attention entropy/max/variance |
| `random_patching_baseline(model, x, n_samples)` | Null distribution |

---

## Running Experiments

```bash
# Run stability experiments on all datasets
python Scripts/run_stability_experiments.py --dataset all

# Run full statistical analysis (generates figures and tables)
python Scripts/run_complete_analysis.py

# Run tests
pytest tests/ -v
```

---

## Results

All results are in `Results/Summary/`:

| File | Contents |
|------|----------|
| `COMPLETE_STATISTICAL_REPORT.md` | Executive summary of findings |
| `data/confidence_intervals.csv` | 95% CIs for all 72 heads |
| `data/fdr_corrected_pvalues.csv` | Multiple comparison correction |
| `data/effect_sizes.csv` | Cohen's d with CIs |
| `data/baseline_comparisons.csv` | Patching vs IG vs attention |
| `data/stability_with_statistics.csv` | Perturbation results |
| `figures/fig1-5.pdf` | Publication-ready figures |
| `tables/table1-5.md` | Statistical tables |

---

## Reproducibility

All experiments use:
- **Random seed**: 42
- **Bootstrap iterations**: 10,000
- **FDR alpha**: 0.05
- **Perturbation levels**: σ = {0.05, 0.10, 0.20}, factor = {0.05, 0.10, 0.20}, shift = {1, 2, 3}

To reproduce:
```bash
python Scripts/run_complete_analysis.py
```

---

## Citation

```bibtex
@misc{tst-mechanistic-interp,
  author = {Kalnare, Matiss and Baniya, Aayush},
  title = {Mechanistic Interpretability for Time Series Transformers},
  year = {2025},
  url = {https://github.com/mathiisk/TSTpatching}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
