#!/usr/bin/env python3
"""
Complete Statistical Analysis Script for TST Mechanistic Interpretability

This script runs ALL statistical analyses needed for a publication-ready paper:
1. Loads all 162 pairs from existing results
2. Computes 95% CIs for all metrics
3. Applies FDR correction for multiple comparisons
4. Computes effect sizes (Cohen's d)
5. Runs baseline comparisons (Integrated Gradients, Attention weights)
6. Generates null distribution via random patching
7. Creates all publication figures
8. Exports all tables
9. Generates final statistical report

Estimated runtime: 2.5-3.5 hours

Usage:
    python Scripts/run_complete_analysis.py
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import spearmanr, ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import json

# Project imports
from Utilities.statistics import (
    compute_confidence_interval,
    compute_effect_size,
    apply_fdr_correction,
    significance_test,
    power_analysis,
    compute_correlation_ci,
    compute_aggregate_statistics
)
from Utilities.baselines import (
    random_patching_baseline,
    integrated_gradients_importance,
    attention_weight_importance,
    compare_all_methods,
    compute_empirical_p_value
)
from Utilities.stability_metrics import (
    head_rank_correlation,
    topk_overlap,
    mechanism_stability_score,
    get_head_importance
)
from Utilities.TST_trainer import TimeSeriesTransformer, load_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
N_BOOTSTRAP = 10000  # For confidence intervals
N_PERMUTATIONS = 100  # For null distribution per pair
SAMPLE_SIZE_BASELINES = 20  # Pairs per dataset for baseline comparisons

DATASET_CONFIGS = {
    'JapaneseVowels': {'input_dim': 12, 'seq_len': 25, 'num_classes': 10},
    'PenDigits': {'input_dim': 2, 'seq_len': 8, 'num_classes': 10},
    'LSST': {'input_dim': 6, 'seq_len': 36, 'num_classes': 96}
}

RESULTS_DIR = project_root / 'Results'
OUTPUT_DIR = RESULTS_DIR / 'Summary'
DATA_DIR = OUTPUT_DIR / 'data'
FIGURES_DIR = OUTPUT_DIR / 'figures'
TABLES_DIR = OUTPUT_DIR / 'tables'

# Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================================
# PHASE 1: DATA LOADING
# ============================================================================

def load_all_results() -> Dict[str, Dict[str, List[Dict]]]:
    """Load all raw results from the Results directory."""
    print("\n" + "="*60)
    print("PHASE 1: Loading Data")
    print("="*60)

    results = {}

    for dataset in DATASET_CONFIGS.keys():
        results[dataset] = {'denoise': [], 'noise': []}
        dataset_path = RESULTS_DIR / dataset

        for mode in ['denoise', 'noise']:
            mode_path = dataset_path / mode
            if not mode_path.exists():
                continue

            for class_dir in sorted(mode_path.iterdir()):
                if not class_dir.is_dir() or not class_dir.name.startswith('class_'):
                    continue

                class_id = int(class_dir.name.split('_')[1])

                for pair_dir in sorted(class_dir.iterdir()):
                    if not pair_dir.is_dir() or not pair_dir.name.startswith('pair_'):
                        continue

                    npz_path = pair_dir / 'raw_results.npz'
                    if npz_path.exists():
                        try:
                            data = np.load(npz_path, allow_pickle=True)

                            # Extract what we need
                            baseline = data['baseline'] if 'baseline' in data else None
                            head_patch = data['head_patch'] if 'head_patch' in data else None
                            layer_patch = data['layer_patch'] if 'layer_patch' in data else None

                            if baseline is not None and head_patch is not None:
                                # Compute delta_p
                                delta_p = head_patch[:, :, class_id] - baseline[class_id]

                                results[dataset][mode].append({
                                    'class_id': class_id,
                                    'pair_name': pair_dir.name,
                                    'baseline': baseline,
                                    'head_patch': head_patch,
                                    'layer_patch': layer_patch,
                                    'delta_p': delta_p,
                                    'path': str(pair_dir)
                                })
                        except Exception as e:
                            print(f"  Warning: Could not load {npz_path}: {e}")

    # Summary
    for dataset, modes in results.items():
        total = len(modes['denoise']) + len(modes['noise'])
        print(f"  {dataset}: {len(modes['denoise'])} denoise + {len(modes['noise'])} noise = {total} pairs")

    return results


# ============================================================================
# PHASE 2: CORE STATISTICAL COMPUTATIONS
# ============================================================================

def compute_all_confidence_intervals(results: Dict) -> pd.DataFrame:
    """Compute 95% CIs for all delta_P values."""
    print("\n" + "="*60)
    print("PHASE 2.1: Computing Confidence Intervals")
    print("="*60)

    ci_data = []

    for dataset in tqdm(DATASET_CONFIGS.keys(), desc="Computing CIs"):
        # Combine denoise and noise for overall analysis
        all_pairs = results[dataset]['denoise'] + results[dataset]['noise']

        if not all_pairs:
            continue

        # Per-head CIs
        for l in range(3):
            for h in range(8):
                values = [p['delta_p'][l, h] for p in all_pairs]
                if len(values) >= 2:
                    ci = compute_confidence_interval(np.array(values), confidence=0.95, method='bca', n_bootstrap=N_BOOTSTRAP)
                    ci_data.append({
                        'dataset': dataset,
                        'level': 'head',
                        'layer': l,
                        'head': h,
                        'n': len(values),
                        'mean': ci.mean,
                        'ci_lower': ci.lower,
                        'ci_upper': ci.upper,
                        'significant': ci.lower > 0 or ci.upper < 0  # CI doesn't contain 0
                    })

        # Per-layer CIs
        for l in range(3):
            values = [p['delta_p'][l, :].mean() for p in all_pairs]
            if len(values) >= 2:
                ci = compute_confidence_interval(np.array(values), confidence=0.95, method='bca', n_bootstrap=N_BOOTSTRAP)
                ci_data.append({
                    'dataset': dataset,
                    'level': 'layer',
                    'layer': l,
                    'head': -1,
                    'n': len(values),
                    'mean': ci.mean,
                    'ci_lower': ci.lower,
                    'ci_upper': ci.upper,
                    'significant': ci.lower > 0 or ci.upper < 0
                })

        # Overall dataset CI
        all_delta_p = np.concatenate([p['delta_p'].flatten() for p in all_pairs])
        ci = compute_confidence_interval(all_delta_p, confidence=0.95, method='bca', n_bootstrap=N_BOOTSTRAP)
        ci_data.append({
            'dataset': dataset,
            'level': 'dataset',
            'layer': -1,
            'head': -1,
            'n': len(all_delta_p),
            'mean': ci.mean,
            'ci_lower': ci.lower,
            'ci_upper': ci.upper,
            'significant': ci.lower > 0 or ci.upper < 0
        })

    df = pd.DataFrame(ci_data)
    df.to_csv(DATA_DIR / 'confidence_intervals.csv', index=False)
    print(f"  Saved {len(df)} CI records to confidence_intervals.csv")

    return df


def compute_fdr_correction(results: Dict) -> pd.DataFrame:
    """Apply FDR correction for multiple comparisons."""
    print("\n" + "="*60)
    print("PHASE 2.2: FDR Correction")
    print("="*60)

    p_value_data = []

    for dataset in DATASET_CONFIGS.keys():
        all_pairs = results[dataset]['denoise'] + results[dataset]['noise']

        if not all_pairs:
            continue

        # Test each head against H0: mean = 0
        for l in range(3):
            for h in range(8):
                values = np.array([p['delta_p'][l, h] for p in all_pairs])
                if len(values) >= 2:
                    t_stat, p_val = ttest_1samp(values, 0)
                    p_value_data.append({
                        'dataset': dataset,
                        'layer': l,
                        'head': h,
                        'mean_delta_p': np.mean(values),
                        'std_delta_p': np.std(values),
                        'n': len(values),
                        't_statistic': t_stat,
                        'p_value_raw': p_val
                    })

    df = pd.DataFrame(p_value_data)

    # Apply FDR correction
    fdr_result = apply_fdr_correction(df['p_value_raw'].values, method='benjamini_hochberg')
    df['p_value_fdr'] = fdr_result['p_corrected']
    df['significant_raw'] = df['p_value_raw'] < 0.05
    df['significant_fdr'] = fdr_result['significant']

    df.to_csv(DATA_DIR / 'fdr_corrected_pvalues.csv', index=False)

    # Summary
    for dataset in DATASET_CONFIGS.keys():
        subset = df[df['dataset'] == dataset]
        n_sig_raw = subset['significant_raw'].sum()
        n_sig_fdr = subset['significant_fdr'].sum()
        print(f"  {dataset}: {n_sig_raw}/24 significant (raw) -> {n_sig_fdr}/24 (FDR-corrected)")

    return df


def compute_all_effect_sizes(results: Dict) -> pd.DataFrame:
    """Compute effect sizes for all comparisons."""
    print("\n" + "="*60)
    print("PHASE 2.3: Effect Sizes")
    print("="*60)

    effect_data = []

    for dataset in DATASET_CONFIGS.keys():
        denoise_pairs = results[dataset]['denoise']
        noise_pairs = results[dataset]['noise']

        if not denoise_pairs or not noise_pairs:
            continue

        # Overall denoise vs noise comparison
        denoise_effects = np.array([p['delta_p'].mean() for p in denoise_pairs])
        noise_effects = np.array([p['delta_p'].mean() for p in noise_pairs])

        # Since denoise and noise may have different sizes, use unpaired
        effect = compute_effect_size(denoise_effects, noise_effects, paired=False)

        effect_data.append({
            'dataset': dataset,
            'comparison': 'denoise_vs_noise',
            'layer': -1,
            'head': -1,
            'cohens_d': effect.cohens_d,
            'cohens_d_ci_lower': effect.cohens_d_ci[0],
            'cohens_d_ci_upper': effect.cohens_d_ci[1],
            'cliffs_delta': effect.cliffs_delta,
            'interpretation': effect.interpretation
        })

        # Per-head effect sizes (vs baseline = 0)
        all_pairs = denoise_pairs + noise_pairs
        for l in range(3):
            for h in range(8):
                values = np.array([p['delta_p'][l, h] for p in all_pairs])
                if len(values) >= 2:
                    # Effect size relative to zero baseline
                    baseline = np.zeros_like(values)
                    effect = compute_effect_size(values, baseline, paired=True)

                    effect_data.append({
                        'dataset': dataset,
                        'comparison': 'vs_zero',
                        'layer': l,
                        'head': h,
                        'cohens_d': effect.cohens_d,
                        'cohens_d_ci_lower': effect.cohens_d_ci[0],
                        'cohens_d_ci_upper': effect.cohens_d_ci[1],
                        'cliffs_delta': effect.cliffs_delta,
                        'interpretation': effect.interpretation
                    })

    df = pd.DataFrame(effect_data)
    df.to_csv(DATA_DIR / 'effect_sizes.csv', index=False)
    print(f"  Saved {len(df)} effect size records")

    return df


# ============================================================================
# PHASE 3: BASELINE COMPARISONS (Requires Model Inference)
# ============================================================================

def load_models() -> Dict[str, torch.nn.Module]:
    """Load all pre-trained models."""
    print("\n" + "="*60)
    print("PHASE 3.1: Loading Models")
    print("="*60)

    models = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    model_dir = project_root / 'TST_models'

    for name, cfg in DATASET_CONFIGS.items():
        model_path = model_dir / f'TST_{name.lower()}.pth'
        if model_path.exists():
            model = TimeSeriesTransformer(
                input_dim=cfg['input_dim'],
                num_classes=cfg['num_classes'],
                seq_len=cfg['seq_len'],
                d_model=128,
                n_head=8,
                num_encoder_layers=3
            )
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            models[name] = model
            print(f"  Loaded {name} model")
        else:
            print(f"  Warning: Model not found at {model_path}")

    return models


def compute_null_distributions(models: Dict, results: Dict) -> pd.DataFrame:
    """Compute null distribution via random patching."""
    print("\n" + "="*60)
    print("PHASE 3.2: Null Distribution (Random Patching)")
    print("="*60)

    null_data = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for dataset in tqdm(DATASET_CONFIGS.keys(), desc="Computing null"):
        if dataset not in models:
            continue

        model = models[dataset]
        cfg = DATASET_CONFIGS[dataset]

        # Load actual data to get clean/corrupt pairs
        try:
            train_loader, test_loader = load_dataset(dataset)
            test_data = list(test_loader.dataset)
        except Exception as e:
            print(f"  Could not load dataset {dataset}: {e}")
            continue

        # Sample pairs for null distribution
        all_pairs = results[dataset]['denoise'][:SAMPLE_SIZE_BASELINES]

        if not all_pairs:
            continue

        null_effects = []

        for pair in tqdm(all_pairs, desc=f"  {dataset}", leave=False):
            try:
                # Get indices from pair name
                parts = pair['pair_name'].split('_')
                src_idx = int(parts[1])
                tgt_idx = int(parts[2])

                # Get data
                if src_idx < len(test_data) and tgt_idx < len(test_data):
                    clean_x = torch.tensor(test_data[src_idx][0]).unsqueeze(0).float().to(device)
                    corrupt_x = torch.tensor(test_data[tgt_idx][0]).unsqueeze(0).float().to(device)

                    null = random_patching_baseline(
                        model, clean_x, corrupt_x,
                        num_classes=cfg['num_classes'],
                        n_permutations=N_PERMUTATIONS,
                        seed=RANDOM_SEED
                    )
                    null_effects.extend(null['null_distribution'])
            except Exception as e:
                continue

        if null_effects:
            null_effects = np.array(null_effects)
            null_data.append({
                'dataset': dataset,
                'n_effects': len(null_effects),
                'mean': np.mean(null_effects),
                'std': np.std(null_effects),
                'p5': np.percentile(null_effects, 5),
                'p25': np.percentile(null_effects, 25),
                'p50': np.percentile(null_effects, 50),
                'p75': np.percentile(null_effects, 75),
                'p95': np.percentile(null_effects, 95),
                'p99': np.percentile(null_effects, 99)
            })
            print(f"  {dataset}: 95th percentile = {np.percentile(null_effects, 95):.4f}")

    df = pd.DataFrame(null_data)
    df.to_csv(DATA_DIR / 'null_distribution.csv', index=False)

    return df


def compute_baseline_comparisons(models: Dict, results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare activation patching to Integrated Gradients and Attention weights."""
    print("\n" + "="*60)
    print("PHASE 3.3: Baseline Method Comparisons")
    print("="*60)

    ig_data = []
    attn_data = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for dataset in tqdm(DATASET_CONFIGS.keys(), desc="Comparing methods"):
        if dataset not in models:
            continue

        model = models[dataset]
        cfg = DATASET_CONFIGS[dataset]

        # Load actual data
        try:
            train_loader, test_loader = load_dataset(dataset)
            test_data = list(test_loader.dataset)
        except Exception as e:
            print(f"  Could not load dataset {dataset}: {e}")
            continue

        # Sample pairs
        all_pairs = results[dataset]['denoise'][:SAMPLE_SIZE_BASELINES]

        ig_correlations = []
        attn_correlations = {'entropy': [], 'max': [], 'variance': []}

        for pair in tqdm(all_pairs, desc=f"  {dataset}", leave=False):
            try:
                parts = pair['pair_name'].split('_')
                src_idx = int(parts[1])

                if src_idx >= len(test_data):
                    continue

                clean_x = torch.tensor(test_data[src_idx][0]).unsqueeze(0).float().to(device)
                true_label = pair['class_id']
                patching_importance = pair['delta_p']

                # Integrated Gradients
                try:
                    ig_result = integrated_gradients_importance(
                        model, clean_x, true_label,
                        steps=50, device=device
                    )
                    rho, _ = spearmanr(patching_importance.flatten(), ig_result.importance_matrix.flatten())
                    if not np.isnan(rho):
                        ig_correlations.append(rho)
                except Exception:
                    pass

                # Attention weights
                for agg in ['entropy', 'max', 'variance']:
                    try:
                        attn_result = attention_weight_importance(model, clean_x, device=device, aggregation=agg)
                        rho, _ = spearmanr(patching_importance.flatten(), attn_result.importance_matrix.flatten())
                        if not np.isnan(rho):
                            attn_correlations[agg].append(rho)
                    except Exception:
                        pass

            except Exception as e:
                continue

        # Compute CIs for correlations
        if ig_correlations:
            ci = compute_confidence_interval(np.array(ig_correlations), confidence=0.95, method='bca')
            ig_data.append({
                'dataset': dataset,
                'method': 'integrated_gradients',
                'n': len(ig_correlations),
                'mean_rho': ci.mean,
                'ci_lower': ci.lower,
                'ci_upper': ci.upper
            })

        for agg, corrs in attn_correlations.items():
            if corrs:
                ci = compute_confidence_interval(np.array(corrs), confidence=0.95, method='bca')
                attn_data.append({
                    'dataset': dataset,
                    'method': f'attention_{agg}',
                    'n': len(corrs),
                    'mean_rho': ci.mean,
                    'ci_lower': ci.lower,
                    'ci_upper': ci.upper
                })

    ig_df = pd.DataFrame(ig_data)
    attn_df = pd.DataFrame(attn_data)

    # Combine and save
    combined = pd.concat([ig_df, attn_df], ignore_index=True)
    combined.to_csv(DATA_DIR / 'baseline_comparisons.csv', index=False)

    print("\n  Method comparison summary:")
    for _, row in combined.iterrows():
        print(f"    {row['dataset']} vs {row['method']}: ρ = {row['mean_rho']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")

    return ig_df, attn_df


# ============================================================================
# PHASE 4: STABILITY ANALYSIS WITH STATISTICS
# ============================================================================

def compute_stability_statistics() -> pd.DataFrame:
    """Add CIs to existing stability results."""
    print("\n" + "="*60)
    print("PHASE 4: Stability Analysis Statistics")
    print("="*60)

    stability_path = RESULTS_DIR / 'Stability' / 'Summary' / 'aggregate_results.csv'

    if not stability_path.exists():
        print("  Warning: Stability results not found")
        return pd.DataFrame()

    df = pd.read_csv(stability_path)

    # Group by dataset and compute CIs
    stability_data = []

    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]

        # CI for rank correlation across perturbations
        rank_corrs = subset['rank_corr'].values
        top5_overlaps = subset['top5'].values
        stabilities = subset['stability'].values

        if len(rank_corrs) >= 2:
            rank_ci = compute_confidence_interval(rank_corrs, confidence=0.95, method='bca')
            top5_ci = compute_confidence_interval(top5_overlaps, confidence=0.95, method='bca')
            stab_ci = compute_confidence_interval(stabilities, confidence=0.95, method='bca')

            stability_data.append({
                'dataset': dataset,
                'n_perturbations': len(rank_corrs),
                'rank_corr_mean': rank_ci.mean,
                'rank_corr_ci_lower': rank_ci.lower,
                'rank_corr_ci_upper': rank_ci.upper,
                'top5_mean': top5_ci.mean,
                'top5_ci_lower': top5_ci.lower,
                'top5_ci_upper': top5_ci.upper,
                'stability_mean': stab_ci.mean,
                'stability_ci_lower': stab_ci.lower,
                'stability_ci_upper': stab_ci.upper
            })

    result_df = pd.DataFrame(stability_data)
    result_df.to_csv(DATA_DIR / 'stability_with_statistics.csv', index=False)

    print("  Stability summary with CIs:")
    for _, row in result_df.iterrows():
        print(f"    {row['dataset']}: ρ = {row['rank_corr_mean']:.3f} [{row['rank_corr_ci_lower']:.3f}, {row['rank_corr_ci_upper']:.3f}]")

    return result_df


# ============================================================================
# PHASE 5: PUBLICATION FIGURES
# ============================================================================

def generate_all_figures(ci_df: pd.DataFrame, fdr_df: pd.DataFrame,
                         effect_df: pd.DataFrame, baseline_df: pd.DataFrame,
                         stability_df: pd.DataFrame, null_df: pd.DataFrame,
                         results: Dict):
    """Generate all publication figures."""
    print("\n" + "="*60)
    print("PHASE 5: Generating Figures")
    print("="*60)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: Head Importance Heatmaps
    print("  Figure 1: Head importance heatmaps...")
    fig1_head_importance(ci_df, fdr_df, results)

    # Figure 2: Baseline Comparison
    print("  Figure 2: Baseline comparison...")
    fig2_baseline_comparison(baseline_df)

    # Figure 3: Stability Analysis
    print("  Figure 3: Stability analysis...")
    fig3_stability(stability_df)

    # Figure 4: Effect Size Forest Plot
    print("  Figure 4: Effect sizes...")
    fig4_effect_sizes(effect_df)

    # Figure 5: Null Distribution
    print("  Figure 5: Null distribution...")
    fig5_null_distribution(null_df, results)


def fig1_head_importance(ci_df: pd.DataFrame, fdr_df: pd.DataFrame, results: Dict):
    """Create head importance heatmaps with significance markers."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    datasets = ['JapaneseVowels', 'PenDigits', 'LSST']

    for i, dataset in enumerate(datasets):
        # Get mean delta_p per head
        head_ci = ci_df[(ci_df['dataset'] == dataset) & (ci_df['level'] == 'head')]

        if head_ci.empty:
            continue

        # Create matrix
        mean_matrix = np.zeros((3, 8))
        sig_matrix = np.zeros((3, 8), dtype=bool)

        for _, row in head_ci.iterrows():
            l, h = int(row['layer']), int(row['head'])
            mean_matrix[l, h] = row['mean']

        # Get FDR significance
        fdr_subset = fdr_df[fdr_df['dataset'] == dataset]
        for _, row in fdr_subset.iterrows():
            l, h = int(row['layer']), int(row['head'])
            sig_matrix[l, h] = row['significant_fdr']

        # Plot heatmap
        vmax = max(abs(mean_matrix.min()), abs(mean_matrix.max()))
        sns.heatmap(mean_matrix, ax=axes[i], cmap='RdBu_r', center=0,
                    vmin=-vmax, vmax=vmax, annot=True, fmt='.3f',
                    xticklabels=[f'H{h}' for h in range(8)],
                    yticklabels=[f'L{l}' for l in range(3)])

        # Add significance markers
        for l in range(3):
            for h in range(8):
                if sig_matrix[l, h]:
                    axes[i].text(h + 0.5, l + 0.15, '*', ha='center', va='center',
                               fontsize=16, color='black', fontweight='bold')

        axes[i].set_title(f'{dataset}\n(* = FDR p < 0.05)')
        axes[i].set_xlabel('Head')
        axes[i].set_ylabel('Layer')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_head_importance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig1_head_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig2_baseline_comparison(baseline_df: pd.DataFrame):
    """Create baseline method comparison figure."""
    if baseline_df.empty:
        print("    Skipping (no baseline data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    datasets = baseline_df['dataset'].unique()
    methods = baseline_df['method'].unique()

    x = np.arange(len(datasets))
    width = 0.15

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for j, method in enumerate(methods):
        subset = baseline_df[baseline_df['method'] == method]
        means = []
        errors_lower = []
        errors_upper = []

        for dataset in datasets:
            row = subset[subset['dataset'] == dataset]
            if not row.empty:
                means.append(row['mean_rho'].values[0])
                errors_lower.append(row['mean_rho'].values[0] - row['ci_lower'].values[0])
                errors_upper.append(row['ci_upper'].values[0] - row['mean_rho'].values[0])
            else:
                means.append(0)
                errors_lower.append(0)
                errors_upper.append(0)

        ax.bar(x + j * width, means, width, label=method.replace('_', ' ').title(),
               color=colors[j], yerr=[errors_lower, errors_upper], capsize=3)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Spearman ρ (Patching vs Method)')
    ax.set_title('Activation Patching vs Alternative Methods')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_baseline_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig2_baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig3_stability(stability_df: pd.DataFrame):
    """Create stability analysis figure."""
    if stability_df.empty:
        print("    Skipping (no stability data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = stability_df['dataset'].unique()
    x = np.arange(len(datasets))

    means = stability_df['rank_corr_mean'].values
    ci_lower = means - stability_df['rank_corr_ci_lower'].values
    ci_upper = stability_df['rank_corr_ci_upper'].values - means

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(x, means, yerr=[ci_lower, ci_upper], capsize=5, color=colors[:len(datasets)])

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Mean Rank Correlation (ρ)')
    ax.set_title('Mechanism Stability Across Perturbations (95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Stability threshold')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_stability.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig3_stability.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig4_effect_sizes(effect_df: pd.DataFrame):
    """Create effect size forest plot."""
    # Filter to per-head effect sizes
    head_effects = effect_df[(effect_df['comparison'] == 'vs_zero') & (effect_df['layer'] >= 0)]

    if head_effects.empty:
        print("    Skipping (no effect size data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    datasets = ['JapaneseVowels', 'PenDigits', 'LSST']

    for i, dataset in enumerate(datasets):
        subset = head_effects[head_effects['dataset'] == dataset]

        if subset.empty:
            continue

        # Sort by layer and head
        subset = subset.sort_values(['layer', 'head'])

        y_labels = [f'L{int(r["layer"])}H{int(r["head"])}' for _, r in subset.iterrows()]
        y_pos = np.arange(len(y_labels))

        means = subset['cohens_d'].values
        ci_lower = means - subset['cohens_d_ci_lower'].values
        ci_upper = subset['cohens_d_ci_upper'].values - means

        # Color by interpretation
        colors = []
        for interp in subset['interpretation']:
            if interp == 'large':
                colors.append('#e74c3c')
            elif interp == 'medium':
                colors.append('#f39c12')
            elif interp == 'small':
                colors.append('#3498db')
            else:
                colors.append('#95a5a6')

        axes[i].barh(y_pos, means, xerr=[ci_lower, ci_upper], capsize=2, color=colors)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(y_labels, fontsize=8)
        axes[i].axvline(0, color='black', linestyle='-', alpha=0.5)
        axes[i].axvline(0.2, color='blue', linestyle='--', alpha=0.3, label='small')
        axes[i].axvline(0.5, color='orange', linestyle='--', alpha=0.3, label='medium')
        axes[i].axvline(0.8, color='red', linestyle='--', alpha=0.3, label='large')
        axes[i].set_xlabel("Cohen's d")
        axes[i].set_title(dataset)

        if i == 0:
            axes[i].legend(loc='lower right', fontsize=8)

    plt.suptitle('Effect Sizes by Head (95% CI)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_effect_sizes.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig4_effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig5_null_distribution(null_df: pd.DataFrame, results: Dict):
    """Create null distribution comparison figure."""
    if null_df.empty:
        print("    Skipping (no null distribution data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'JapaneseVowels': '#2ecc71', 'PenDigits': '#3498db', 'LSST': '#e74c3c'}

    for _, row in null_df.iterrows():
        dataset = row['dataset']

        # Get observed effects for this dataset
        all_pairs = results[dataset]['denoise'] + results[dataset]['noise']
        observed_effects = [abs(p['delta_p']).max() for p in all_pairs]

        # Plot observed vs threshold
        ax.scatter([dataset] * len(observed_effects), observed_effects,
                  alpha=0.5, color=colors.get(dataset, 'gray'), s=30, label=f'{dataset} observed')

        # Plot 95th percentile threshold
        ax.scatter([dataset], [row['p95']], marker='_', s=200, color='black', linewidths=3)
        ax.text(dataset, row['p95'] + 0.02, f"95th: {row['p95']:.3f}", ha='center', fontsize=9)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Max |ΔP| Effect')
    ax.set_title('Observed Effects vs Null Distribution 95th Percentile')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_null_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig5_null_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# PHASE 6: PUBLICATION TABLES
# ============================================================================

def generate_all_tables(ci_df: pd.DataFrame, fdr_df: pd.DataFrame,
                        effect_df: pd.DataFrame, baseline_df: pd.DataFrame,
                        stability_df: pd.DataFrame, null_df: pd.DataFrame,
                        results: Dict):
    """Generate all publication tables."""
    print("\n" + "="*60)
    print("PHASE 6: Generating Tables")
    print("="*60)

    # Table 1: Main Results Summary
    print("  Table 1: Main summary...")
    table1_summary(ci_df, fdr_df, effect_df, results)

    # Table 2: Significant Heads
    print("  Table 2: Significant heads...")
    table2_significant_heads(ci_df, fdr_df)

    # Table 3: Method Comparison
    print("  Table 3: Method comparison...")
    table3_method_comparison(baseline_df)

    # Table 4: Stability Results
    print("  Table 4: Stability...")
    table4_stability(stability_df)

    # Table 5: Power Analysis
    print("  Table 5: Power analysis...")
    table5_power_analysis(ci_df, results)


def table1_summary(ci_df: pd.DataFrame, fdr_df: pd.DataFrame,
                   effect_df: pd.DataFrame, results: Dict):
    """Generate main results summary table."""
    rows = []

    for dataset in DATASET_CONFIGS.keys():
        all_pairs = results[dataset]['denoise'] + results[dataset]['noise']
        n_pairs = len(all_pairs)

        # Get overall CI
        dataset_ci = ci_df[(ci_df['dataset'] == dataset) & (ci_df['level'] == 'dataset')]
        if not dataset_ci.empty:
            mean_dp = dataset_ci['mean'].values[0]
            ci_lower = dataset_ci['ci_lower'].values[0]
            ci_upper = dataset_ci['ci_upper'].values[0]
        else:
            mean_dp = ci_lower = ci_upper = np.nan

        # Get effect size
        effect_row = effect_df[(effect_df['dataset'] == dataset) & (effect_df['comparison'] == 'denoise_vs_noise')]
        if not effect_row.empty:
            cohens_d = effect_row['cohens_d'].values[0]
        else:
            cohens_d = np.nan

        # Count significant heads
        fdr_subset = fdr_df[fdr_df['dataset'] == dataset]
        n_sig = fdr_subset['significant_fdr'].sum() if not fdr_subset.empty else 0

        rows.append({
            'Dataset': dataset,
            'N Pairs': n_pairs,
            'Mean ΔP': f'{mean_dp:.3f}',
            '95% CI': f'[{ci_lower:.3f}, {ci_upper:.3f}]',
            "Cohen's d": f'{cohens_d:.2f}' if not np.isnan(cohens_d) else 'N/A',
            'Sig. Heads': f'{n_sig}/24'
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'table1_summary.csv', index=False)

    # Also save as markdown
    with open(TABLES_DIR / 'table1_summary.md', 'w') as f:
        f.write(df.to_markdown(index=False))


def table2_significant_heads(ci_df: pd.DataFrame, fdr_df: pd.DataFrame):
    """Generate significant heads table."""
    # Merge CI and FDR data
    head_ci = ci_df[(ci_df['level'] == 'head')].copy()

    rows = []
    for _, ci_row in head_ci.iterrows():
        dataset = ci_row['dataset']
        layer = int(ci_row['layer'])
        head = int(ci_row['head'])

        fdr_row = fdr_df[(fdr_df['dataset'] == dataset) &
                         (fdr_df['layer'] == layer) &
                         (fdr_df['head'] == head)]

        if not fdr_row.empty and fdr_row['significant_fdr'].values[0]:
            rows.append({
                'Dataset': dataset,
                'Head': f'L{layer}H{head}',
                'Mean ΔP': f"{ci_row['mean']:.4f}",
                '95% CI': f"[{ci_row['ci_lower']:.4f}, {ci_row['ci_upper']:.4f}]",
                'p (raw)': f"{fdr_row['p_value_raw'].values[0]:.4f}",
                'p (FDR)': f"{fdr_row['p_value_fdr'].values[0]:.4f}"
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(['Dataset', 'Mean ΔP'], ascending=[True, False])
    df.to_csv(TABLES_DIR / 'table2_significant_heads.csv', index=False)

    with open(TABLES_DIR / 'table2_significant_heads.md', 'w') as f:
        f.write(df.to_markdown(index=False))


def table3_method_comparison(baseline_df: pd.DataFrame):
    """Generate method comparison table."""
    if baseline_df.empty:
        return

    rows = []
    for _, row in baseline_df.iterrows():
        rows.append({
            'Dataset': row['dataset'],
            'Comparison': f"Patching vs {row['method'].replace('_', ' ').title()}",
            'ρ': f"{row['mean_rho']:.3f}",
            '95% CI': f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]",
            'N': int(row['n'])
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'table3_method_comparison.csv', index=False)

    with open(TABLES_DIR / 'table3_method_comparison.md', 'w') as f:
        f.write(df.to_markdown(index=False))


def table4_stability(stability_df: pd.DataFrame):
    """Generate stability results table."""
    if stability_df.empty:
        return

    rows = []
    for _, row in stability_df.iterrows():
        rows.append({
            'Dataset': row['dataset'],
            'N Perturbations': int(row['n_perturbations']),
            'Rank ρ': f"{row['rank_corr_mean']:.3f}",
            '95% CI': f"[{row['rank_corr_ci_lower']:.3f}, {row['rank_corr_ci_upper']:.3f}]",
            'Top-5 Overlap': f"{row['top5_mean']:.3f}",
            'Stable?': 'Yes' if row['rank_corr_mean'] > 0.7 else 'No'
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'table4_stability.csv', index=False)

    with open(TABLES_DIR / 'table4_stability.md', 'w') as f:
        f.write(df.to_markdown(index=False))


def table5_power_analysis(ci_df: pd.DataFrame, results: Dict):
    """Generate power analysis table."""
    rows = []

    for dataset in DATASET_CONFIGS.keys():
        all_pairs = results[dataset]['denoise'] + results[dataset]['noise']
        n = len(all_pairs)

        # Get mean effect size
        dataset_ci = ci_df[(ci_df['dataset'] == dataset) & (ci_df['level'] == 'dataset')]
        if not dataset_ci.empty and n > 0:
            mean_effect = abs(dataset_ci['mean'].values[0])

            # Compute power for detecting this effect
            power_result = power_analysis(effect_size=0.5, n=n, alpha=0.05)

            rows.append({
                'Dataset': dataset,
                'N': n,
                'Observed |ΔP|': f'{mean_effect:.3f}',
                'Power (d=0.5)': f"{power_result['power']:.1%}",
                'Adequate?': 'Yes' if power_result['power'] >= 0.8 else 'No'
            })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / 'table5_power_analysis.csv', index=False)

    with open(TABLES_DIR / 'table5_power_analysis.md', 'w') as f:
        f.write(df.to_markdown(index=False))


# ============================================================================
# PHASE 7: FINAL REPORT
# ============================================================================

def generate_final_report(ci_df: pd.DataFrame, fdr_df: pd.DataFrame,
                          effect_df: pd.DataFrame, baseline_df: pd.DataFrame,
                          stability_df: pd.DataFrame, null_df: pd.DataFrame,
                          results: Dict):
    """Generate comprehensive statistical report."""
    print("\n" + "="*60)
    print("PHASE 7: Generating Final Report")
    print("="*60)

    report = []
    report.append("# Complete Statistical Analysis Report")
    report.append("")
    report.append("## Executive Summary")
    report.append("")

    # Summary statistics
    total_pairs = sum(len(results[d]['denoise']) + len(results[d]['noise']) for d in DATASET_CONFIGS.keys())
    report.append(f"**Total pairs analyzed:** {total_pairs}")
    report.append(f"**Datasets:** {', '.join(DATASET_CONFIGS.keys())}")
    report.append(f"**Bootstrap iterations:** {N_BOOTSTRAP}")
    report.append(f"**Random seed:** {RANDOM_SEED}")
    report.append("")

    # Key findings
    report.append("## Key Findings")
    report.append("")

    for dataset in DATASET_CONFIGS.keys():
        dataset_ci = ci_df[(ci_df['dataset'] == dataset) & (ci_df['level'] == 'dataset')]
        if not dataset_ci.empty:
            row = dataset_ci.iloc[0]
            report.append(f"### {dataset}")
            report.append(f"- Mean ΔP: {row['mean']:.4f} [95% CI: {row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

            fdr_subset = fdr_df[fdr_df['dataset'] == dataset]
            n_sig = fdr_subset['significant_fdr'].sum() if not fdr_subset.empty else 0
            report.append(f"- Significant heads (FDR-corrected): {n_sig}/24")
            report.append("")

    # Stability results
    if not stability_df.empty:
        report.append("## Stability Analysis")
        report.append("")
        for _, row in stability_df.iterrows():
            report.append(f"### {row['dataset']}")
            report.append(f"- Mean rank correlation: {row['rank_corr_mean']:.3f} [{row['rank_corr_ci_lower']:.3f}, {row['rank_corr_ci_upper']:.3f}]")
            report.append(f"- Verdict: {'Stable' if row['rank_corr_mean'] > 0.7 else 'Unstable'}")
            report.append("")

    # Baseline comparisons
    if not baseline_df.empty:
        report.append("## Baseline Method Comparisons")
        report.append("")
        report.append("| Dataset | Method | ρ | 95% CI |")
        report.append("|---------|--------|---|--------|")
        for _, row in baseline_df.iterrows():
            report.append(f"| {row['dataset']} | {row['method']} | {row['mean_rho']:.3f} | [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] |")
        report.append("")

    # Limitations
    report.append("## Statistical Limitations")
    report.append("")
    report.append("1. **Sample size (n=3 datasets)**: Cannot establish cross-dataset correlations")
    report.append("2. **Single training run**: Results may be specific to training seeds")
    report.append("3. **Multiple comparisons**: FDR correction applied but some false positives possible")
    report.append("")

    # Files generated
    report.append("## Files Generated")
    report.append("")
    report.append("### Data Files")
    report.append("- `data/confidence_intervals.csv`")
    report.append("- `data/fdr_corrected_pvalues.csv`")
    report.append("- `data/effect_sizes.csv`")
    report.append("- `data/baseline_comparisons.csv`")
    report.append("- `data/stability_with_statistics.csv`")
    report.append("- `data/null_distribution.csv`")
    report.append("")
    report.append("### Figures")
    report.append("- `figures/fig1_head_importance.pdf`")
    report.append("- `figures/fig2_baseline_comparison.pdf`")
    report.append("- `figures/fig3_stability.pdf`")
    report.append("- `figures/fig4_effect_sizes.pdf`")
    report.append("- `figures/fig5_null_distribution.pdf`")
    report.append("")
    report.append("### Tables")
    report.append("- `tables/table1_summary.csv`")
    report.append("- `tables/table2_significant_heads.csv`")
    report.append("- `tables/table3_method_comparison.csv`")
    report.append("- `tables/table4_stability.csv`")
    report.append("- `tables/table5_power_analysis.csv`")
    report.append("")

    # Write report
    with open(OUTPUT_DIR / 'COMPLETE_STATISTICAL_REPORT.md', 'w') as f:
        f.write('\n'.join(report))

    print("  Report saved to COMPLETE_STATISTICAL_REPORT.md")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete statistical analysis."""
    print("="*60)
    print("COMPLETE STATISTICAL ANALYSIS")
    print("TST Mechanistic Interpretability")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")

    # Phase 1: Load data
    results = load_all_results()

    # Phase 2: Core statistics
    ci_df = compute_all_confidence_intervals(results)
    fdr_df = compute_fdr_correction(results)
    effect_df = compute_all_effect_sizes(results)

    # Phase 3: Baseline comparisons (requires models)
    models = load_models()
    null_df = compute_null_distributions(models, results)
    baseline_df = pd.DataFrame()
    if models:
        ig_df, attn_df = compute_baseline_comparisons(models, results)
        baseline_df = pd.concat([ig_df, attn_df], ignore_index=True) if not ig_df.empty else pd.DataFrame()

    # Phase 4: Stability statistics
    stability_df = compute_stability_statistics()

    # Phase 5: Figures
    generate_all_figures(ci_df, fdr_df, effect_df, baseline_df, stability_df, null_df, results)

    # Phase 6: Tables
    generate_all_tables(ci_df, fdr_df, effect_df, baseline_df, stability_df, null_df, results)

    # Phase 7: Final report
    generate_final_report(ci_df, fdr_df, effect_df, baseline_df, stability_df, null_df, results)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  - {len(list(DATA_DIR.glob('*.csv')))} data files")
    print(f"  - {len(list(FIGURES_DIR.glob('*.pdf')))} figures")
    print(f"  - {len(list(TABLES_DIR.glob('*.csv')))} tables")
    print(f"  - 1 comprehensive report")


if __name__ == '__main__':
    main()
