"""
Stability metrics for comparing mechanistic explanations.

Measures how much the model's internal "explanation" changes
when inputs are perturbed but predictions stay the same.
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def get_head_importance(patch_probs: np.ndarray, baseline: np.ndarray, true_label: int) -> np.ndarray:
    """
    Compute importance (delta P) for each attention head.

    Args:
        patch_probs: (L, H, num_classes) - probabilities after patching each head
        baseline: (num_classes,) - baseline probabilities
        true_label: The correct class index

    Returns:
        (L, H) array of delta P values
    """
    return patch_probs[:, :, true_label] - baseline[true_label]


def get_head_ranking(importance: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Rank heads by absolute importance.

    Args:
        importance: (L, H) array of delta P values

    Returns:
        List of (layer, head, importance) sorted by |importance| descending
    """
    L, H = importance.shape
    heads = []
    for l in range(L):
        for h in range(H):
            heads.append((l, h, importance[l, h]))

    return sorted(heads, key=lambda x: abs(x[2]), reverse=True)


def head_rank_correlation(
    baseline_importance: np.ndarray,
    perturbed_importance: np.ndarray
) -> float:
    """
    Spearman correlation between head importance rankings.

    High correlation (close to 1) = same heads are important
    Low correlation (close to 0) = different heads become important

    Args:
        baseline_importance: (L, H) delta P from baseline
        perturbed_importance: (L, H) delta P from perturbed

    Returns:
        Spearman rho (-1 to 1)
    """
    baseline_flat = np.abs(baseline_importance).flatten()
    perturbed_flat = np.abs(perturbed_importance).flatten()

    rho, _ = spearmanr(baseline_flat, perturbed_flat)
    return rho


def topk_overlap(
    baseline_importance: np.ndarray,
    perturbed_importance: np.ndarray,
    k: int = 5
) -> float:
    """
    Jaccard overlap of top-K most important heads.

    1.0 = identical top-K heads
    0.0 = completely different top-K heads

    Args:
        baseline_importance: (L, H) delta P from baseline
        perturbed_importance: (L, H) delta P from perturbed
        k: Number of top heads to compare

    Returns:
        Jaccard similarity (0 to 1)
    """
    baseline_ranking = get_head_ranking(baseline_importance)
    perturbed_ranking = get_head_ranking(perturbed_importance)

    baseline_topk = set((l, h) for l, h, _ in baseline_ranking[:k])
    perturbed_topk = set((l, h) for l, h, _ in perturbed_ranking[:k])

    intersection = len(baseline_topk & perturbed_topk)
    union = len(baseline_topk | perturbed_topk)

    return intersection / union if union > 0 else 0.0


def patch_recovery_delta(
    baseline_importance: np.ndarray,
    perturbed_importance: np.ndarray
) -> Dict:
    """
    Compare the magnitude of patching effects.

    Args:
        baseline_importance: (L, H) delta P from baseline
        perturbed_importance: (L, H) delta P from perturbed

    Returns:
        Dict with mean_abs_diff, max_abs_diff, correlation
    """
    diff = np.abs(baseline_importance - perturbed_importance)

    return {
        'mean_abs_diff': float(np.mean(diff)),
        'max_abs_diff': float(np.max(diff)),
        'std_diff': float(np.std(diff))
    }


def mechanism_stability_score(
    baseline_importance: np.ndarray,
    perturbed_importance: np.ndarray,
    k: int = 5
) -> float:
    """
    Composite stability score (0 to 1, higher = more stable).

    Combines:
    - 50% rank correlation
    - 30% top-K overlap
    - 20% inverse of mean difference

    Args:
        baseline_importance: (L, H) delta P from baseline
        perturbed_importance: (L, H) delta P from perturbed
        k: K for top-K overlap

    Returns:
        Stability score (0 to 1)
    """
    rank_corr = head_rank_correlation(baseline_importance, perturbed_importance)
    rank_corr_norm = (rank_corr + 1) / 2  # Normalize from [-1,1] to [0,1]

    topk = topk_overlap(baseline_importance, perturbed_importance, k)

    recovery = patch_recovery_delta(baseline_importance, perturbed_importance)
    # Normalize mean diff (assume max reasonable diff is 1.0)
    diff_norm = 1 - min(recovery['mean_abs_diff'], 1.0)

    return 0.5 * rank_corr_norm + 0.3 * topk + 0.2 * diff_norm


def compute_all_metrics(
    baseline_importance: np.ndarray,
    perturbed_importance: np.ndarray,
    k_values: List[int] = [3, 5, 10]
) -> Dict:
    """
    Compute all stability metrics.

    Args:
        baseline_importance: (L, H) delta P from baseline
        perturbed_importance: (L, H) delta P from perturbed
        k_values: List of K values for top-K overlap

    Returns:
        Dict with all metrics
    """
    result = {
        'rank_correlation': head_rank_correlation(baseline_importance, perturbed_importance),
        'recovery_delta': patch_recovery_delta(baseline_importance, perturbed_importance),
        'stability_score': mechanism_stability_score(baseline_importance, perturbed_importance)
    }

    for k in k_values:
        result[f'topk_overlap_k{k}'] = topk_overlap(baseline_importance, perturbed_importance, k)

    return result


def plot_importance_comparison(
    baseline_importance: np.ndarray,
    perturbed_importance: np.ndarray,
    title: str = "Head Importance: Baseline vs Perturbed",
    save_path: str = None
) -> plt.Figure:
    """
    Side-by-side heatmaps comparing head importance.

    Args:
        baseline_importance: (L, H) delta P from baseline
        perturbed_importance: (L, H) delta P from perturbed
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    L, H = baseline_importance.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vmax = max(np.abs(baseline_importance).max(), np.abs(perturbed_importance).max())
    vmin = -vmax

    # Baseline
    sns.heatmap(baseline_importance, ax=axes[0], cmap='RdBu_r', center=0,
                vmin=vmin, vmax=vmax, annot=True, fmt='.2f',
                xticklabels=[f'H{h}' for h in range(H)],
                yticklabels=[f'L{l}' for l in range(L)])
    axes[0].set_title('Baseline')
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')

    # Perturbed
    sns.heatmap(perturbed_importance, ax=axes[1], cmap='RdBu_r', center=0,
                vmin=vmin, vmax=vmax, annot=True, fmt='.2f',
                xticklabels=[f'H{h}' for h in range(H)],
                yticklabels=[f'L{l}' for l in range(L)])
    axes[1].set_title('Perturbed')
    axes[1].set_xlabel('Head')
    axes[1].set_ylabel('Layer')

    # Difference
    diff = perturbed_importance - baseline_importance
    sns.heatmap(diff, ax=axes[2], cmap='PuOr', center=0,
                annot=True, fmt='.2f',
                xticklabels=[f'H{h}' for h in range(H)],
                yticklabels=[f'L{l}' for l in range(L)])
    axes[2].set_title('Difference (Pert - Base)')
    axes[2].set_xlabel('Head')
    axes[2].set_ylabel('Layer')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_correlation_decay(
    results: Dict[str, Dict],
    metric: str = 'rank_correlation',
    title: str = "Mechanism Stability vs Perturbation Strength",
    save_path: str = None
) -> plt.Figure:
    """
    Plot how stability metrics change with perturbation strength.

    Args:
        results: Dict mapping perturbation config names to metric dicts
        metric: Which metric to plot
        title: Plot title
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by perturbation type
    by_type = {}
    for config_name, metrics in results.items():
        parts = config_name.split('_')
        ptype = parts[0]
        if ptype not in by_type:
            by_type[ptype] = []
        by_type[ptype].append((config_name, metrics.get(metric, 0)))

    colors = {'gaussian': 'blue', 'time': 'green', 'phase': 'red'}
    for ptype, data in by_type.items():
        data.sort(key=lambda x: x[0])
        names = [d[0] for d in data]
        values = [d[1] for d in data]
        color = colors.get(ptype, 'gray')
        ax.plot(names, values, 'o-', label=ptype, color=color, markersize=10)

    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_xlabel('Perturbation Config')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_topk_bars(
    results: Dict[str, Dict],
    k_values: List[int] = [3, 5, 10],
    title: str = "Top-K Overlap Across Perturbations",
    save_path: str = None
) -> plt.Figure:
    """
    Bar chart of top-K overlap for different perturbations.

    Args:
        results: Dict mapping perturbation names to metric dicts
        k_values: K values to plot
        title: Plot title
        save_path: If provided, save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = list(results.keys())
    x = np.arange(len(configs))
    width = 0.25

    for i, k in enumerate(k_values):
        values = [results[c].get(f'topk_overlap_k{k}', 0) for c in configs]
        ax.bar(x + i * width, values, width, label=f'K={k}')

    ax.set_ylabel('Jaccard Overlap')
    ax.set_xlabel('Perturbation')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_summary_table(all_results: Dict[str, Dict]) -> str:
    """
    Create a markdown summary table of all results.

    Args:
        all_results: Dict mapping config names to metric dicts

    Returns:
        Markdown table string
    """
    lines = [
        "| Perturbation | Rank ρ | Top-3 | Top-5 | Top-10 | Stability |",
        "|--------------|--------|-------|-------|--------|-----------|"
    ]

    for config, metrics in all_results.items():
        rank = metrics.get('rank_correlation', 0)
        t3 = metrics.get('topk_overlap_k3', 0)
        t5 = metrics.get('topk_overlap_k5', 0)
        t10 = metrics.get('topk_overlap_k10', 0)
        stab = metrics.get('stability_score', 0)

        lines.append(f"| {config} | {rank:.3f} | {t3:.3f} | {t5:.3f} | {t10:.3f} | {stab:.3f} |")

    return '\n'.join(lines)
