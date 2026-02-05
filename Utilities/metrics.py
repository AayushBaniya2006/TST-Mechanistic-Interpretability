"""
Additional patching metrics per best practices literature.

This module provides alternative metrics for measuring activation
patching effects beyond simple probability difference (ΔP).

Metrics included:
- Logit difference (recommended by ICLR 2024 best practices)
- KL divergence (full distributional change)
- Cross-entropy change
- Probability ratio

Per "Towards Best Practices of Activation Patching" (ICLR 2024):
- Logit difference is preferred as it's linear in logit space
- KL divergence captures full distributional changes
- Probability can saturate, making effects hard to interpret

Usage:
    >>> from Utilities.metrics import (
    ...     compute_logit_difference,
    ...     compute_kl_divergence,
    ...     compute_all_metrics
    ... )
    >>>
    >>> # After patching
    >>> ld = compute_logit_difference(patched_logits, class_a=0, class_b=1)
    >>> kl = compute_kl_divergence(original_probs, patched_probs)

References:
    Zhang et al. "Towards Best Practices of Activation Patching" (ICLR 2024)
"""

from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PatchingMetrics:
    """Complete metrics for a patching intervention."""
    delta_p: float              # Probability change for target class
    logit_diff: float           # Logit difference (class_a - class_b)
    kl_divergence: float        # KL(original || patched)
    cross_entropy_change: float # Change in CE loss
    probability_ratio: float    # P(target|patched) / P(target|original)

    # Raw values
    original_logits: Optional[np.ndarray] = None
    patched_logits: Optional[np.ndarray] = None
    original_probs: Optional[np.ndarray] = None
    patched_probs: Optional[np.ndarray] = None


# =============================================================================
# LOGIT DIFFERENCE
# =============================================================================

def compute_logit_difference(
    logits: torch.Tensor,
    class_a: int,
    class_b: int
) -> float:
    """
    Compute logit difference between two classes.

    This is the recommended metric per ICLR 2024 best practices:
    - Linear in logit space (no saturation)
    - More interpretable than probability
    - Directly measures relative class preference

    Args:
        logits: Model logits, shape (batch, num_classes) or (num_classes,)
        class_a: First class index (typically true class)
        class_b: Second class index (typically predicted/corrupt class)

    Returns:
        Logit difference: logits[class_a] - logits[class_b]

    Example:
        >>> logits = model(x)
        >>> ld = compute_logit_difference(logits, true_class, predicted_class)
        >>> # Positive ld means model prefers true_class
    """
    if logits.dim() == 2:
        logits = logits[0]  # Remove batch dimension

    return (logits[class_a] - logits[class_b]).item()


def compute_logit_difference_change(
    original_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    class_a: int,
    class_b: int
) -> float:
    """
    Compute change in logit difference due to patching.

    Measures how much the patching intervention changed the model's
    relative preference between two classes.

    Args:
        original_logits: Logits before patching
        patched_logits: Logits after patching
        class_a: First class (typically true class)
        class_b: Second class (typically corrupt class)

    Returns:
        Change in logit difference: LD_patched - LD_original
        Positive means patching increased preference for class_a
    """
    ld_original = compute_logit_difference(original_logits, class_a, class_b)
    ld_patched = compute_logit_difference(patched_logits, class_a, class_b)

    return ld_patched - ld_original


def compute_normalized_logit_diff(
    logits: torch.Tensor,
    class_a: int,
    class_b: int,
    temperature: float = 1.0
) -> float:
    """
    Compute temperature-normalized logit difference.

    Useful for comparing across models with different logit scales.

    Args:
        logits: Model logits
        class_a: First class
        class_b: Second class
        temperature: Softmax temperature (higher = softer distribution)

    Returns:
        Normalized logit difference
    """
    if logits.dim() == 2:
        logits = logits[0]

    # Normalize by temperature
    scaled_logits = logits / temperature

    return (scaled_logits[class_a] - scaled_logits[class_b]).item()


# =============================================================================
# KL DIVERGENCE
# =============================================================================

def compute_kl_divergence(
    original_probs: torch.Tensor,
    patched_probs: torch.Tensor,
    eps: float = 1e-10
) -> float:
    """
    Compute KL divergence between original and patched distributions.

    KL(P_original || P_patched) measures how much information is lost
    when using patched distribution to approximate original.

    This captures full distributional change, not just target class.

    Args:
        original_probs: Original probability distribution
        patched_probs: Probability distribution after patching
        eps: Small constant for numerical stability

    Returns:
        KL divergence (non-negative, 0 = identical distributions)

    Note:
        KL divergence is asymmetric: KL(P||Q) ≠ KL(Q||P)
    """
    if original_probs.dim() == 2:
        original_probs = original_probs[0]
    if patched_probs.dim() == 2:
        patched_probs = patched_probs[0]

    # Ensure valid probabilities
    original_probs = original_probs.clamp(min=eps)
    patched_probs = patched_probs.clamp(min=eps)

    # KL(original || patched) = sum(original * log(original / patched))
    kl = (original_probs * torch.log(original_probs / patched_probs)).sum()

    return kl.item()


def compute_symmetric_kl(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    eps: float = 1e-10
) -> float:
    """
    Compute symmetric KL divergence (Jensen-Shannon-like).

    Symmetric KL = 0.5 * (KL(A||B) + KL(B||A))

    Args:
        probs_a: First distribution
        probs_b: Second distribution
        eps: Numerical stability constant

    Returns:
        Symmetric KL divergence
    """
    kl_ab = compute_kl_divergence(probs_a, probs_b, eps)
    kl_ba = compute_kl_divergence(probs_b, probs_a, eps)

    return 0.5 * (kl_ab + kl_ba)


def compute_js_divergence(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    eps: float = 1e-10
) -> float:
    """
    Compute Jensen-Shannon divergence.

    JS(A||B) = 0.5 * KL(A||M) + 0.5 * KL(B||M)
    where M = 0.5 * (A + B)

    JS is always finite and symmetric.

    Args:
        probs_a: First distribution
        probs_b: Second distribution
        eps: Numerical stability constant

    Returns:
        JS divergence (bounded in [0, log(2)] ≈ [0, 0.693])
    """
    if probs_a.dim() == 2:
        probs_a = probs_a[0]
    if probs_b.dim() == 2:
        probs_b = probs_b[0]

    m = 0.5 * (probs_a + probs_b)

    kl_am = compute_kl_divergence(probs_a, m, eps)
    kl_bm = compute_kl_divergence(probs_b, m, eps)

    return 0.5 * (kl_am + kl_bm)


# =============================================================================
# CROSS-ENTROPY METRICS
# =============================================================================

def compute_cross_entropy(
    probs: torch.Tensor,
    true_label: int,
    eps: float = 1e-10
) -> float:
    """
    Compute cross-entropy loss for given true label.

    CE(y, p) = -log(p[y])

    Args:
        probs: Probability distribution
        true_label: Index of true class
        eps: Numerical stability

    Returns:
        Cross-entropy loss (lower = better prediction)
    """
    if probs.dim() == 2:
        probs = probs[0]

    return -torch.log(probs[true_label].clamp(min=eps)).item()


def compute_cross_entropy_change(
    original_probs: torch.Tensor,
    patched_probs: torch.Tensor,
    true_label: int
) -> float:
    """
    Compute change in cross-entropy due to patching.

    Negative change means patching improved prediction (reduced loss).

    Args:
        original_probs: Probabilities before patching
        patched_probs: Probabilities after patching
        true_label: True class index

    Returns:
        CE_patched - CE_original
        Negative = improvement, Positive = degradation
    """
    ce_original = compute_cross_entropy(original_probs, true_label)
    ce_patched = compute_cross_entropy(patched_probs, true_label)

    return ce_patched - ce_original


# =============================================================================
# PROBABILITY METRICS
# =============================================================================

def compute_probability_ratio(
    original_probs: torch.Tensor,
    patched_probs: torch.Tensor,
    target_class: int,
    eps: float = 1e-10
) -> float:
    """
    Compute probability ratio for target class.

    Ratio > 1 means patching increased probability of target class.

    Args:
        original_probs: Original distribution
        patched_probs: Patched distribution
        target_class: Class to measure
        eps: Numerical stability

    Returns:
        P(target|patched) / P(target|original)
    """
    if original_probs.dim() == 2:
        original_probs = original_probs[0]
    if patched_probs.dim() == 2:
        patched_probs = patched_probs[0]

    p_original = original_probs[target_class].clamp(min=eps).item()
    p_patched = patched_probs[target_class].clamp(min=eps).item()

    return p_patched / p_original


def compute_delta_probability(
    original_probs: torch.Tensor,
    patched_probs: torch.Tensor,
    target_class: int
) -> float:
    """
    Compute probability change for target class.

    This is the traditional ΔP metric.

    Args:
        original_probs: Original distribution
        patched_probs: Patched distribution
        target_class: Class to measure

    Returns:
        P(target|patched) - P(target|original)
    """
    if original_probs.dim() == 2:
        original_probs = original_probs[0]
    if patched_probs.dim() == 2:
        patched_probs = patched_probs[0]

    return (patched_probs[target_class] - original_probs[target_class]).item()


def compute_max_probability_change(
    original_probs: torch.Tensor,
    patched_probs: torch.Tensor
) -> Tuple[float, int]:
    """
    Find the class with maximum absolute probability change.

    Args:
        original_probs: Original distribution
        patched_probs: Patched distribution

    Returns:
        (max_change, class_index) tuple
    """
    if original_probs.dim() == 2:
        original_probs = original_probs[0]
    if patched_probs.dim() == 2:
        patched_probs = patched_probs[0]

    delta = patched_probs - original_probs
    max_idx = torch.argmax(torch.abs(delta)).item()
    max_change = delta[max_idx].item()

    return max_change, max_idx


# =============================================================================
# COMPREHENSIVE METRICS
# =============================================================================

def compute_all_metrics(
    original_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    true_class: int,
    corrupt_class: Optional[int] = None
) -> PatchingMetrics:
    """
    Compute all patching metrics at once.

    Args:
        original_logits: Logits before patching
        patched_logits: Logits after patching
        true_class: True (clean) class label
        corrupt_class: Corrupt class label (for logit diff)
                       If None, uses argmax of original logits

    Returns:
        PatchingMetrics dataclass with all metrics
    """
    # Compute probabilities
    original_probs = F.softmax(original_logits, dim=-1)
    patched_probs = F.softmax(patched_logits, dim=-1)

    # Determine corrupt class if not provided
    if corrupt_class is None:
        if original_probs.dim() == 2:
            corrupt_class = original_probs[0].argmax().item()
        else:
            corrupt_class = original_probs.argmax().item()

    # Compute all metrics
    delta_p = compute_delta_probability(original_probs, patched_probs, true_class)
    logit_diff = compute_logit_difference_change(
        original_logits, patched_logits, true_class, corrupt_class
    )
    kl_div = compute_kl_divergence(original_probs, patched_probs)
    ce_change = compute_cross_entropy_change(original_probs, patched_probs, true_class)
    prob_ratio = compute_probability_ratio(original_probs, patched_probs, true_class)

    return PatchingMetrics(
        delta_p=delta_p,
        logit_diff=logit_diff,
        kl_divergence=kl_div,
        cross_entropy_change=ce_change,
        probability_ratio=prob_ratio,
        original_logits=original_logits.detach().cpu().numpy() if isinstance(original_logits, torch.Tensor) else original_logits,
        patched_logits=patched_logits.detach().cpu().numpy() if isinstance(patched_logits, torch.Tensor) else patched_logits,
        original_probs=original_probs.detach().cpu().numpy() if isinstance(original_probs, torch.Tensor) else original_probs,
        patched_probs=patched_probs.detach().cpu().numpy() if isinstance(patched_probs, torch.Tensor) else patched_probs,
    )


def compute_metrics_for_sweep(
    model: nn.Module,
    clean_x: torch.Tensor,
    corrupt_x: torch.Tensor,
    true_class: int,
    num_classes: int,
    device: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Compute multiple metrics for a full head sweep.

    Instead of just returning probabilities, returns multiple
    metric matrices for richer analysis.

    Args:
        model: Trained model
        clean_x: Clean input
        corrupt_x: Corrupt input
        true_class: True class label
        num_classes: Number of classes
        device: Device to run on

    Returns:
        Dict with metric matrices:
            - 'delta_p': (n_layers, n_heads)
            - 'logit_diff': (n_layers, n_heads)
            - 'kl_divergence': (n_layers, n_heads)
            - 'cross_entropy': (n_layers, n_heads)
    """
    try:
        from .utils import sweep_heads, get_probs
    except ImportError:
        raise ImportError("Could not import sweep_heads from utils")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    clean_x = clean_x.to(device)
    corrupt_x = corrupt_x.to(device)

    # Get baseline
    with torch.no_grad():
        baseline_logits = model(corrupt_x)
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        corrupt_class = baseline_probs[0].argmax().item()

    # Run standard sweep (gets probabilities)
    patch_probs = sweep_heads(model, clean_x, corrupt_x, num_classes)

    n_layers, n_heads, _ = patch_probs.shape

    # Initialize metric matrices
    metrics = {
        'delta_p': np.zeros((n_layers, n_heads)),
        'logit_diff': np.zeros((n_layers, n_heads)),
        'kl_divergence': np.zeros((n_layers, n_heads)),
        'cross_entropy': np.zeros((n_layers, n_heads)),
    }

    # Compute metrics for each head
    baseline_probs_np = baseline_probs[0].cpu().numpy()

    for l in range(n_layers):
        for h in range(n_heads):
            patched_probs = torch.tensor(patch_probs[l, h])

            # Delta P
            metrics['delta_p'][l, h] = patch_probs[l, h, true_class] - baseline_probs_np[true_class]

            # KL divergence
            metrics['kl_divergence'][l, h] = compute_kl_divergence(
                torch.tensor(baseline_probs_np),
                patched_probs
            )

            # Cross-entropy change
            metrics['cross_entropy'][l, h] = compute_cross_entropy_change(
                torch.tensor(baseline_probs_np),
                patched_probs,
                true_class
            )

            # Logit diff (approximate from probs - not ideal but works without logits)
            # Use log-odds as proxy
            eps = 1e-10
            log_odds_baseline = np.log(baseline_probs_np[true_class] + eps) - np.log(baseline_probs_np[corrupt_class] + eps)
            log_odds_patched = np.log(patch_probs[l, h, true_class] + eps) - np.log(patch_probs[l, h, corrupt_class] + eps)
            metrics['logit_diff'][l, h] = log_odds_patched - log_odds_baseline

    return metrics


# =============================================================================
# METRIC ANALYSIS UTILITIES
# =============================================================================

def rank_heads_by_metric(
    metric_matrix: np.ndarray,
    top_k: int = 5
) -> List[Tuple[int, int, float]]:
    """
    Rank heads by a given metric matrix.

    Args:
        metric_matrix: (n_layers, n_heads) array of metric values
        top_k: Number of top heads to return

    Returns:
        List of (layer, head, value) tuples, sorted by |value|
    """
    flat = metric_matrix.flatten()
    top_indices = np.argsort(-np.abs(flat))[:top_k]

    n_heads = metric_matrix.shape[1]
    results = []
    for idx in top_indices:
        layer = idx // n_heads
        head = idx % n_heads
        results.append((layer, head, flat[idx]))

    return results


def compare_metric_rankings(
    metrics: Dict[str, np.ndarray],
    top_k: int = 5
) -> Dict[Tuple[str, str], float]:
    """
    Compare rankings from different metrics using Spearman correlation.

    Args:
        metrics: Dict of metric_name -> (n_layers, n_heads) arrays
        top_k: K for top-K overlap calculation

    Returns:
        Dict of (metric1, metric2) -> correlation
    """
    from scipy import stats

    names = list(metrics.keys())
    correlations = {}

    for i, m1 in enumerate(names):
        for j, m2 in enumerate(names):
            if i < j:
                flat1 = np.abs(metrics[m1].flatten())
                flat2 = np.abs(metrics[m2].flatten())
                rho, _ = stats.spearmanr(flat1, flat2)
                correlations[(m1, m2)] = rho

    return correlations


def print_metric_summary(metrics: Dict[str, np.ndarray], top_k: int = 5) -> None:
    """Pretty print summary of multiple metrics."""
    print("=" * 60)
    print("PATCHING METRICS SUMMARY")
    print("=" * 60)

    for name, matrix in metrics.items():
        print(f"\n{name.upper()}:")
        print(f"  Range: [{matrix.min():.4f}, {matrix.max():.4f}]")
        print(f"  Mean: {matrix.mean():.4f}, Std: {matrix.std():.4f}")

        top_heads = rank_heads_by_metric(matrix, top_k)
        print(f"  Top-{top_k} heads:")
        for layer, head, val in top_heads:
            print(f"    L{layer}H{head}: {val:+.4f}")

    # Print correlations
    correlations = compare_metric_rankings(metrics)
    if correlations:
        print("\nMetric Correlations (Spearman ρ):")
        for (m1, m2), rho in correlations.items():
            print(f"  {m1} vs {m2}: {rho:.3f}")
