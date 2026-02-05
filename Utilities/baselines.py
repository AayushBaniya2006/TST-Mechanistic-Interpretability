"""
Baseline comparison methods for activation patching validation.

This module provides alternative interpretability methods to compare
against activation patching, establishing statistical significance
and methodological validation.

Methods included:
- Random patching baseline (null distribution)
- Integrated Gradients (gradient-based attribution)
- Attention weight importance (raw attention analysis)
- Gradient × Input saliency
- Method comparison utilities

Usage:
    >>> from Utilities.baselines import (
    ...     random_patching_baseline,
    ...     integrated_gradients_importance,
    ...     attention_weight_importance,
    ...     compare_all_methods
    ... )
    >>>
    >>> # Establish null distribution
    >>> null_dist = random_patching_baseline(model, clean, corrupt, num_classes)
    >>>
    >>> # Compare methods
    >>> comparison = compare_all_methods(model, pairs, num_classes)

References:
    - Sundararajan et al. "Axiomatic Attribution for Deep Networks" (2017)
    - Zhang et al. "Towards Best Practices of Activation Patching" (ICLR 2024)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
import pandas as pd


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class BaselineResult:
    """Result from a baseline method."""
    method: str
    importance_matrix: np.ndarray  # Shape: (n_layers, n_heads)
    raw_values: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_ranking(self) -> np.ndarray:
        """Get flat ranking of head importance (highest = most important)."""
        flat = self.importance_matrix.flatten()
        return np.argsort(np.argsort(-flat))  # Rank from 0 (most important)

    def get_top_k(self, k: int = 5) -> List[Tuple[int, int, float]]:
        """Get top-k most important heads as (layer, head, value) tuples."""
        flat = self.importance_matrix.flatten()
        top_indices = np.argsort(-flat)[:k]
        results = []
        n_heads = self.importance_matrix.shape[1]
        for idx in top_indices:
            layer = idx // n_heads
            head = idx % n_heads
            results.append((layer, head, flat[idx]))
        return results


@dataclass
class MethodComparison:
    """Comparison of multiple interpretability methods."""
    methods: List[str]
    importance_matrices: Dict[str, np.ndarray]
    correlation_matrix: pd.DataFrame
    rankings: Dict[str, np.ndarray]
    top_k_overlap: Dict[Tuple[str, str], float]


# =============================================================================
# RANDOM PATCHING BASELINE
# =============================================================================

def random_patching_baseline(
    model: nn.Module,
    clean_x: torch.Tensor,
    corrupt_x: torch.Tensor,
    num_classes: int,
    n_permutations: int = 100,
    seed: Optional[int] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute null distribution via random activation patching.

    Patches random activations (from permuted samples) to establish
    what effect size we'd expect by chance. This provides statistical
    context for interpreting actual patching results.

    Args:
        model: Trained TimeSeriesTransformer
        clean_x: Clean input tensor, shape (1, seq_len, input_dim)
        corrupt_x: Corrupt input tensor, shape (1, seq_len, input_dim)
        num_classes: Number of output classes
        n_permutations: Number of random permutations
        seed: Random seed for reproducibility
        device: Device to run on (auto-detected if None)

    Returns:
        Dict containing:
            - null_distribution: np.ndarray of random patch effects
            - mean: Mean effect under null
            - std: Standard deviation under null
            - percentiles: 5th, 25th, 50th, 75th, 95th percentiles
            - p_value_threshold: Effect size needed for p < 0.05

    Example:
        >>> null = random_patching_baseline(model, clean, corrupt, 9, n_permutations=1000)
        >>> print(f"95th percentile: {null['percentiles'][95]:.4f}")
        >>> # Effects above this are statistically significant
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    clean_x = clean_x.to(device)
    corrupt_x = corrupt_x.to(device)

    # Get baseline probability
    with torch.no_grad():
        baseline_logits = model(corrupt_x)
        baseline_probs = F.softmax(baseline_logits, dim=-1)

    # Get model architecture info
    n_layers = len(model.transformer_encoder.layers)
    n_heads = model.transformer_encoder.layers[0].self_attn.num_heads

    # Collect random patch effects
    random_effects = []

    for _ in range(n_permutations):
        # Generate random activations by permuting along batch dimension
        # Since we only have one sample, we create noise-perturbed versions
        noise_scale = 0.1 * clean_x.std()
        random_clean = clean_x + torch.randn_like(clean_x) * noise_scale

        # Pick random layer and head
        rand_layer = np.random.randint(0, n_layers)
        rand_head = np.random.randint(0, n_heads)

        # Patch with random activation
        with torch.no_grad():
            patched_probs = _patch_single_head(
                model, random_clean, corrupt_x,
                rand_layer, rand_head, num_classes, device
            )

        # Compute effect (max absolute change)
        effect = torch.abs(patched_probs - baseline_probs).max().item()
        random_effects.append(effect)

    random_effects = np.array(random_effects)

    return {
        'null_distribution': random_effects,
        'mean': random_effects.mean(),
        'std': random_effects.std(),
        'percentiles': {
            5: np.percentile(random_effects, 5),
            25: np.percentile(random_effects, 25),
            50: np.percentile(random_effects, 50),
            75: np.percentile(random_effects, 75),
            95: np.percentile(random_effects, 95),
            99: np.percentile(random_effects, 99),
        },
        'p_value_threshold_05': np.percentile(random_effects, 95),
        'p_value_threshold_01': np.percentile(random_effects, 99),
    }


def compute_empirical_p_value(
    observed_effect: float,
    null_distribution: np.ndarray,
    alternative: str = 'greater'
) -> float:
    """
    Compute empirical p-value for observed effect against null distribution.

    Args:
        observed_effect: The observed patching effect
        null_distribution: Array of effects under null hypothesis
        alternative: 'greater', 'less', or 'two-sided'

    Returns:
        Empirical p-value
    """
    n = len(null_distribution)

    if alternative == 'greater':
        return (np.sum(null_distribution >= observed_effect) + 1) / (n + 1)
    elif alternative == 'less':
        return (np.sum(null_distribution <= observed_effect) + 1) / (n + 1)
    else:  # two-sided
        return (np.sum(np.abs(null_distribution) >= np.abs(observed_effect)) + 1) / (n + 1)


# =============================================================================
# INTEGRATED GRADIENTS
# =============================================================================

def integrated_gradients_importance(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50,
    device: Optional[str] = None,
    aggregate_to_heads: bool = True
) -> BaselineResult:
    """
    Compute head importance using Integrated Gradients.

    Integrated Gradients satisfies key axioms (sensitivity, implementation
    invariance) that make it a principled attribution method. We attribute
    importance to attention head outputs.

    Args:
        model: Trained TimeSeriesTransformer
        x: Input tensor, shape (1, seq_len, input_dim)
        target_class: Class to compute attribution for
        baseline: Baseline input (zeros if None)
        steps: Number of interpolation steps
        device: Device to run on
        aggregate_to_heads: Whether to aggregate gradients to head-level

    Returns:
        BaselineResult with head importance matrix

    References:
        Sundararajan et al. "Axiomatic Attribution for Deep Networks" (2017)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = x.to(device)

    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(device)

    # Get architecture info
    n_layers = len(model.transformer_encoder.layers)
    n_heads = model.transformer_encoder.layers[0].self_attn.num_heads
    d_model = model.transformer_encoder.layers[0].self_attn.embed_dim
    head_dim = d_model // n_heads

    # Storage for gradients
    head_attributions = np.zeros((n_layers, n_heads))

    # Interpolation path
    alphas = torch.linspace(0, 1, steps, device=device)

    for layer_idx in range(n_layers):
        # Hook to capture attention output gradients
        attention_grads = []

        def make_hook(grads_list):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grads_list.append(grad_output[0].detach())
            return hook

        layer = model.transformer_encoder.layers[layer_idx]
        handle = layer.self_attn.register_full_backward_hook(make_hook(attention_grads))

        integrated_grads = torch.zeros(1, x.shape[1], d_model, device=device)

        for alpha in alphas:
            # Interpolated input
            interp_x = baseline + alpha * (x - baseline)
            interp_x.requires_grad_(True)

            # Forward pass
            output = model(interp_x)

            # Backward pass for target class
            model.zero_grad()
            output[0, target_class].backward(retain_graph=True)

            if attention_grads:
                integrated_grads += attention_grads[-1]
                attention_grads.clear()

        handle.remove()

        # Average and multiply by (x - baseline) difference
        # For attention outputs, we use the gradient magnitude as importance
        integrated_grads = integrated_grads / steps

        if aggregate_to_heads:
            # Reshape to separate heads and aggregate
            ig_reshaped = integrated_grads.view(1, -1, n_heads, head_dim)
            head_importance = torch.abs(ig_reshaped).sum(dim=(1, 3)).squeeze()
            head_attributions[layer_idx] = head_importance.cpu().numpy()

    # Normalize to [0, 1]
    if head_attributions.max() > 0:
        head_attributions = head_attributions / head_attributions.max()

    return BaselineResult(
        method='integrated_gradients',
        importance_matrix=head_attributions,
        metadata={'steps': steps, 'target_class': target_class}
    )


def gradient_x_input_importance(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    device: Optional[str] = None
) -> BaselineResult:
    """
    Compute head importance using Gradient × Input saliency.

    Simple but fast gradient-based attribution method.

    Args:
        model: Trained TimeSeriesTransformer
        x: Input tensor, shape (1, seq_len, input_dim)
        target_class: Class to compute attribution for
        device: Device to run on

    Returns:
        BaselineResult with head importance matrix
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = x.to(device).requires_grad_(True)

    n_layers = len(model.transformer_encoder.layers)
    n_heads = model.transformer_encoder.layers[0].self_attn.num_heads
    d_model = model.transformer_encoder.layers[0].self_attn.embed_dim
    head_dim = d_model // n_heads

    head_attributions = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        layer = model.transformer_encoder.layers[layer_idx]

        # Storage for activation and gradient
        activation = [None]
        gradient = [None]

        def forward_hook(module, input, output):
            activation[0] = output[0] if isinstance(output, tuple) else output

        def backward_hook(module, grad_input, grad_output):
            gradient[0] = grad_output[0] if grad_output[0] is not None else None

        fwd_handle = layer.self_attn.register_forward_hook(forward_hook)
        bwd_handle = layer.self_attn.register_full_backward_hook(backward_hook)

        # Forward and backward
        output = model(x)
        model.zero_grad()
        output[0, target_class].backward()

        fwd_handle.remove()
        bwd_handle.remove()

        if activation[0] is not None and gradient[0] is not None:
            # Gradient × Activation
            saliency = activation[0] * gradient[0]
            saliency = saliency.view(1, -1, n_heads, head_dim)
            head_importance = torch.abs(saliency).sum(dim=(1, 3)).squeeze()
            head_attributions[layer_idx] = head_importance.detach().cpu().numpy()

    # Normalize
    if head_attributions.max() > 0:
        head_attributions = head_attributions / head_attributions.max()

    return BaselineResult(
        method='gradient_x_input',
        importance_matrix=head_attributions,
        metadata={'target_class': target_class}
    )


# =============================================================================
# ATTENTION WEIGHT IMPORTANCE
# =============================================================================

def attention_weight_importance(
    model: nn.Module,
    x: torch.Tensor,
    device: Optional[str] = None,
    aggregation: str = 'entropy'
) -> BaselineResult:
    """
    Compute head importance from raw attention weights.

    This baseline doesn't use gradients - it analyzes the attention
    patterns themselves to measure head "activity" or "focus".

    Args:
        model: Trained TimeSeriesTransformer
        x: Input tensor, shape (1, seq_len, input_dim)
        device: Device to run on
        aggregation: How to aggregate attention weights:
            - 'entropy': Higher entropy = more distributed attention
            - 'max': Maximum attention weight (spikiness)
            - 'variance': Variance across positions

    Returns:
        BaselineResult with head importance matrix
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = x.to(device)

    n_layers = len(model.transformer_encoder.layers)
    n_heads = model.transformer_encoder.layers[0].self_attn.num_heads

    head_importance = np.zeros((n_layers, n_heads))
    attention_patterns = {}

    with torch.no_grad():
        # First get embedded input
        embedded = model.conv1(x.transpose(1, 2))
        embedded = F.relu(embedded)
        embedded = model.conv2(embedded)
        embedded = F.relu(embedded)
        embedded = model.conv3(embedded)
        embedded = embedded.transpose(1, 2)

        # Add positional encoding
        embedded = embedded + model.pos_enc[:, :embedded.shape[1], :]

        current = embedded

        for layer_idx, layer in enumerate(model.transformer_encoder.layers):
            # Get attention weights
            # PyTorch's MultiheadAttention can return weights
            attn_output, attn_weights = layer.self_attn(
                current, current, current,
                need_weights=True,
                average_attn_weights=False  # Get per-head weights
            )

            # attn_weights shape: (batch, n_heads, seq_len, seq_len)
            attention_patterns[layer_idx] = attn_weights.cpu().numpy()

            # Aggregate based on method
            for head_idx in range(n_heads):
                head_attn = attn_weights[0, head_idx].cpu().numpy()

                if aggregation == 'entropy':
                    # Higher entropy = more distributed (potentially more "processing")
                    # Negative entropy so higher = more focused
                    eps = 1e-10
                    entropy = -np.sum(head_attn * np.log(head_attn + eps), axis=-1).mean()
                    head_importance[layer_idx, head_idx] = -entropy  # Invert: lower entropy = more focused

                elif aggregation == 'max':
                    # Maximum attention weight (how "focused" the head is)
                    head_importance[layer_idx, head_idx] = head_attn.max()

                elif aggregation == 'variance':
                    # Variance in attention (how variable the pattern is)
                    head_importance[layer_idx, head_idx] = head_attn.var()

                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")

            # Continue forward pass
            attn_output = layer.self_attn(current, current, current)[0]
            current = layer.norm1(current + layer.dropout1(attn_output))
            ff_output = layer.linear2(layer.dropout(F.relu(layer.linear1(current))))
            current = layer.norm2(current + layer.dropout2(ff_output))

    # Normalize to [0, 1]
    if head_importance.max() > head_importance.min():
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    return BaselineResult(
        method=f'attention_{aggregation}',
        importance_matrix=head_importance,
        raw_values=attention_patterns,
        metadata={'aggregation': aggregation}
    )


# =============================================================================
# METHOD COMPARISON
# =============================================================================

def compare_all_methods(
    model: nn.Module,
    clean_x: torch.Tensor,
    corrupt_x: torch.Tensor,
    true_label: int,
    num_classes: int,
    device: Optional[str] = None,
    include_random: bool = True,
    n_random_permutations: int = 100
) -> MethodComparison:
    """
    Run all baseline methods and compare to activation patching.

    Computes correlation between method rankings to understand
    how different interpretability approaches relate.

    Args:
        model: Trained TimeSeriesTransformer
        clean_x: Clean input tensor
        corrupt_x: Corrupt input tensor
        true_label: True class label
        num_classes: Number of classes
        device: Device to run on
        include_random: Whether to include random baseline
        n_random_permutations: Permutations for random baseline

    Returns:
        MethodComparison with correlation matrix and rankings
    """
    if device is None:
        device = next(model.parameters()).device

    results = {}

    # 1. Activation patching (using sweep_heads from utils)
    try:
        from .utils import sweep_heads, get_probs

        patch_probs = sweep_heads(model, clean_x, corrupt_x, num_classes)
        baseline_probs = get_probs(model, corrupt_x)

        # Compute delta_p for true label
        delta_p = patch_probs[:, :, true_label] - baseline_probs[true_label].item()
        results['activation_patching'] = delta_p
    except ImportError:
        warnings.warn("Could not import sweep_heads from utils")
        # Create placeholder
        n_layers = len(model.transformer_encoder.layers)
        n_heads = model.transformer_encoder.layers[0].self_attn.num_heads
        results['activation_patching'] = np.zeros((n_layers, n_heads))

    # 2. Integrated Gradients
    ig_result = integrated_gradients_importance(
        model, clean_x, true_label, device=device
    )
    results['integrated_gradients'] = ig_result.importance_matrix

    # 3. Gradient × Input
    gxi_result = gradient_x_input_importance(
        model, clean_x, true_label, device=device
    )
    results['gradient_x_input'] = gxi_result.importance_matrix

    # 4. Attention weights (entropy-based)
    attn_result = attention_weight_importance(
        model, clean_x, device=device, aggregation='entropy'
    )
    results['attention_entropy'] = attn_result.importance_matrix

    # 5. Attention weights (max-based)
    attn_max_result = attention_weight_importance(
        model, clean_x, device=device, aggregation='max'
    )
    results['attention_max'] = attn_max_result.importance_matrix

    # Compute rankings
    rankings = {}
    for method, importance in results.items():
        flat = importance.flatten()
        rankings[method] = np.argsort(np.argsort(-flat))

    # Compute correlation matrix
    methods = list(results.keys())
    n_methods = len(methods)
    corr_matrix = np.zeros((n_methods, n_methods))

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                rho, _ = stats.spearmanr(
                    results[m1].flatten(),
                    results[m2].flatten()
                )
                corr_matrix[i, j] = rho

    corr_df = pd.DataFrame(corr_matrix, index=methods, columns=methods)

    # Compute top-K overlap for all pairs
    top_k_overlap = {}
    k = 5
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                top1 = set(np.argsort(-results[m1].flatten())[:k])
                top2 = set(np.argsort(-results[m2].flatten())[:k])
                overlap = len(top1 & top2) / len(top1 | top2)
                top_k_overlap[(m1, m2)] = overlap

    return MethodComparison(
        methods=methods,
        importance_matrices=results,
        correlation_matrix=corr_df,
        rankings=rankings,
        top_k_overlap=top_k_overlap
    )


def compare_methods_across_pairs(
    model: nn.Module,
    pairs: List[Tuple[torch.Tensor, torch.Tensor, int]],
    num_classes: int,
    device: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare method correlations across multiple sample pairs.

    Args:
        model: Trained model
        pairs: List of (clean_x, corrupt_x, true_label) tuples
        num_classes: Number of classes
        device: Device to run on

    Returns:
        DataFrame with mean correlations and confidence intervals
    """
    all_correlations = []

    for clean_x, corrupt_x, true_label in pairs:
        comparison = compare_all_methods(
            model, clean_x, corrupt_x, true_label,
            num_classes, device, include_random=False
        )

        # Extract patching vs other methods correlations
        corr_row = {}
        for method in comparison.methods:
            if method != 'activation_patching':
                corr_row[f'patching_vs_{method}'] = \
                    comparison.correlation_matrix.loc['activation_patching', method]

        all_correlations.append(corr_row)

    df = pd.DataFrame(all_correlations)

    # Compute summary statistics
    summary = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'ci_lower': df.mean() - 1.96 * df.std() / np.sqrt(len(df)),
        'ci_upper': df.mean() + 1.96 * df.std() / np.sqrt(len(df))
    })

    return summary


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _patch_single_head(
    model: nn.Module,
    clean_x: torch.Tensor,
    corrupt_x: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    num_classes: int,
    device: str
) -> torch.Tensor:
    """
    Patch a single attention head and return resulting probabilities.

    Internal helper function for random baseline.
    """
    n_heads = model.transformer_encoder.layers[0].self_attn.num_heads
    d_model = model.transformer_encoder.layers[0].self_attn.embed_dim
    head_dim = d_model // n_heads

    # Get clean activations
    clean_activations = {}

    def cache_hook(name):
        def hook(module, input, output):
            clean_activations[name] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
        return hook

    handles = []
    for idx, layer in enumerate(model.transformer_encoder.layers):
        handle = layer.self_attn.register_forward_hook(cache_hook(f'layer_{idx}'))
        handles.append(handle)

    with torch.no_grad():
        _ = model(clean_x)

    for h in handles:
        h.remove()

    # Patch during corrupt forward pass
    def patch_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        patched = out.clone()

        # Reshape to access heads
        batch, seq, _ = patched.shape
        patched_reshaped = patched.view(batch, seq, n_heads, head_dim)
        clean_reshaped = clean_activations[f'layer_{layer_idx}'].view(batch, seq, n_heads, head_dim)

        # Patch specific head
        patched_reshaped[:, :, head_idx, :] = clean_reshaped[:, :, head_idx, :]

        if isinstance(output, tuple):
            return (patched_reshaped.view(batch, seq, d_model),) + output[1:]
        return patched_reshaped.view(batch, seq, d_model)

    handle = model.transformer_encoder.layers[layer_idx].self_attn.register_forward_hook(patch_hook)

    with torch.no_grad():
        patched_logits = model(corrupt_x)
        patched_probs = F.softmax(patched_logits, dim=-1)

    handle.remove()

    return patched_probs


# =============================================================================
# NULL DISTRIBUTION FROM EXISTING RESULTS
# =============================================================================

def compute_null_distribution_from_results(
    all_delta_p: np.ndarray,
    n_permutations: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute null distribution by permuting existing ΔP values.

    This provides a baseline without needing to re-run the model.
    Under the null hypothesis, patching should have no systematic effect,
    so we permute the observed effects and compute the distribution.

    Args:
        all_delta_p: Array of all observed ΔP values across heads/pairs
        n_permutations: Number of permutations
        seed: Random seed

    Returns:
        Dict with null distribution statistics and thresholds
    """
    if seed is not None:
        np.random.seed(seed)

    observed_mean = np.mean(all_delta_p)
    observed_max = np.max(all_delta_p)

    # Generate null by shuffling signs (if effect is real, sign matters)
    null_means = []
    null_maxes = []

    for _ in range(n_permutations):
        # Randomly flip signs - under null, positive and negative effects are equally likely
        signs = np.random.choice([-1, 1], size=len(all_delta_p))
        permuted = all_delta_p * signs
        null_means.append(np.mean(permuted))
        null_maxes.append(np.max(np.abs(permuted)))

    null_means = np.array(null_means)
    null_maxes = np.array(null_maxes)

    # Compute p-values
    p_mean = np.mean(null_means >= observed_mean)
    p_max = np.mean(null_maxes >= observed_max)

    return {
        'observed_mean': observed_mean,
        'observed_max': observed_max,
        'null_means': null_means,
        'null_maxes': null_maxes,
        'p_value_mean': p_mean,
        'p_value_max': p_max,
        'threshold_05_mean': np.percentile(null_means, 95),
        'threshold_01_mean': np.percentile(null_means, 99),
        'threshold_05_max': np.percentile(null_maxes, 95),
        'threshold_01_max': np.percentile(null_maxes, 99),
        'significant_05': p_mean < 0.05,
        'significant_01': p_mean < 0.01,
    }


def validate_effect_against_null(
    observed_effect: float,
    null_distribution: np.ndarray,
    alternative: str = 'greater'
) -> Dict[str, Any]:
    """
    Validate an observed effect against a null distribution.

    Args:
        observed_effect: The observed patching effect
        null_distribution: Array of effects under null
        alternative: 'greater', 'less', or 'two-sided'

    Returns:
        Dict with p-value, percentile, and significance
    """
    n = len(null_distribution)

    if alternative == 'greater':
        p_value = (np.sum(null_distribution >= observed_effect) + 1) / (n + 1)
        percentile = np.mean(null_distribution < observed_effect) * 100
    elif alternative == 'less':
        p_value = (np.sum(null_distribution <= observed_effect) + 1) / (n + 1)
        percentile = np.mean(null_distribution > observed_effect) * 100
    else:  # two-sided
        p_value = (np.sum(np.abs(null_distribution) >= np.abs(observed_effect)) + 1) / (n + 1)
        percentile = np.mean(np.abs(null_distribution) < np.abs(observed_effect)) * 100

    return {
        'observed': observed_effect,
        'p_value': p_value,
        'percentile': percentile,
        'null_mean': np.mean(null_distribution),
        'null_std': np.std(null_distribution),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'effect_vs_null': f"{percentile:.1f}th percentile of null (p={p_value:.4f})"
    }


def print_method_comparison(comparison: MethodComparison) -> None:
    """Pretty print method comparison results."""
    print("=" * 60)
    print("METHOD COMPARISON RESULTS")
    print("=" * 60)

    print("\nCorrelation Matrix (Spearman ρ):")
    print("-" * 40)
    print(comparison.correlation_matrix.round(3).to_string())

    print("\nTop-5 Overlap (Jaccard Similarity):")
    print("-" * 40)
    for (m1, m2), overlap in comparison.top_k_overlap.items():
        print(f"  {m1} vs {m2}: {overlap:.3f}")

    print("\nTop-5 Heads per Method:")
    print("-" * 40)
    for method, importance in comparison.importance_matrices.items():
        flat = importance.flatten()
        top_5 = np.argsort(-flat)[:5]
        n_heads = importance.shape[1]
        heads = [(idx // n_heads, idx % n_heads) for idx in top_5]
        print(f"  {method}: {heads}")
