"""
TST Mechanistic Interpretability Utilities

A package for analyzing Transformer-based time series classifiers
through activation patching and causal analysis.

Modules:
    utils: Core patching functions and visualization
    perturbations: Label-preserving input perturbations
    stability_metrics: Metrics for comparing mechanism stability
    statistics: Statistical analysis (CIs, effect sizes, significance tests)
    metrics: Additional patching metrics (logit diff, KL divergence)
    baselines: Baseline comparison methods (random, IG, attention)
    config: Configuration management and reproducibility
    TST_trainer: Model architecture and training utilities

Example:
    >>> from Utilities import sweep_heads, get_probs, TimeSeriesTransformer
    >>> from Utilities import gaussian_noise, head_rank_correlation
    >>> from Utilities.config import ExperimentConfig, set_all_seeds
    >>> from Utilities.statistics import compute_confidence_interval
    >>>
    >>> # Set up reproducible experiment
    >>> config = ExperimentConfig(seed=42)
    >>> set_all_seeds(config.seed)
    >>>
    >>> # Load model and run patching analysis
    >>> probs = sweep_heads(model, clean_x, corrupt_x, num_classes=9)
    >>> baseline = get_probs(model, corrupt_x)
    >>>
    >>> # Test stability under perturbation
    >>> perturbed = gaussian_noise(clean_x, sigma=0.1)
    >>> probs_pert = sweep_heads(model, perturbed, corrupt_x, num_classes=9)
    >>> rho = head_rank_correlation(probs[:,:,label], probs_pert[:,:,label])
    >>>
    >>> # Compute confidence intervals
    >>> ci = compute_confidence_interval(results, confidence=0.95)
"""

__version__ = "2.0.0"

# Core patching functions
from .utils import (
    # Probability and caching
    get_probs,
    run_and_cache,
    patch_activations,
    get_encoder_inputs,
    get_attention_saliency,

    # Single-component patching
    patch_attention_head,
    patch_all_heads_in_layer,
    patch_mlp_activation,
    patch_attention_head_at_position,
    patch_mlp_at_position,
    patch_multiple_attention_heads_positions,

    # Context manager
    with_head_patch,

    # Sweep functions (main API)
    sweep_heads,
    sweep_layerwise_patch,
    sweep_mlp_layers,
    sweep_attention_head_positions,
    sweep_mlp_positions,

    # Head-to-head analysis
    capture_all_heads,
    sweep_head_to_head_influence,
    sweep_head_to_output_deltas,

    # Causal graph construction
    find_critical_patches,
    build_causal_graph,

    # Visualization
    plot_influence,
    plot_layerwise_influence,
    plot_mlp_influence,
    plot_mini_deltas,
    plot_timeseries_with_attention_overlay,
    plot_head_position_patch_heatmap,
    plot_mlp_position_patch_heatmap,
    plot_causal_graph,
    plot_structured_graph_with_heads,

    # Utility classes
    FigureHolder,
)

# Perturbation functions
from .perturbations import (
    gaussian_noise,
    time_warp,
    phase_shift,
    apply_perturbation,
    validate_perturbation,
    get_perturbation_configs,
)

# Stability metrics
from .stability_metrics import (
    get_head_importance,
    get_head_ranking,
    head_rank_correlation,
    topk_overlap,
    patch_recovery_delta,
    mechanism_stability_score,
    compute_all_metrics,
    plot_importance_comparison,
    plot_correlation_decay,
    plot_topk_bars,
    create_summary_table,
)

# Model and training
from .TST_trainer import (
    TimeSeriesTransformer,
    Trainer,
    load_dataset,
)

# Configuration and reproducibility
from .config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    PatchingConfig,
    PerturbationConfig,
    StatisticsConfig,
    set_all_seeds,
    get_seed_state,
    set_seed_state,
    get_system_info,
    save_experiment_metadata,
    get_dataset_config,
    get_default_config,
    get_quick_test_config,
    get_full_experiment_config,
    DATASET_CONFIGS,
)

# Statistical analysis
from .statistics import (
    compute_confidence_interval,
    compute_correlation_ci,
    compute_effect_size,
    significance_test,
    apply_fdr_correction,
    power_analysis,
    compute_variance_decomposition,
    compute_aggregate_statistics,
    format_result_table,
    ConfidenceInterval,
    EffectSize,
    SignificanceResult,
)

# Additional metrics
from .metrics import (
    compute_logit_difference,
    compute_logit_difference_change,
    compute_normalized_logit_diff,
    compute_kl_divergence,
    compute_symmetric_kl,
    compute_js_divergence,
    compute_cross_entropy,
    compute_cross_entropy_change,
    compute_probability_ratio,
    compute_delta_probability,
    compute_max_probability_change,
    compute_all_metrics as compute_patching_metrics,
    compute_metrics_for_sweep,
    rank_heads_by_metric,
    compare_metric_rankings,
    print_metric_summary,
    PatchingMetrics,
)

# Baseline comparison methods
from .baselines import (
    random_patching_baseline,
    compute_empirical_p_value,
    integrated_gradients_importance,
    gradient_x_input_importance,
    attention_weight_importance,
    compare_all_methods,
    compare_methods_across_pairs,
    print_method_comparison,
    BaselineResult,
    MethodComparison,
)

__all__ = [
    # Version
    "__version__",

    # Core patching
    "get_probs",
    "run_and_cache",
    "patch_activations",
    "get_encoder_inputs",
    "get_attention_saliency",
    "patch_attention_head",
    "patch_all_heads_in_layer",
    "patch_mlp_activation",
    "patch_attention_head_at_position",
    "patch_mlp_at_position",
    "patch_multiple_attention_heads_positions",
    "with_head_patch",

    # Sweeps
    "sweep_heads",
    "sweep_layerwise_patch",
    "sweep_mlp_layers",
    "sweep_attention_head_positions",
    "sweep_mlp_positions",
    "capture_all_heads",
    "sweep_head_to_head_influence",
    "sweep_head_to_output_deltas",

    # Causal graphs
    "find_critical_patches",
    "build_causal_graph",

    # Visualization
    "plot_influence",
    "plot_layerwise_influence",
    "plot_mlp_influence",
    "plot_mini_deltas",
    "plot_timeseries_with_attention_overlay",
    "plot_head_position_patch_heatmap",
    "plot_mlp_position_patch_heatmap",
    "plot_causal_graph",
    "plot_structured_graph_with_heads",
    "FigureHolder",

    # Perturbations
    "gaussian_noise",
    "time_warp",
    "phase_shift",
    "apply_perturbation",
    "validate_perturbation",
    "get_perturbation_configs",

    # Stability metrics
    "get_head_importance",
    "get_head_ranking",
    "head_rank_correlation",
    "topk_overlap",
    "patch_recovery_delta",
    "mechanism_stability_score",
    "compute_all_metrics",
    "plot_importance_comparison",
    "plot_correlation_decay",
    "plot_topk_bars",
    "create_summary_table",

    # Model and training
    "TimeSeriesTransformer",
    "Trainer",
    "load_dataset",

    # Configuration
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "PatchingConfig",
    "PerturbationConfig",
    "StatisticsConfig",
    "set_all_seeds",
    "get_seed_state",
    "set_seed_state",
    "get_system_info",
    "save_experiment_metadata",
    "get_dataset_config",
    "get_default_config",
    "get_quick_test_config",
    "get_full_experiment_config",
    "DATASET_CONFIGS",

    # Statistics
    "compute_confidence_interval",
    "compute_correlation_ci",
    "compute_effect_size",
    "significance_test",
    "apply_fdr_correction",
    "power_analysis",
    "compute_variance_decomposition",
    "compute_aggregate_statistics",
    "format_result_table",
    "ConfidenceInterval",
    "EffectSize",
    "SignificanceResult",

    # Metrics
    "compute_logit_difference",
    "compute_logit_difference_change",
    "compute_normalized_logit_diff",
    "compute_kl_divergence",
    "compute_symmetric_kl",
    "compute_js_divergence",
    "compute_cross_entropy",
    "compute_cross_entropy_change",
    "compute_probability_ratio",
    "compute_delta_probability",
    "compute_max_probability_change",
    "compute_patching_metrics",
    "compute_metrics_for_sweep",
    "rank_heads_by_metric",
    "compare_metric_rankings",
    "print_metric_summary",
    "PatchingMetrics",

    # Baselines
    "random_patching_baseline",
    "compute_empirical_p_value",
    "integrated_gradients_importance",
    "gradient_x_input_importance",
    "attention_weight_importance",
    "compare_all_methods",
    "compare_methods_across_pairs",
    "print_method_comparison",
    "BaselineResult",
    "MethodComparison",
]
