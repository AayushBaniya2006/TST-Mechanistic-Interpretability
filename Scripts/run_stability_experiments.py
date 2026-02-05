#!/usr/bin/env python3
"""
Stability Stress-Testing Experiments for Mechanistic Interpretability.

This script runs the Phase II experiments from RESEARCH_EXTENSION_PLAN.md:
- Applies perturbations (gaussian noise, time warp, phase shift) to clean inputs
- Validates perturbations preserve model predictions
- Computes stability metrics comparing baseline vs perturbed head importance
- Generates visualizations and summary reports

Usage:
    python Scripts/run_stability_experiments.py --dataset JapaneseVowels
    python Scripts/run_stability_experiments.py --dataset all
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aeon.datasets import load_classification
from Utilities.TST_trainer import TimeSeriesTransformer
from Utilities.utils import sweep_heads, get_probs
from Utilities.perturbations import (
    gaussian_noise, time_warp, phase_shift,
    get_perturbation_configs
)
from Utilities.stability_metrics import (
    get_head_importance, compute_all_metrics,
    plot_importance_comparison, create_summary_table
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = ["JapaneseVowels", "PenDigits", "LSST"]

DATASET_CONFIG = {
    "JapaneseVowels": {
        "model_path": "TST_models/TST_japanesevowels.pth",
        "num_classes": 10,  # Labels are 1-9, so max+1=10
        "seq_len": 25,
        "input_dim": 12
    },
    "PenDigits": {
        "model_path": "TST_models/TST_pendigits.pth",
        "num_classes": 10,  # Labels are 0-9
        "seq_len": 8,
        "input_dim": 2
    },
    "LSST": {
        "model_path": "TST_models/TST_lsst.pth",
        "num_classes": 96,  # Labels go up to 95
        "seq_len": 36,
        "input_dim": 6
    }
}

# Max accuracy drop allowed for valid perturbation
MAX_ACCURACY_DROP = 0.05


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_model(dataset_name: str, device: torch.device) -> TimeSeriesTransformer:
    """Load pre-trained model for dataset."""
    config = DATASET_CONFIG[dataset_name]
    model = TimeSeriesTransformer(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        seq_len=config["seq_len"],
        d_model=128,
        n_head=8,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1
    ).to(device)

    model_path = PROJECT_ROOT / config["model_path"]
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def load_test_data(dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess test data."""
    X_test, y_test = load_classification(dataset_name, split="test")

    # Preprocess: (N, channels, seq_len) -> (N, seq_len, channels)
    X_test = X_test.astype(np.float32)
    X_test = np.swapaxes(X_test, 1, 2)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test.astype(np.int64))

    return X_test, y_test


def get_existing_pairs(dataset_name: str) -> List[Dict]:
    """
    Find all existing baseline pairs from Results/{dataset}/denoise/.

    Returns list of dicts with:
        - pair_dir: Path to the pair directory
        - class_label: The true class label
        - src_idx: Source (clean) sample index
        - tgt_idx: Target (corrupt) sample index
    """
    results_dir = PROJECT_ROOT / "Results" / dataset_name / "denoise"
    pairs = []

    if not results_dir.exists():
        print(f"Warning: No baseline results found at {results_dir}")
        return pairs

    for class_dir in results_dir.iterdir():
        if not class_dir.is_dir() or not class_dir.name.startswith("class_"):
            continue

        class_label = int(class_dir.name.split("_")[1])

        for pair_dir in class_dir.iterdir():
            if not pair_dir.is_dir() or not pair_dir.name.startswith("pair_"):
                continue

            # Parse pair_src_tgt format
            parts = pair_dir.name.split("_")
            src_idx = int(parts[1])
            tgt_idx = int(parts[2])

            # Check raw_results.npz exists
            if (pair_dir / "raw_results.npz").exists():
                pairs.append({
                    "pair_dir": pair_dir,
                    "class_label": class_label,
                    "src_idx": src_idx,
                    "tgt_idx": tgt_idx
                })

    return pairs


def apply_perturbation_to_sample(
    X: torch.Tensor,
    method: str,
    params: Dict,
    seed: int = 42
) -> torch.Tensor:
    """Apply a perturbation to a single sample."""
    if method == "gaussian":
        return gaussian_noise(X, sigma=params["sigma"], seed=seed)
    elif method == "time_warp":
        return time_warp(X, warp_factor=params["warp_factor"], seed=seed)
    elif method == "phase_shift":
        return phase_shift(X, max_shift=params["max_shift"], seed=seed)
    else:
        raise ValueError(f"Unknown perturbation method: {method}")


def validate_single_prediction(
    model: torch.nn.Module,
    X_orig: torch.Tensor,
    X_pert: torch.Tensor,
    true_label: int,
    device: torch.device
) -> Tuple[bool, float, float]:
    """
    Check if perturbation preserves the prediction for a single sample.

    Returns:
        (is_valid, orig_prob, pert_prob)
    """
    model.eval()
    X_orig = X_orig.to(device)
    X_pert = X_pert.to(device)

    with torch.no_grad():
        # Original prediction
        logits_orig = model(X_orig)
        probs_orig = torch.softmax(logits_orig, dim=1)[0]
        pred_orig = probs_orig.argmax().item()

        # Perturbed prediction
        logits_pert = model(X_pert)
        probs_pert = torch.softmax(logits_pert, dim=1)[0]
        pred_pert = probs_pert.argmax().item()

    # Check if prediction is preserved
    is_valid = (pred_orig == pred_pert) and (pred_orig == true_label)

    return is_valid, probs_orig[true_label].item(), probs_pert[true_label].item()


def get_perturbation_name(method: str, params: Dict) -> str:
    """Generate a readable name for the perturbation configuration."""
    if method == "gaussian":
        return f"gaussian_sigma_{params['sigma']:.2f}"
    elif method == "time_warp":
        return f"time_warp_factor_{params['warp_factor']:.2f}"
    elif method == "phase_shift":
        return f"phase_shift_max_{params['max_shift']}"
    return f"{method}_{list(params.values())[0]}"


def get_output_dir(method: str, params: Dict) -> str:
    """Generate output directory name for perturbation config."""
    if method == "gaussian":
        return f"sigma_{params['sigma']:.2f}"
    elif method == "time_warp":
        return f"factor_{params['warp_factor']:.2f}"
    elif method == "phase_shift":
        return f"shift_{params['max_shift']}"
    return f"{list(params.values())[0]}"


# =============================================================================
# MAIN EXPERIMENT FUNCTIONS
# =============================================================================

def run_stability_experiment_for_pair(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    pair_info: Dict,
    baseline_data: Dict,
    device: torch.device,
    output_dir: Path,
    num_classes: int
) -> Dict[str, Dict]:
    """
    Run stability experiments for a single pair across all perturbations.

    Returns dict mapping perturbation_name -> metrics
    """
    src_idx = pair_info["src_idx"]
    true_label = pair_info["class_label"]

    # Get clean sample (source)
    clean_x = X_test[[src_idx]].to(device)

    # Get corrupt sample for patching baseline
    tgt_idx = pair_info["tgt_idx"]
    corrupt_x = X_test[[tgt_idx]].to(device)

    # Compute baseline importance from stored data
    baseline_probs = baseline_data["baseline"]
    baseline_head_patch = baseline_data["head_patch"]
    baseline_importance = get_head_importance(baseline_head_patch, baseline_probs, true_label)

    # Results storage
    results = {}
    perturbation_configs = get_perturbation_configs()

    for method, configs in perturbation_configs.items():
        method_dir = output_dir / method

        for params in configs:
            config_name = get_perturbation_name(method, params)
            config_dir = method_dir / get_output_dir(method, params)
            config_dir.mkdir(parents=True, exist_ok=True)

            # Apply perturbation to clean sample
            perturbed_clean = apply_perturbation_to_sample(
                clean_x.cpu(), method, params, seed=42
            ).to(device)

            # Validate prediction is preserved
            is_valid, orig_prob, pert_prob = validate_single_prediction(
                model, clean_x, perturbed_clean, true_label, device
            )

            if not is_valid:
                # Skip this perturbation - prediction not preserved
                results[config_name] = {
                    "valid": False,
                    "reason": "prediction_changed",
                    "orig_prob": orig_prob,
                    "pert_prob": pert_prob
                }
                continue

            # Run sweep_heads on perturbed clean vs corrupt
            perturbed_probs = sweep_heads(model, perturbed_clean, corrupt_x, num_classes)
            perturbed_importance = get_head_importance(
                perturbed_probs,
                get_probs(model, corrupt_x),  # Baseline from corrupt
                true_label
            )

            # Compute stability metrics
            metrics = compute_all_metrics(baseline_importance, perturbed_importance)
            metrics["valid"] = True
            metrics["orig_prob"] = orig_prob
            metrics["pert_prob"] = pert_prob

            # Save raw results
            np.savez(
                config_dir / "raw_results.npz",
                baseline_importance=baseline_importance,
                perturbed_importance=perturbed_importance,
                baseline_probs=baseline_probs,
                perturbed_probs=perturbed_probs
            )

            # Generate comparison heatmap
            fig = plot_importance_comparison(
                baseline_importance,
                perturbed_importance,
                title=f"Head Importance: Baseline vs {config_name}",
                save_path=str(config_dir / "head_comparison_heatmap.png")
            )
            plt.close(fig)

            # Save metrics
            with open(config_dir / "stability_metrics.json", "w") as f:
                # Convert numpy types to Python types for JSON
                metrics_json = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics.items()
                    if not isinstance(v, dict)
                }
                if "recovery_delta" in metrics:
                    metrics_json["recovery_delta"] = {
                        k: float(v) for k, v in metrics["recovery_delta"].items()
                    }
                json.dump(metrics_json, f, indent=2)

            results[config_name] = metrics

    return results


def run_stability_experiments(
    dataset_name: str,
    device: torch.device,
    verbose: bool = True
) -> Dict:
    """
    Run full stability experiments for a dataset.

    Returns aggregate results across all pairs.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running stability experiments for {dataset_name}")
        print(f"{'='*60}")

    # Load model and data
    model = load_model(dataset_name, device)
    X_test, y_test = load_test_data(dataset_name)
    num_classes = DATASET_CONFIG[dataset_name]["num_classes"]

    # Get existing pairs
    pairs = get_existing_pairs(dataset_name)
    if not pairs:
        print(f"No baseline pairs found for {dataset_name}. Skipping.")
        return {}

    if verbose:
        print(f"Found {len(pairs)} baseline pairs to analyze")

    # Create output directory
    stability_dir = PROJECT_ROOT / "Results" / "Stability" / dataset_name
    stability_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate results
    all_results = defaultdict(list)
    pair_results = []

    for i, pair_info in enumerate(pairs):
        if verbose:
            print(f"\nProcessing pair {i+1}/{len(pairs)}: "
                  f"class_{pair_info['class_label']}/pair_{pair_info['src_idx']}_{pair_info['tgt_idx']}")

        # Load baseline data
        baseline_path = pair_info["pair_dir"] / "raw_results.npz"
        baseline_data = dict(np.load(baseline_path, allow_pickle=True))

        # Create output directory for this pair
        pair_output_dir = (stability_dir /
                          f"class_{pair_info['class_label']}" /
                          f"pair_{pair_info['src_idx']}_{pair_info['tgt_idx']}")

        # Save baseline info
        baseline_dir = pair_output_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        # Extract and save baseline head importance ranking
        baseline_probs = baseline_data["baseline"]
        baseline_head_patch = baseline_data["head_patch"]
        baseline_importance = get_head_importance(
            baseline_head_patch, baseline_probs, pair_info["class_label"]
        )

        np.savetxt(
            baseline_dir / "head_importance_ranking.csv",
            baseline_importance,
            delimiter=",",
            header="Head importance (L x H matrix)"
        )

        with open(baseline_dir / "accuracy.txt", "w") as f:
            f.write(f"Baseline P(true): {baseline_probs[pair_info['class_label']]:.4f}\n")

        # Run experiments
        results = run_stability_experiment_for_pair(
            model, X_test, y_test,
            pair_info, baseline_data,
            device, pair_output_dir,
            num_classes
        )

        # Aggregate
        for config_name, metrics in results.items():
            if metrics.get("valid", False):
                all_results[config_name].append(metrics)

        pair_results.append({
            "pair": f"class_{pair_info['class_label']}/pair_{pair_info['src_idx']}_{pair_info['tgt_idx']}",
            **{k: v.get("rank_correlation", None) if isinstance(v, dict) else None
               for k, v in results.items()}
        })

    # Compute aggregate statistics
    aggregate_metrics = {}
    for config_name, metrics_list in all_results.items():
        if not metrics_list:
            continue

        # Average across valid pairs
        aggregate_metrics[config_name] = {
            "n_valid": len(metrics_list),
            "rank_correlation_mean": np.mean([m["rank_correlation"] for m in metrics_list]),
            "rank_correlation_std": np.std([m["rank_correlation"] for m in metrics_list]),
            "topk_overlap_k3_mean": np.mean([m["topk_overlap_k3"] for m in metrics_list]),
            "topk_overlap_k5_mean": np.mean([m["topk_overlap_k5"] for m in metrics_list]),
            "topk_overlap_k10_mean": np.mean([m["topk_overlap_k10"] for m in metrics_list]),
            "stability_score_mean": np.mean([m["stability_score"] for m in metrics_list]),
            "stability_score_std": np.std([m["stability_score"] for m in metrics_list]),
        }

    # Save aggregate results
    summary_dir = stability_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    if aggregate_metrics:
        df = pd.DataFrame([
            {"perturbation": k, **v}
            for k, v in aggregate_metrics.items()
        ])
        df.to_csv(summary_dir / "aggregate_stability_metrics.csv", index=False)

        if verbose:
            print(f"\n{dataset_name} Summary:")
            print(df.to_string(index=False))

    return aggregate_metrics


def generate_cross_dataset_summary(all_dataset_results: Dict[str, Dict]):
    """Generate summary visualizations and tables across all datasets."""
    summary_dir = PROJECT_ROOT / "Results" / "Stability" / "Summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Combine results into single DataFrame
    rows = []
    for dataset, results in all_dataset_results.items():
        for config, metrics in results.items():
            rows.append({
                "dataset": dataset,
                "perturbation": config,
                **metrics
            })

    if not rows:
        print("No results to summarize.")
        return

    df = pd.DataFrame(rows)

    # Save aggregate table
    df.to_csv(summary_dir / "aggregate_stability_table.csv", index=False)

    # Generate correlation decay plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    perturbation_types = ["gaussian", "time_warp", "phase_shift"]
    colors = {"JapaneseVowels": "blue", "PenDigits": "green", "LSST": "red"}

    for ax, ptype in zip(axes, perturbation_types):
        for dataset in DATASETS:
            subset = df[(df["dataset"] == dataset) &
                       (df["perturbation"].str.startswith(ptype))]
            if subset.empty:
                continue

            # Sort by perturbation strength
            subset = subset.sort_values("perturbation")

            ax.plot(
                range(len(subset)),
                subset["rank_correlation_mean"],
                'o-',
                label=dataset,
                color=colors.get(dataset, "gray"),
                markersize=8
            )

            # Add error bars
            ax.fill_between(
                range(len(subset)),
                subset["rank_correlation_mean"] - subset["rank_correlation_std"],
                subset["rank_correlation_mean"] + subset["rank_correlation_std"],
                alpha=0.2,
                color=colors.get(dataset, "gray")
            )

        ax.set_title(f"{ptype.replace('_', ' ').title()}")
        ax.set_xlabel("Perturbation Strength")
        ax.set_ylabel("Rank Correlation")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Mechanism Stability vs Perturbation Strength", fontsize=14)
    plt.tight_layout()
    plt.savefig(summary_dir / "correlation_decay_all_datasets.png", dpi=150)
    plt.close()

    # Generate findings summary markdown
    findings = generate_findings_summary(df)
    with open(summary_dir / "findings_summary.md", "w") as f:
        f.write(findings)

    print(f"\nCross-dataset summary saved to {summary_dir}")


def generate_findings_summary(df: pd.DataFrame) -> str:
    """Generate markdown summary of findings."""
    lines = [
        "# Stability Experiment Findings Summary",
        "",
        "## Overview",
        "",
        f"- **Total experiments**: {len(df)}",
        f"- **Datasets analyzed**: {df['dataset'].nunique()}",
        f"- **Perturbation types**: {df['perturbation'].str.split('_').str[0].nunique()}",
        "",
        "## Key Findings",
        "",
    ]

    # Best/worst stability
    best = df.loc[df["stability_score_mean"].idxmax()]
    worst = df.loc[df["stability_score_mean"].idxmin()]

    lines.extend([
        f"### Most Stable Configuration",
        f"- Dataset: {best['dataset']}",
        f"- Perturbation: {best['perturbation']}",
        f"- Stability Score: {best['stability_score_mean']:.3f}",
        "",
        f"### Least Stable Configuration",
        f"- Dataset: {worst['dataset']}",
        f"- Perturbation: {worst['perturbation']}",
        f"- Stability Score: {worst['stability_score_mean']:.3f}",
        "",
    ])

    # Per-dataset summary
    lines.append("## Per-Dataset Summary")
    lines.append("")

    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        lines.extend([
            f"### {dataset}",
            f"- Mean rank correlation: {subset['rank_correlation_mean'].mean():.3f}",
            f"- Mean stability score: {subset['stability_score_mean'].mean():.3f}",
            f"- Valid experiments: {int(subset['n_valid'].sum())}",
            "",
        ])

    # Aggregate metrics table
    lines.extend([
        "## Aggregate Metrics Table",
        "",
        create_summary_table({
            row["perturbation"]: {
                "rank_correlation": row["rank_correlation_mean"],
                "topk_overlap_k3": row["topk_overlap_k3_mean"],
                "topk_overlap_k5": row["topk_overlap_k5_mean"],
                "topk_overlap_k10": row["topk_overlap_k10_mean"],
                "stability_score": row["stability_score_mean"]
            }
            for _, row in df.iterrows()
        }),
        "",
        "---",
        "*Generated automatically by run_stability_experiments.py*"
    ])

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run stability stress-testing experiments"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset to analyze: JapaneseVowels, PenDigits, LSST, or 'all'"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cuda', 'cpu', or 'auto'"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Determine which datasets to process
    if args.dataset.lower() == "all":
        datasets = DATASETS
    else:
        datasets = [args.dataset]

    # Run experiments
    all_results = {}
    for dataset in datasets:
        if dataset not in DATASET_CONFIG:
            print(f"Unknown dataset: {dataset}. Skipping.")
            continue

        results = run_stability_experiments(
            dataset, device, verbose=not args.quiet
        )
        all_results[dataset] = results

    # Generate cross-dataset summary
    if len(all_results) > 0:
        generate_cross_dataset_summary(all_results)

    print("\nExperiments complete!")


if __name__ == "__main__":
    main()
