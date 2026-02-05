"""
Perturbation functions for stress-testing mechanistic interpretability.

Three label-preserving perturbations:
- Gaussian noise: adds small random noise
- Time warp: locally stretches/compresses the time axis
- Phase shift: circular shift in time
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Literal
from scipy.interpolate import interp1d


def gaussian_noise(
    X: torch.Tensor,
    sigma: float = 0.1,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Add Gaussian noise scaled by each channel's standard deviation.

    Args:
        X: Input tensor (N, seq_len, channels) or (seq_len, channels)
        sigma: Noise level relative to channel std (0.1 = 10% of std)
        seed: Random seed for reproducibility

    Returns:
        Perturbed tensor of same shape
    """
    if seed is not None:
        torch.manual_seed(seed)

    X_pert = X.clone()

    # Scale noise by per-channel std
    if X.dim() == 3:
        std = X.std(dim=(0, 1), keepdim=True)
    else:
        std = X.std(dim=0, keepdim=True)

    noise = torch.randn_like(X) * sigma * std
    return X_pert + noise


def time_warp(
    X: torch.Tensor,
    warp_factor: float = 0.1,
    num_segments: int = 5,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Apply random local time warping (stretch/compress segments).

    Args:
        X: Input tensor (N, seq_len, channels) or (seq_len, channels)
        warp_factor: Max relative stretch/compress (0.1 = ±10%)
        num_segments: Number of segments to warp independently
        seed: Random seed for reproducibility

    Returns:
        Warped tensor of same shape
    """
    if seed is not None:
        np.random.seed(seed)

    single = X.dim() == 2
    if single:
        X = X.unsqueeze(0)

    N, seq_len, channels = X.shape
    device = X.device
    X_warped = torch.zeros_like(X)

    for i in range(N):
        x = X[i].cpu().numpy()
        t_orig = np.linspace(0, 1, seq_len)

        # Random warp per segment
        seg_bounds = np.linspace(0, seq_len, num_segments + 1).astype(int)
        warp_amounts = 1 + np.random.uniform(-warp_factor, warp_factor, num_segments)

        t_warped = np.zeros(seq_len)
        for seg in range(num_segments):
            start, end = seg_bounds[seg], seg_bounds[seg + 1]
            seg_len = end - start
            warped_len = seg_len * warp_amounts[seg]

            if seg == 0:
                t_warped[start:end] = np.linspace(0, warped_len / seq_len, seg_len)
            else:
                t_warped[start:end] = t_warped[start - 1] + np.linspace(
                    0, warped_len / seq_len, seg_len + 1
                )[1:]

        t_warped = t_warped / t_warped[-1]  # Normalize to [0, 1]

        # Interpolate each channel
        x_warped = np.zeros_like(x)
        for c in range(channels):
            interp_fn = interp1d(t_warped, x[:, c], kind='linear', fill_value='extrapolate')
            x_warped[:, c] = interp_fn(t_orig)

        X_warped[i] = torch.from_numpy(x_warped).to(device)

    if single:
        X_warped = X_warped.squeeze(0)

    return X_warped.float()


def phase_shift(
    X: torch.Tensor,
    max_shift: int = 3,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Apply random circular time shift.

    Args:
        X: Input tensor (N, seq_len, channels) or (seq_len, channels)
        max_shift: Maximum timesteps to shift (randomly chosen 1 to max_shift)
        seed: Random seed for reproducibility

    Returns:
        Shifted tensor of same shape
    """
    if seed is not None:
        np.random.seed(seed)

    single = X.dim() == 2
    if single:
        X = X.unsqueeze(0)

    N = X.shape[0]
    X_shifted = torch.zeros_like(X)

    for i in range(N):
        shift = np.random.randint(1, max_shift + 1)
        direction = np.random.choice([-1, 1])
        X_shifted[i] = torch.roll(X[i], shifts=shift * direction, dims=0)

    if single:
        X_shifted = X_shifted.squeeze(0)

    return X_shifted


def apply_perturbation(
    X: torch.Tensor,
    method: Literal['gaussian', 'time_warp', 'phase_shift'],
    seed: Optional[int] = None,
    **kwargs
) -> torch.Tensor:
    """
    Apply specified perturbation.

    Args:
        X: Input tensor
        method: 'gaussian', 'time_warp', or 'phase_shift'
        seed: Random seed
        **kwargs: Passed to perturbation function

    Example:
        X_pert = apply_perturbation(X, 'gaussian', sigma=0.1)
    """
    fns = {
        'gaussian': gaussian_noise,
        'time_warp': time_warp,
        'phase_shift': phase_shift
    }

    if method not in fns:
        raise ValueError(f"Unknown method: {method}. Use: {list(fns.keys())}")

    return fns[method](X, seed=seed, **kwargs)


@torch.no_grad()
def validate_perturbation(
    model: nn.Module,
    X_orig: torch.Tensor,
    X_pert: torch.Tensor,
    y: torch.Tensor,
    max_accuracy_drop: float = 0.05
) -> Dict:
    """
    Check that perturbation preserves model accuracy within tolerance.

    Args:
        model: Trained model
        X_orig: Original inputs (N, seq_len, channels)
        X_pert: Perturbed inputs
        y: True labels (N,)
        max_accuracy_drop: Max allowed accuracy drop (default 5%)

    Returns:
        Dict with: valid, original_acc, perturbed_acc, accuracy_delta, message
    """
    model.eval()
    device = next(model.parameters()).device

    X_orig = X_orig.to(device)
    X_pert = X_pert.to(device)
    y = y.to(device)

    # Original accuracy
    preds_orig = model(X_orig).argmax(dim=1)
    acc_orig = (preds_orig == y).float().mean().item()

    # Perturbed accuracy
    preds_pert = model(X_pert).argmax(dim=1)
    acc_pert = (preds_pert == y).float().mean().item()

    delta = acc_pert - acc_orig
    valid = abs(delta) <= max_accuracy_drop

    return {
        'valid': valid,
        'original_acc': acc_orig,
        'perturbed_acc': acc_pert,
        'accuracy_delta': delta,
        'message': f"{'VALID' if valid else 'INVALID'}: Δ={delta:.4f} (threshold={max_accuracy_drop})"
    }


def get_perturbation_configs() -> Dict:
    """Standard perturbation configurations for experiments."""
    return {
        'gaussian': [
            {'sigma': 0.05},
            {'sigma': 0.10},
            {'sigma': 0.20}
        ],
        'time_warp': [
            {'warp_factor': 0.05},
            {'warp_factor': 0.10},
            {'warp_factor': 0.20}
        ],
        'phase_shift': [
            {'max_shift': 1},
            {'max_shift': 2},
            {'max_shift': 3}
        ]
    }
