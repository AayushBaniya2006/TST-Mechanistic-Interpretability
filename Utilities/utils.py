"""
Utilities for mechanistic interpretability of Time Series Transformers.

This module provides tools for activation patching, importance analysis,
and visualization of transformer-based time series classifiers.

Key functionality:
- Activation caching and patching
- Head-level and layer-level interventions
- Position-wise patching for temporal analysis
- Influence graph construction
- Visualization of patching effects
"""

from typing import List, Tuple, Dict, Optional, Callable, Any
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import torch
import torch.nn as nn
import networkx as nx


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def _validate_tensor_3d(x: torch.Tensor, name: str = "input") -> None:
    """
    Validate that input tensor has correct 3D shape for time series.

    Args:
        x: Input tensor to validate
        name: Name for error messages

    Raises:
        ValueError: If tensor is not 3D or has incorrect batch size
    """
    if x.dim() != 3:
        raise ValueError(f"{name} must be 3D (batch, seq_len, input_dim), got {x.dim()}D")
    if x.size(0) != 1:
        raise ValueError(f"{name} batch size must be 1, got {x.size(0)}")


def _get_device(model: nn.Module) -> torch.device:
    """Get the device of a model's parameters."""
    return next(model.parameters()).device


# =============================================================================
# FIGURE HOLDER
# =============================================================================

class FigureHolder:
    """
    Container for holding and saving matplotlib figures.

    Useful for storing figures generated during analysis for later export.

    Attributes:
        fig: The currently held matplotlib figure, or None

    Example:
        >>> holder = FigureHolder()
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> holder.update(fig)
        >>> holder.save("output.pdf")
    """

    def __init__(self) -> None:
        """Initialize empty figure holder."""
        self.fig: Optional[plt.Figure] = None

    def update(self, fig: plt.Figure) -> None:
        """
        Update the held figure.

        Args:
            fig: New matplotlib figure to hold
        """
        self.fig = fig

    def save(self, path: Optional[str] = None) -> str:
        """
        Save the held figure to a file.

        Args:
            path: File path to save to (should end in .pdf, .png, etc.)

        Returns:
            Success message or error message if no figure held
        """
        if self.fig:
            self.fig.savefig(path, format="pdf", bbox_inches="tight")
            return f"Saved to {path}"
        return "No figure to save"


# =============================================================================
# HOOK FUNCTIONS
# =============================================================================

def _register_hooks(
    model: nn.Module,
    substrings: List[str],
    hook_fn: Callable
) -> Dict[nn.Module, Any]:
    """
    Register forward hooks on modules matching given substrings.

    Args:
        model: PyTorch model to hook
        substrings: List of substrings to match in module names
        hook_fn: Hook function to register (signature: module, input, output)

    Returns:
        Dictionary mapping modules to their hook handles
    """
    handles = {}
    for name, module in model.named_modules():
        if any(sub in name for sub in substrings):
            handles[module] = module.register_forward_hook(hook_fn)
    return handles


def _remove_hooks(handles: Dict[nn.Module, Any]) -> None:
    """
    Remove all registered hooks.

    Args:
        handles: Dictionary of hook handles returned by _register_hooks
    """
    for handle in handles.values():
        handle.remove()


# =============================================================================
# CACHING AND PATCHING FUNCTIONS
# =============================================================================

def run_and_cache(
    model: nn.Module,
    x: torch.Tensor,
    targets: List[str]
) -> Dict[nn.Module, torch.Tensor]:
    """
    Run forward pass and cache activations at specified modules.

    Args:
        model: PyTorch model
        x: Input tensor (batch, seq_len, input_dim)
        targets: List of substrings to match module names

    Returns:
        Dictionary mapping modules to their cached output tensors

    Example:
        >>> cache = run_and_cache(model, clean_x, ["self_attn"])
        >>> clean_attn = cache[model.transformer_encoder.layers[0].self_attn]
    """
    cache = {}

    def hook(m: nn.Module, inp: Any, out: Any) -> None:
        val = out[0] if isinstance(out, tuple) else out
        cache[m] = val.detach().clone()

    handles = _register_hooks(model, targets, hook)
    model(x)
    _remove_hooks(handles)
    return cache


def patch_activations(
    model: nn.Module,
    x: torch.Tensor,
    cache: Dict[nn.Module, torch.Tensor],
    targets: List[str]
) -> torch.Tensor:
    """
    Run forward pass with cached activations patched into specified modules.

    Args:
        model: PyTorch model
        x: Input tensor to run (activations will be replaced)
        cache: Cached activations from run_and_cache
        targets: List of substrings to match module names

    Returns:
        Model output logits after patching
    """
    def patch_hook(m: nn.Module, inp: Any, out: Any) -> torch.Tensor:
        return cache[m]

    handles = _register_hooks(model, targets, patch_hook)
    out = model(x)
    _remove_hooks(handles)
    return out


# =============================================================================
# ATTENTION ANALYSIS
# =============================================================================

def get_attention_saliency(
    model: nn.Module,
    instance: torch.Tensor,
    layer_idx: int,
    head_idx: int
) -> torch.Tensor:
    """
    Extract attention saliency for a specific head.

    Computes how much each position attends to other positions,
    averaged across all query positions.

    Args:
        model: TimeSeriesTransformer model
        instance: Input tensor (seq_len, input_dim) - will be batched
        layer_idx: Transformer layer index (0-indexed)
        head_idx: Attention head index within layer

    Returns:
        Saliency tensor of shape (seq_len,) indicating importance of each position

    Example:
        >>> saliency = get_attention_saliency(model, x, layer_idx=0, head_idx=2)
        >>> # saliency[t] = how important timestep t is for this head's output
    """
    model.eval()
    device = _get_device(model)
    instance = instance.unsqueeze(0).to(device)

    attn_result = {}

    sa = model.transformer_encoder.layers[layer_idx].self_attn
    original_forward = sa.forward

    def forward_with_weights(query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return original_forward(query, key, value, **kwargs)

    sa.forward = forward_with_weights

    def hook(module: nn.Module, input: Any, output: Any) -> None:
        weights = output[1]  # (batch, heads, seq_len, seq_len)
        if weights is not None:
            attn_result['weights'] = weights.detach().cpu()

    handle = sa.register_forward_hook(hook)

    try:
        _ = model(instance)
    finally:
        handle.remove()
        sa.forward = original_forward  # Restore original

    full_attn = attn_result['weights'][0, head_idx]  # (seq_len, seq_len)
    saliency = full_attn.mean(dim=0)  # mean over queries (axis 0)

    return saliency  # (seq_len,)


# =============================================================================
# PROBABILITY AND HELPER FUNCTIONS
# =============================================================================

def get_probs(model: nn.Module, instance: torch.Tensor) -> NDArray[np.float32]:
    """
    Get class probability distribution from model.

    Args:
        model: Classification model
        instance: Input tensor (1, seq_len, input_dim)

    Returns:
        Probability array of shape (num_classes,)

    Example:
        >>> probs = get_probs(model, x)
        >>> predicted_class = probs.argmax()
    """
    device = _get_device(model)
    with torch.no_grad():
        logits = model(instance.to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


def get_encoder_inputs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Get the input to the transformer encoder after CNN embedding.

    Applies convolutional layers and positional encoding to get
    the sequence that enters the transformer.

    Args:
        model: TimeSeriesTransformer model with conv layers and pos_enc
        x: Raw input tensor (batch, seq_len, input_dim)

    Returns:
        Encoded tensor (batch, seq_len, d_model) ready for transformer
    """
    z = x.transpose(1, 2)
    for conv, bn in ((model.conv1, model.bn1), (model.conv2, model.bn2), (model.conv3, model.bn3)):
        z = torch.relu(bn(conv(z)))
    z = z.transpose(1, 2)
    return z + model.pos_enc


# =============================================================================
# ATTENTION HEAD PATCHING
# =============================================================================

def patch_attention_head(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    layer_idx: int,
    head_idx: int
) -> torch.Tensor:
    """
    Patch a single attention head from clean into corrupt forward pass.

    Replaces the output of one attention head in the corrupt run with
    its output from the clean run, measuring component influence.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor (1, seq_len, input_dim)
        corrupt: Corrupt/target input tensor (1, seq_len, input_dim)
        layer_idx: Layer containing the head to patch
        head_idx: Head index within the layer

    Returns:
        Output logits after patching

    Example:
        >>> logits = patch_attention_head(model, clean_x, corrupt_x, layer_idx=1, head_idx=3)
        >>> patched_probs = torch.softmax(logits, dim=1)
    """
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    layer_mod = model.transformer_encoder.layers[layer_idx]
    attn_mod = layer_mod.self_attn
    cache = {}

    handle = attn_mod.register_forward_hook(
        lambda m, i, o: cache.setdefault(m, o[0].detach().clone())
    )
    _ = model(clean_b)
    handle.remove()

    clean_val = cache[attn_mod]  # (1, seq_len, d_model)
    B, S, E = clean_val.shape
    H = attn_mod.num_heads
    d = E // H
    clean_heads = clean_val.view(B, S, H, d)[:, :, head_idx, :]

    def patch_hook(m: nn.Module, inp: Any, out: Any) -> torch.Tensor:
        out_val = out[0]
        out_heads = out_val.view(B, S, H, d)
        out_heads[:, :, head_idx, :] = clean_heads
        return out_heads.reshape(B, S, E)

    handle2 = attn_mod.register_forward_hook(patch_hook)
    logits = model(corrupt_b)
    handle2.remove()
    return logits


@contextmanager
def with_head_patch(
    model: nn.Module,
    clean: torch.Tensor,
    layer_idx: int,
    head_idx: int
):
    """
    Context manager for temporarily patching an attention head.

    Useful for running multiple forward passes with the same head patched.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean input to cache activations from
        layer_idx: Layer containing the head to patch
        head_idx: Head index within the layer

    Yields:
        None - use model within context to run patched forward passes

    Example:
        >>> with with_head_patch(model, clean_x, layer_idx=1, head_idx=2):
        ...     logits1 = model(corrupt_x1)
        ...     logits2 = model(corrupt_x2)
    """
    device = _get_device(model)
    clean_b = clean.to(device)

    layer_mod = model.transformer_encoder.layers[layer_idx]
    attn_mod = layer_mod.self_attn
    cache = {}

    def save_hook(m: nn.Module, i: Any, o: Any) -> None:
        cache[m] = o[0].detach().clone()

    handle1 = attn_mod.register_forward_hook(save_hook)
    _ = model(clean_b)
    handle1.remove()

    clean_val = cache[attn_mod]
    B, S, E = clean_val.shape
    H = attn_mod.num_heads
    d = E // H
    clean_heads = clean_val.view(B, S, H, d)[:, :, head_idx, :]

    def patch_hook(m: nn.Module, inp: Any, out: Any) -> torch.Tensor:
        out_val = out[0]
        heads = out_val.view(B, S, H, d)
        heads[:, :, head_idx, :] = clean_heads
        return heads.reshape(B, S, E)

    handle2 = attn_mod.register_forward_hook(patch_hook)
    try:
        yield
    finally:
        handle2.remove()


def patch_all_heads_in_layer(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    layer: int
) -> torch.Tensor:
    """
    Patch all attention heads in a single layer.

    Replaces the entire attention output of a layer from the clean
    forward pass into the corrupt forward pass.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor
        layer: Layer index to patch

    Returns:
        Output logits after patching entire layer's attention
    """
    model.eval()
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    cached = {}

    def save_clean_attn_output(module: nn.Module, input: Any, output: Any) -> None:
        cached["attn_output"] = output

    hook = model.transformer_encoder.layers[layer].self_attn.register_forward_hook(save_clean_attn_output)
    _ = model(clean_b)
    hook.remove()

    def patch_attn_output(module: nn.Module, input: Any, output: Any):
        return cached["attn_output"]

    hook = model.transformer_encoder.layers[layer].self_attn.register_forward_hook(patch_attn_output)
    with torch.no_grad():
        logits = model(corrupt_b)
    hook.remove()

    return logits


def patch_mlp_activation(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    layer_idx: int
) -> torch.Tensor:
    """
    Patch MLP layer activation from clean into corrupt forward pass.

    Patches the output of the second linear layer (after activation)
    in the feedforward network of the specified transformer layer.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor
        layer_idx: Layer containing the MLP to patch

    Returns:
        Output logits after patching MLP
    """
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    layer_mod = model.transformer_encoder.layers[layer_idx]
    mlp_layer = layer_mod.linear2

    cache = {}

    def cache_hook(m: nn.Module, inp: Any, out: Any) -> None:
        cache[m] = out.detach().clone()

    handle_cache = mlp_layer.register_forward_hook(cache_hook)
    _ = model(clean_b)
    handle_cache.remove()

    def patch_hook(m: nn.Module, inp: Any, out: Any) -> torch.Tensor:
        return cache[m]

    handle_patch = mlp_layer.register_forward_hook(patch_hook)
    logits = model(corrupt_b)
    handle_patch.remove()

    return logits


# =============================================================================
# POSITION-WISE PATCHING
# =============================================================================

def patch_attention_head_at_position(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    pos_idx: int
) -> torch.Tensor:
    """
    Patch a single attention head at a specific sequence position.

    Fine-grained patching that only replaces the head's output
    at one timestep, leaving other positions unchanged.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor
        layer_idx: Layer containing the head
        head_idx: Head index within the layer
        pos_idx: Sequence position to patch (0-indexed timestep)

    Returns:
        Output logits after position-specific patching
    """
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    layer_mod = model.transformer_encoder.layers[layer_idx]
    attn_mod = layer_mod.self_attn
    cache = {}

    def cache_hook(m: nn.Module, inp: Any, out: Any) -> None:
        cache[m] = out[0].detach().clone()

    handle_cache = attn_mod.register_forward_hook(cache_hook)
    _ = model(clean_b)
    handle_cache.remove()

    clean_val = cache[attn_mod]  # (1, seq_len, d_model)
    B, S, E = clean_val.shape
    H = attn_mod.num_heads
    d = E // H

    clean_heads = clean_val.view(B, S, H, d)

    def patch_hook(m: nn.Module, inp: Any, out: Any) -> torch.Tensor:
        out_val = out[0]
        heads = out_val.view(B, S, H, d)
        heads[:, pos_idx, head_idx, :] = clean_heads[:, pos_idx, head_idx, :]
        return heads.reshape(B, S, E)

    handle_patch = attn_mod.register_forward_hook(patch_hook)
    logits = model(corrupt_b)
    handle_patch.remove()

    return logits


def patch_mlp_at_position(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    layer_idx: int,
    pos_idx: int
) -> torch.Tensor:
    """
    Patch MLP activation at a specific sequence position.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor
        layer_idx: Layer containing the MLP
        pos_idx: Sequence position to patch (0-indexed timestep)

    Returns:
        Output logits after position-specific MLP patching
    """
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    layer_mod = model.transformer_encoder.layers[layer_idx]
    mlp_layer = layer_mod.linear2
    cache = {}

    def cache_hook(m: nn.Module, inp: Any, out: Any) -> None:
        cache[m] = out.detach().clone()

    handle_cache = mlp_layer.register_forward_hook(cache_hook)
    _ = model(clean_b)
    handle_cache.remove()

    clean_val = cache[mlp_layer]  # (batch, seq_len, d_model)

    def patch_hook(m: nn.Module, inp: Any, out: Any) -> torch.Tensor:
        patched = out.clone()
        patched[:, pos_idx, :] = clean_val[:, pos_idx, :]
        return patched

    handle_patch = mlp_layer.register_forward_hook(patch_hook)
    logits = model(corrupt_b)
    handle_patch.remove()

    return logits


def patch_multiple_attention_heads_positions(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    critical_edges: List[Tuple[str, str, dict]]
) -> torch.Tensor:
    """
    Patch multiple attention heads at specific positions simultaneously.

    Used for circuit-level interventions where multiple components
    need to be patched together.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor
        critical_edges: List of (timestep_node, head_node, edge_dict) tuples
                       where timestep_node is "Time X" and head_node is "LxHy"

    Returns:
        Output logits after patching all specified head-position combinations
    """
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    cache = {}

    hook_layers = set()
    for u, v, _ in critical_edges:
        layer = int(v[1])
        hook_layers.add(layer)

    hook_layers = list(hook_layers)

    def cache_hook(m: nn.Module, inp: Any, out: Any) -> None:
        if m not in cache:
            cache[m] = out[0].detach().clone()

    handles_cache = []
    for layer_idx in hook_layers:
        attn_mod = model.transformer_encoder.layers[layer_idx].self_attn
        handles_cache.append(attn_mod.register_forward_hook(cache_hook))

    _ = model(clean_b)
    for h in handles_cache:
        h.remove()

    def patch_hook_factory(layer_idx: int):
        def patch_fn(m: nn.Module, inp: Any, out: Any) -> torch.Tensor:
            out_val = out[0]
            B, S, E = out_val.shape
            H = m.num_heads
            d = E // H
            heads = out_val.view(B, S, H, d)

            clean_val = cache[m]
            clean_heads = clean_val.view(B, S, H, d)

            for u, v, _ in critical_edges:
                l = int(v[1])
                h = int(v[3])
                pos = int(u.split()[1])

                if l == layer_idx:
                    heads[:, pos, h, :] = clean_heads[:, pos, h, :]

            return heads.reshape(B, S, E)

        return patch_fn

    handles_patch = []
    for layer_idx in hook_layers:
        attn_mod = model.transformer_encoder.layers[layer_idx].self_attn
        handles_patch.append(attn_mod.register_forward_hook(patch_hook_factory(layer_idx)))

    logits = model(corrupt_b)

    for h in handles_patch:
        h.remove()

    return logits


# =============================================================================
# SWEEP FUNCTIONS
# =============================================================================

def sweep_heads(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    num_classes: int
) -> NDArray[np.float32]:
    """
    Sweep over all attention heads, patching each individually.

    For each (layer, head) combination, patches that head from clean
    into corrupt and records the resulting probability distribution.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor (1, seq_len, input_dim)
        corrupt: Corrupt/target input tensor (1, seq_len, input_dim)
        num_classes: Number of output classes

    Returns:
        Array of shape (num_layers, num_heads, num_classes) containing
        probability distributions after patching each head

    Example:
        >>> probs = sweep_heads(model, clean_x, corrupt_x, num_classes=9)
        >>> delta = probs[:, :, true_label] - baseline_probs[true_label]
        >>> # delta[l, h] = change in P(true_label) from patching head h in layer l
    """
    model.eval()
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    L = len(model.transformer_encoder.layers)
    H = model.transformer_encoder.layers[0].self_attn.num_heads
    patch_probs = np.zeros((L, H, num_classes))

    for l in range(L):
        for h in range(H):
            logits = patch_attention_head(model, clean, corrupt, l, h)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            patch_probs[l, h] = probs
    return patch_probs


def sweep_layerwise_patch(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    num_classes: int
) -> NDArray[np.float32]:
    """
    Sweep over layers, patching all attention heads in each layer.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor
        num_classes: Number of output classes

    Returns:
        Array of shape (num_layers, num_classes) containing probability
        distributions after patching each layer's attention
    """
    L = len(model.transformer_encoder.layers)
    patched_probs = np.zeros((L, num_classes))

    for l in range(L):
        logits = patch_all_heads_in_layer(model, clean, corrupt, l)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        patched_probs[l] = probs

    return patched_probs


def sweep_mlp_layers(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    num_classes: int
) -> NDArray[np.float32]:
    """
    Sweep over layers, patching each MLP individually.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor
        num_classes: Number of output classes

    Returns:
        Array of shape (num_layers, num_classes) containing probability
        distributions after patching each layer's MLP
    """
    model.eval()
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    L = len(model.transformer_encoder.layers)
    patch_probs = np.zeros((L, num_classes))

    for l in range(L):
        logits = patch_mlp_activation(model, clean, corrupt, l)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        patch_probs[l] = probs
    return patch_probs


def sweep_attention_head_positions(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    num_layers: int,
    num_heads: int,
    seq_len: int,
    num_classes: int
) -> NDArray[np.float32]:
    """
    Sweep over all (layer, head, position) combinations.

    Fine-grained analysis of which head-position combinations
    are influential for predictions.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        seq_len: Sequence length
        num_classes: Number of output classes

    Returns:
        Array of shape (num_layers, num_heads, seq_len, num_classes)
    """
    model.eval()
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    patch_probs = np.zeros((num_layers, num_heads, seq_len, num_classes))

    for l in range(num_layers):
        for h in range(num_heads):
            for pos in range(seq_len):
                logits = patch_attention_head_at_position(model, clean, corrupt, l, h, pos)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                patch_probs[l, h, pos] = probs

    return patch_probs


def sweep_mlp_positions(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    num_layers: int,
    seq_len: int,
    num_classes: int
) -> NDArray[np.float32]:
    """
    Sweep over all (layer, position) combinations for MLP patching.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor
        num_layers: Number of transformer layers
        seq_len: Sequence length
        num_classes: Number of output classes

    Returns:
        Array of shape (num_layers, seq_len, num_classes)
    """
    model.eval()
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    patch_probs = np.zeros((num_layers, seq_len, num_classes))

    for l in range(num_layers):
        for pos in range(seq_len):
            logits = patch_mlp_at_position(model, clean, corrupt, l, pos)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            patch_probs[l, pos] = probs

    return patch_probs


# =============================================================================
# CAUSAL GRAPH CONSTRUCTION
# =============================================================================

def find_critical_patches(
    patch_probs: NDArray[np.float32],
    baseline_probs: NDArray[np.float32],
    true_label: int,
    threshold: float = 0.05
) -> List[Tuple[int, int, int, float]]:
    """
    Find head-position combinations with significant influence on predictions.

    Identifies patches where the change in probability of the true label
    exceeds the specified threshold.

    Args:
        patch_probs: Array of shape (L, H, P, C) from sweep_attention_head_positions
        baseline_probs: Array of shape (C,) baseline probabilities
        true_label: Index of the correct class
        threshold: Minimum absolute change to be considered critical

    Returns:
        List of (layer, head, position, delta) tuples for critical patches
    """
    delta = patch_probs[:, :, :, true_label] - baseline_probs[true_label]
    critical = []

    L, H, P = delta.shape

    for l in range(L):
        for h in range(H):
            for p in range(P):
                d = delta[l, h, p]
                if abs(d) > threshold:
                    critical.append((l, h, p, delta[l, h, p]))

    return critical


def build_causal_graph(
    critical_patches: List[Tuple[int, int, int, float]]
) -> nx.DiGraph:
    """
    Build an influence graph from critical patches.

    Note: Named 'causal_graph' for API compatibility. The graph represents
    component influence relationships, not formal causation.

    Creates a directed graph where timesteps connect to the attention
    heads that process them with significant effect.

    Args:
        critical_patches: Output from find_critical_patches

    Returns:
        NetworkX directed graph with edges weighted by delta values
    """
    G = nx.DiGraph()

    for layer, head, pos, delta in critical_patches:
        timestep_node = f"T {pos}"
        head_node = f"L{layer}H{head}"

        G.add_edge(timestep_node, head_node, weight=delta)

    return G


# =============================================================================
# HEAD-TO-HEAD AND HEAD-TO-OUTPUT ANALYSIS
# =============================================================================

def capture_all_heads(
    model: nn.Module,
    x: torch.Tensor,
    num_layers: int,
    num_heads: int
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Capture activations from all attention heads in a single forward pass.

    Args:
        model: TimeSeriesTransformer model
        x: Input tensor
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer

    Returns:
        Dictionary mapping (layer_idx, head_idx) to head activation tensors
    """
    cache = {}

    def make_hook(layer_idx: int, head_idx: int):
        def hook(m: nn.Module, inp: Any, out: Any) -> None:
            val = out[0] if isinstance(out, tuple) else out
            B, S, E = val.shape
            d = E // num_heads
            heads = val.view(B, S, num_heads, d)
            cache[(layer_idx, head_idx)] = heads[:, :, head_idx, :].detach().clone()
        return hook

    handles = []
    for l in range(num_layers):
        mod = model.transformer_encoder.layers[l].self_attn
        for h in range(num_heads):
            handles.append(mod.register_forward_hook(make_hook(l, h)))

    _ = model(x.to(_get_device(model)))

    for h in handles:
        h.remove()

    return cache


def sweep_head_to_head_influence(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor
) -> NDArray[np.float32]:
    """
    Measure how patching each source head affects all target heads.

    Computes MSE difference in target head activations when source
    head is patched from clean into corrupt.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor

    Returns:
        Array of shape (L, H, L, H) where [l1, h1, l2, h2] is the
        influence of head (l1, h1) on head (l2, h2)
    """
    model.eval()
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    L = len(model.transformer_encoder.layers)
    H = model.transformer_encoder.layers[0].self_attn.num_heads

    # Get baseline outputs
    corrupt_cache = capture_all_heads(model, corrupt, L, H)

    influence = np.zeros((L, H, L, H))

    for l_src in range(L):
        for h_src in range(H):
            with with_head_patch(model, clean, l_src, h_src):
                patched_cache = capture_all_heads(model, corrupt, L, H)

                for l_tgt in range(L):
                    for h_tgt in range(H):
                        if (l_tgt, h_tgt) in corrupt_cache:
                            diff = torch.nn.functional.mse_loss(
                                patched_cache[(l_tgt, h_tgt)],
                                corrupt_cache[(l_tgt, h_tgt)],
                                reduction='mean'
                            ).item()
                            influence[l_src, h_src, l_tgt, h_tgt] = diff

    return influence


def sweep_head_to_output_deltas(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    true_label: int,
    num_classes: int
) -> NDArray[np.float32]:
    """
    Compute change in P(true_label) from patching each head.

    Args:
        model: TimeSeriesTransformer model
        clean: Clean/source input tensor
        corrupt: Corrupt/target input tensor
        true_label: Index of the correct class
        num_classes: Number of output classes

    Returns:
        Array of shape (L, H) where [l, h] is the delta P(true_label)
        from patching head h in layer l
    """
    model.eval()
    device = _get_device(model)
    clean_b = clean.to(device)
    corrupt_b = corrupt.to(device)

    # Get baseline probabilities from corrupted input
    with torch.no_grad():
        baseline_logits = model(corrupt_b)
        baseline_probs = torch.softmax(baseline_logits, dim=1)[0].cpu().numpy()
        baseline_p = baseline_probs[true_label]

    # Init matrix
    L = len(model.transformer_encoder.layers)
    H = model.transformer_encoder.layers[0].self_attn.num_heads
    delta_p = np.zeros((L, H))

    for l in range(L):
        for h in range(H):
            patched_logits = patch_attention_head(model, clean, corrupt, l, h)
            patched_probs = torch.softmax(patched_logits, dim=1)[0].detach().cpu().numpy()
            delta = patched_probs[true_label] - baseline_p
            delta_p[l, h] = delta

    return delta_p


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_influence(
    patch_probs: NDArray[np.float32],
    baseline_probs: NDArray[np.float32],
    true_label: int
) -> None:
    """
    Plot heatmap of patching effect by layer and head.

    Shows delta P(true_label) for each (layer, head) combination.

    Args:
        patch_probs: Array of shape (L, H, C) from sweep_heads
        baseline_probs: Array of shape (C,) baseline probabilities
        true_label: Index of the correct class
    """
    delta = patch_probs[:, :, true_label] - baseline_probs[true_label]
    L, H = delta.shape

    fig, ax = plt.subplots(figsize=(H * 0.9, L * 0.9))
    sns.heatmap(delta, annot=True, fmt="+.2f",
                annot_kws={"fontsize": 11},
                xticklabels=[f"H{h}" for h in range(H)],
                yticklabels=[f"L{l}" for l in range(L)],
                center=0, cmap="Blues", ax=ax,
                cbar_kws={'label': 'ΔP'})

    rect = patches.Rectangle((0, 0), H, L, linewidth=1, edgecolor='black',
                              facecolor='none', transform=ax.transData, clip_on=False)
    ax.add_patch(rect)

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("ΔP(True) by Layer & Head")
    plt.tight_layout()
    plt.show()


def plot_layerwise_influence(
    patched_probs: NDArray[np.float32],
    baseline_probs: NDArray[np.float32],
    true_label: int
) -> None:
    """
    Plot bar chart of patching effect by layer.

    Shows delta P(true_label) when all heads in each layer are patched.

    Args:
        patched_probs: Array of shape (L, C) from sweep_layerwise_patch
        baseline_probs: Array of shape (C,) baseline probabilities
        true_label: Index of the correct class
    """
    palette = sns.color_palette("Blues_r", n_colors=6)[1:]

    delta = patched_probs[:, true_label] - baseline_probs[true_label]
    L = delta.shape[0]

    fig, ax = plt.subplots(figsize=(6, L * 0.6))
    sns.barplot(y=np.arange(L), x=delta, hue=np.arange(L), palette=palette[:L],
                dodge=False, ax=ax, legend=False, orient="h")

    for i, val in enumerate(delta):
        ax.text(val - 0.02, i, f"{val:.2f}",
                va='center', ha='right',
                fontsize=11, color='white' if val > 0.4 else 'black')

    ax.set_ylabel("Layer")
    ax.set_xlabel("ΔP(True Label)")
    ax.set_title("ΔP(True) by Layer (All Attention Heads Patched)")
    plt.tight_layout()
    plt.show()


def plot_mlp_influence(
    patch_probs: NDArray[np.float32],
    baseline_probs: NDArray[np.float32],
    true_label: int
) -> None:
    """
    Plot bar chart of MLP patching effect by layer.

    Args:
        patch_probs: Array of shape (L, C) from sweep_mlp_layers
        baseline_probs: Array of shape (C,) baseline probabilities
        true_label: Index of the correct class
    """
    palette = sns.color_palette("Blues_r", n_colors=6)[1:]
    delta = patch_probs[:, true_label] - baseline_probs[true_label]
    L = delta.shape[0]

    fig, ax = plt.subplots(figsize=(6, min(L * 0.6, 10)))
    sns.barplot(x=delta, y=np.arange(L), hue=np.arange(L), palette=palette[:L],
                dodge=False, ax=ax, legend=False, orient="h")

    for i, val in enumerate(delta):
        ax.text(val - 0.02, i, f"{val:.2f}",
                va='center', ha='right',
                fontsize=11, color='white' if val > 0.4 else 'black')

    ax.set_ylabel("Layer")
    ax.set_xlabel("ΔP(True Label)")
    ax.set_title("ΔP(True) by MLP Layer")
    plt.tight_layout()
    plt.show()


def plot_mini_deltas(
    patch_probs: NDArray[np.float32],
    baseline_probs: NDArray[np.float32],
    class_labels: Optional[List[str]] = None
) -> None:
    """
    Plot grid of small heatmaps showing class probability changes.

    Creates a (layers x heads) grid where each cell shows how patching
    that head changes the probability of each class.

    Args:
        patch_probs: Array of shape (L, H, C) from sweep_heads
        baseline_probs: Array of shape (C,) baseline probabilities
        class_labels: Optional list of class names

    Raises:
        ValueError: If array shapes don't match
    """
    pp = np.asarray(patch_probs)
    bp = np.asarray(baseline_probs)
    if pp.ndim != 3 or bp.ndim != 1 or pp.shape[2] != bp.shape[0]:
        raise ValueError(f"Shapes mismatch: patch_probs {pp.shape}, baseline_probs {bp.shape}")

    L, H, C = pp.shape
    cl = class_labels or [str(i) for i in range(C)]
    delta = pp - bp[None, None, :]
    vmax = np.abs(delta).max()

    fig, axes = plt.subplots(L, H, figsize=(2.2 * H, 2 * L), squeeze=False)
    for i in range(L):
        for j in range(H):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            data = delta[i, j, :].reshape(1, C)
            ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect='auto')
            for k in range(C):
                ax.text(k, 0, f"{data[0, k]:+.2f}", ha='center', va='center', fontsize=6)
            if i == 0:
                ax.set_title(f"H{j}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"L{i}", rotation=0, labelpad=12, fontsize=9)

    plt.suptitle("Change in Class Probabilities from Patching Each Head", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_timeseries_with_attention_overlay(
    instances: List[torch.Tensor],
    saliencies: List[torch.Tensor],
    labels: List[str],
    title: str = "Time Series with Attention Overlay"
) -> None:
    """
    Plot time series with attention saliency overlay.

    Shows time series values with shaded regions indicating attention
    importance at each timestep.

    Args:
        instances: List of time series tensors (seq_len, input_dim)
        saliencies: List of saliency tensors (seq_len,) from get_attention_saliency
        labels: List of labels for each instance
        title: Plot title
    """
    n = len(instances)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n), squeeze=False)

    blue_rgb = mpl.colors.to_rgb('blue')
    alpha = 0.4
    cmap = mpl.colors.ListedColormap([
        (*blue_rgb, alpha * v) for v in np.linspace(0, 1, 256)
    ])

    for idx, (instance, saliency) in enumerate(zip(instances, saliencies)):
        instance = instance.detach().cpu().numpy()
        saliency = saliency.detach().cpu().numpy()
        saliency = saliency / (saliency.max() + 1e-8)

        ax = axes[idx, 0]
        seq_len, input_dim = instance.shape
        time = np.arange(seq_len)

        for d in range(input_dim):
            ax.plot(time, instance[:, d], label=f"dim {d}", linewidth=1.5)

        for t in range(seq_len):
            ax.axvspan(t - 0.5, t + 0.5, color=blue_rgb, alpha=saliency[t] * alpha)

        ax.set_title(f"Clean Instance" if idx == 0 else f"Corrupt Instance")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.grid(True)
        if input_dim <= 5:
            ax.legend(fontsize=7)

    # Saliency-matching colorbar with alpha
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention Saliency")

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_head_position_patch_heatmap(
    patch_probs: NDArray[np.float32],
    baseline_probs: NDArray[np.float32],
    true_label: int,
    layer_idx: int,
    head_idx: int,
    title: str = "Attention Head Patch Effect by Position"
) -> None:
    """
    Plot position-wise patching effect for a specific head.

    Shows how patching each position of one head affects the output.

    Args:
        patch_probs: Array of shape (L, H, P, C) from sweep_attention_head_positions
        baseline_probs: Array of shape (C,) baseline probabilities
        true_label: Index of the correct class
        layer_idx: Layer to visualize
        head_idx: Head to visualize
        title: Plot title
    """
    delta = patch_probs[layer_idx, head_idx, :, true_label] - baseline_probs[true_label]
    seq_len = delta.shape[0]

    fig, ax = plt.subplots(figsize=(12, 2))
    sns.heatmap(delta[np.newaxis, :], cmap="Blues", annot=True, center=0,
                annot_kws={"rotation": 90, "fontsize": 11, "color": "white"},
                xticklabels=20, yticklabels=[f"L{layer_idx} H{head_idx}"],
                cbar_kws={'label': 'ΔP'}, ax=ax)

    rect = patches.Rectangle((0, 0), seq_len, 1, linewidth=1, edgecolor='black',
                              facecolor='none', transform=ax.transData, clip_on=False)
    ax.add_patch(rect)

    ax.set_xlabel("Position (Timestep)")
    ax.set_ylabel("")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_mlp_position_patch_heatmap(
    patch_probs: NDArray[np.float32],
    baseline_probs: NDArray[np.float32],
    true_label: int,
    title: str = "ΔP(True) by Layer & Position"
) -> None:
    """
    Plot heatmap of MLP patching effect by layer and position.

    Args:
        patch_probs: Array of shape (L, P, C) from sweep_mlp_positions
        baseline_probs: Array of shape (C,) baseline probabilities
        true_label: Index of the correct class
        title: Plot title
    """
    delta = patch_probs[:, :, true_label] - baseline_probs[true_label]
    L, P = delta.shape

    fig, ax = plt.subplots(figsize=(P * 0.4, L * 0.6))
    sns.heatmap(delta, annot=True, fmt="+.2f",
                annot_kws={"fontsize": 6},
                xticklabels=[f"T{p}" for p in range(P)],
                yticklabels=[f"L{l}" for l in range(L)],
                center=0, cmap="Blues", ax=ax,
                cbar_kws={'label': 'ΔP(True Label)'})

    rect = patches.Rectangle((0, 0), P, L, linewidth=1, edgecolor='black',
                              facecolor='none', transform=ax.transData, clip_on=False)
    ax.add_patch(rect)

    ax.set_xlabel("Position (Timestep)")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_causal_graph(
    G: nx.DiGraph,
    title: str = "Critical Circuits Graph"
) -> None:
    """
    Plot influence graph using spring layout.

    Note: Named 'causal_graph' for API compatibility. The graph represents
    component influence relationships, not formal causation.

    Nodes are colored by type (timestep vs head) and sized by degree.
    Edge colors indicate positive (red) vs negative (blue) effects.

    Args:
        G: NetworkX directed graph from build_causal_graph
        title: Plot title
    """
    pos = nx.spring_layout(G, k=0.8, seed=42)

    weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_colors = ["red" if w > 0 else "blue" for w in weights]

    # Color nodes by type
    node_colors = []
    for node in G.nodes():
        if node.startswith("Time"):
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightblue")

    # Size nodes by degree
    node_sizes = []
    for node in G.nodes():
        deg = G.degree(node)
        node_sizes.append(300 + deg * 100)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           width=[abs(w) * 10 for w in weights], arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_structured_graph_with_heads(
    G: nx.DiGraph,
    title: Optional[str] = None,
    node_labels: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Plot causal graph with structured layout showing layer hierarchy.

    Creates a layered visualization with:
    - Class nodes at top
    - Head nodes in middle (organized by layer)
    - Timestep nodes at bottom

    Args:
        G: NetworkX directed graph
        title: Plot title
        node_labels: Optional dictionary mapping node names to display labels

    Returns:
        Matplotlib figure object
    """
    # Group nodes (use internal names for layout)
    time_nodes = sorted([n for n in G.nodes if n.startswith("Time")],
                        key=lambda x: int(x.split()[1]))
    head_nodes = [n for n in G.nodes if n.startswith("L")]
    class_nodes = sorted([n for n in G.nodes if n.startswith("Class")],
                         key=lambda x: int(x.split()[1]))
    special_nodes = [n for n in G.nodes if n not in time_nodes + head_nodes + class_nodes]

    max_x = len(time_nodes) - 1 if time_nodes else 10

    # Group heads by layer
    heads_by_layer = defaultdict(list)
    for node in head_nodes:
        layer = int(node[1])
        heads_by_layer[layer].append(node)
    max_layer = max(heads_by_layer.keys()) if heads_by_layer else 0

    pos_raw = {}
    row_class = 0
    row_heads_start = 1
    row_heads_end = row_heads_start + max_layer
    row_time = row_heads_end + 1

    # Class nodes
    for i, node in enumerate(class_nodes):
        x = int(i * max_x / max(1, len(class_nodes) - 1))
        pos_raw[node] = (x, row_class)

    # Head layers (L2 -> L0 from top to bottom)
    for layer in sorted(heads_by_layer.keys(), reverse=True):
        heads = heads_by_layer[layer]
        for i, node in enumerate(sorted(heads, key=lambda x: int(x[3:]))):
            x = int(i * max_x / max(1, len(heads) - 1))
            y = row_heads_start + (max_layer - layer)
            pos_raw[node] = (x, y)

    # Time nodes
    for i, node in enumerate(time_nodes):
        pos_raw[node] = (i, row_time)

    # Special nodes
    for node in special_nodes:
        pos_raw[node] = (-1, row_time + 1)

    # Flip Y-axis for top-down layout
    max_y = max(y for _, y in pos_raw.values())
    pos = {n: (x, max_y - y) for n, (x, y) in pos_raw.items()}

    # Node colors
    node_colors = []
    for n in G.nodes:
        if n.startswith("Time"):
            node_colors.append("seagreen")
        elif n.startswith("L"):
            node_colors.append("steelblue")
        elif n.startswith("Class"):
            node_colors.append("orange")
        else:
            node_colors.append("gray")

    node_sizes = [800 + 200 * G.degree(n) for n in G.nodes]
    edge_weights = [G[u][v].get('weight', 0.1) for u, v in G.edges]
    edge_widths = [max(1.0, abs(w) * 8) for w in edge_weights]

    fig = plt.figure(figsize=(18, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=node_labels if node_labels else None,
        node_color=node_colors,
        node_size=node_sizes,
        width=edge_widths,
        edge_color='maroon',
        edgecolors='black',
        linewidths=1.2,
        font_size=11,
        font_color='white',
        font_weight='bold',
        arrows=True
    )

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return fig
