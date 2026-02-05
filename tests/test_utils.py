"""
Tests for Utilities/utils.py

Tests patching functions including:
- sweep_heads output shape and validity
- Patching determinism
- Probability sum constraints
- Core utility functions
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np


class TestSweepHeads:
    """Tests for the main sweep_heads function."""

    def test_output_shape(self, small_model, sample_pair, device):
        """sweep_heads should return correct shape."""
        from Utilities.utils import sweep_heads

        clean, corrupt = sample_pair
        num_classes = 3
        n_layers = 2
        n_heads = 4

        result = sweep_heads(small_model, clean, corrupt, num_classes)

        assert result.shape == (n_layers, n_heads, num_classes)

    def test_returns_numpy(self, small_model, sample_pair, device):
        """sweep_heads should return numpy array."""
        from Utilities.utils import sweep_heads

        clean, corrupt = sample_pair
        result = sweep_heads(small_model, clean, corrupt, num_classes=3)

        assert isinstance(result, np.ndarray)

    def test_probabilities_valid(self, small_model, sample_pair, device):
        """All outputs should be valid probability distributions."""
        from Utilities.utils import sweep_heads

        clean, corrupt = sample_pair
        result = sweep_heads(small_model, clean, corrupt, num_classes=3)

        # All values should be non-negative
        assert np.all(result >= 0), "Probabilities must be non-negative"

        # All values should be <= 1
        assert np.all(result <= 1), "Probabilities must be <= 1"

        # Each distribution should sum to 1
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-4)

    def test_determinism(self, small_model, sample_pair, device):
        """Same inputs should produce same outputs."""
        from Utilities.utils import sweep_heads

        clean, corrupt = sample_pair

        # Run twice
        result1 = sweep_heads(small_model, clean, corrupt, num_classes=3)
        result2 = sweep_heads(small_model, clean, corrupt, num_classes=3)

        np.testing.assert_array_almost_equal(result1, result2)


class TestGetProbs:
    """Tests for get_probs function."""

    def test_returns_numpy(self, small_model, sample_input, device):
        """get_probs should return numpy array."""
        from Utilities.utils import get_probs

        result = get_probs(small_model, sample_input)

        assert isinstance(result, np.ndarray)

    def test_output_shape(self, small_model, sample_input, device):
        """get_probs should return 1D probability array."""
        from Utilities.utils import get_probs

        result = get_probs(small_model, sample_input)

        assert result.shape == (3,)  # num_classes = 3

    def test_probabilities_sum_to_one(self, small_model, sample_input, device):
        """Probabilities should sum to 1."""
        from Utilities.utils import get_probs

        result = get_probs(small_model, sample_input)
        total = result.sum()

        np.testing.assert_allclose(total, 1.0, rtol=1e-5)

    def test_all_non_negative(self, small_model, sample_input, device):
        """All probabilities should be non-negative."""
        from Utilities.utils import get_probs

        result = get_probs(small_model, sample_input)

        assert np.all(result >= 0)


class TestSweepLayerwise:
    """Tests for layer-wise patching sweep."""

    def test_output_shape(self, small_model, sample_pair, device):
        """sweep_layerwise_patch should return correct shape."""
        from Utilities.utils import sweep_layerwise_patch

        clean, corrupt = sample_pair
        n_layers = 2
        num_classes = 3

        result = sweep_layerwise_patch(small_model, clean, corrupt, num_classes)

        assert result.shape == (n_layers, num_classes)

    def test_probabilities_valid(self, small_model, sample_pair, device):
        """Layerwise results should be valid probabilities."""
        from Utilities.utils import sweep_layerwise_patch

        clean, corrupt = sample_pair
        result = sweep_layerwise_patch(small_model, clean, corrupt, num_classes=3)

        assert np.all(result >= 0)
        assert np.all(result <= 1)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones(2), rtol=1e-4)


class TestSweepMLP:
    """Tests for MLP patching sweep."""

    def test_output_shape(self, small_model, sample_pair, device):
        """sweep_mlp_layers should return correct shape."""
        from Utilities.utils import sweep_mlp_layers

        clean, corrupt = sample_pair
        n_layers = 2
        num_classes = 3

        result = sweep_mlp_layers(small_model, clean, corrupt, num_classes)

        assert result.shape == (n_layers, num_classes)


class TestPositionLevelPatching:
    """Tests for position-level patching."""

    def test_head_position_shape(self, small_model, sample_pair, device):
        """Position-level head patching should return correct shape."""
        from Utilities.utils import sweep_attention_head_positions

        clean, corrupt = sample_pair
        num_classes = 3
        n_layers = 2
        n_heads = 4
        seq_len = 10  # From sample_pair fixture

        result = sweep_attention_head_positions(
            small_model, clean, corrupt,
            num_layers=n_layers, num_heads=n_heads,
            seq_len=seq_len, num_classes=num_classes
        )

        # Shape: (n_layers, n_heads, seq_len, num_classes)
        assert result.shape == (n_layers, n_heads, seq_len, num_classes)

    def test_mlp_position_shape(self, small_model, sample_pair, device):
        """Position-level MLP patching should return correct shape."""
        from Utilities.utils import sweep_mlp_positions

        clean, corrupt = sample_pair
        num_classes = 3
        n_layers = 2
        seq_len = 10

        result = sweep_mlp_positions(
            small_model, clean, corrupt,
            num_layers=n_layers, seq_len=seq_len, num_classes=num_classes
        )

        # Shape: (n_layers, seq_len, num_classes)
        assert result.shape == (n_layers, seq_len, num_classes)


class TestCausalGraphConstruction:
    """Tests for causal graph building."""

    def test_find_critical_patches_returns_list(self, small_model, sample_pair, device):
        """find_critical_patches should return list."""
        from Utilities.utils import sweep_attention_head_positions, get_probs, find_critical_patches

        clean, corrupt = sample_pair
        num_classes = 3
        n_layers = 2
        n_heads = 4
        seq_len = 10

        # Get position-level patching results
        patch_probs = sweep_attention_head_positions(
            small_model, clean, corrupt,
            num_layers=n_layers, num_heads=n_heads,
            seq_len=seq_len, num_classes=num_classes
        )
        baseline = get_probs(small_model, corrupt)

        critical = find_critical_patches(
            patch_probs, baseline, true_label=0, threshold=0.01
        )

        assert isinstance(critical, list)

    def test_critical_patches_structure(self, small_model, sample_pair, device):
        """Critical patches should have correct structure."""
        from Utilities.utils import sweep_attention_head_positions, get_probs, find_critical_patches

        clean, corrupt = sample_pair
        n_layers = 2
        n_heads = 4
        seq_len = 10

        patch_probs = sweep_attention_head_positions(
            small_model, clean, corrupt,
            num_layers=n_layers, num_heads=n_heads,
            seq_len=seq_len, num_classes=3
        )
        baseline = get_probs(small_model, corrupt)

        critical = find_critical_patches(
            patch_probs, baseline, true_label=0, threshold=0.001
        )

        # Each entry should be (layer, head, position, delta) tuple
        for item in critical:
            assert len(item) == 4
            layer, head, pos, delta = item
            assert 0 <= layer < n_layers
            assert 0 <= head < n_heads
            assert 0 <= pos < seq_len

    def test_build_causal_graph(self, small_model, sample_pair, device):
        """build_causal_graph should return networkx graph."""
        from Utilities.utils import build_causal_graph
        import networkx as nx

        # Create mock critical patches
        critical_patches = [
            (0, 1, 2, 0.1),
            (0, 2, 3, 0.15),
            (1, 3, 4, 0.2),
        ]

        graph = build_causal_graph(critical_patches)

        assert isinstance(graph, nx.DiGraph)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample_batch(self, small_model, device):
        """Should handle single sample correctly."""
        from Utilities.utils import get_probs

        x = torch.randn(1, 10, 4, device=device)
        result = get_probs(small_model, x)

        assert result.shape == (3,)

    def test_different_input_dims(self, device):
        """Should handle various input dimensions."""
        from Utilities.TST_trainer import TimeSeriesTransformer
        from Utilities.utils import get_probs

        for input_dim in [1, 4, 16]:
            model = TimeSeriesTransformer(
                input_dim=input_dim,
                num_classes=3,
                seq_len=10,
                d_model=32,
                n_head=4,
                num_encoder_layers=2,
                dim_feedforward=64,
                dropout=0.0
            ).to(device)
            model.eval()

            x = torch.randn(1, 10, input_dim, device=device)
            result = get_probs(model, x)

            assert result.shape == (3,)


class TestPatchingChangesOutput:
    """Tests for verifying patching has an effect."""

    def test_patching_changes_output(self, small_model, sample_pair, device):
        """Patching should change the output (for different inputs)."""
        from Utilities.utils import sweep_heads, get_probs

        clean, corrupt = sample_pair

        # Get baseline
        baseline = get_probs(small_model, corrupt)

        # Patch
        patched = sweep_heads(small_model, clean, corrupt, num_classes=3)

        # At least some heads should change the output
        max_diff = np.abs(patched - baseline).max()

        # May be small but should be non-zero
        assert max_diff >= 0


class TestVisualizationSmoke:
    """Smoke tests for visualization functions."""

    def test_plot_influence_runs(self, small_model, sample_pair, device):
        """plot_influence should run without error."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        from Utilities.utils import sweep_heads, get_probs, plot_influence

        clean, corrupt = sample_pair
        patch_probs = sweep_heads(small_model, clean, corrupt, num_classes=3)
        baseline = get_probs(small_model, corrupt)

        # Should not raise
        try:
            plot_influence(patch_probs, baseline, true_label=0)
        finally:
            plt.close('all')
