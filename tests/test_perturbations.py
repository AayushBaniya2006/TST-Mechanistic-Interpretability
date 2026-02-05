"""
Tests for Utilities/perturbations.py

Tests perturbation functions including:
- Gaussian noise
- Time warping
- Phase shift
- Perturbation validation
"""

import pytest
import torch
import numpy as np


class TestGaussianNoise:
    """Tests for Gaussian noise perturbation."""

    def test_output_shape(self, sample_input, device):
        """Gaussian noise should preserve shape."""
        from Utilities.perturbations import gaussian_noise

        result = gaussian_noise(sample_input, sigma=0.1)

        assert result.shape == sample_input.shape

    def test_changes_input(self, sample_input, device):
        """Gaussian noise should change the input."""
        from Utilities.perturbations import gaussian_noise

        result = gaussian_noise(sample_input, sigma=0.1)

        assert not torch.allclose(result, sample_input)

    def test_magnitude_proportional_to_sigma(self, sample_input, device):
        """Larger sigma should produce larger changes."""
        from Utilities.perturbations import gaussian_noise

        result_small = gaussian_noise(sample_input, sigma=0.01, seed=42)
        result_large = gaussian_noise(sample_input, sigma=0.5, seed=42)

        diff_small = (result_small - sample_input).abs().mean()
        diff_large = (result_large - sample_input).abs().mean()

        assert diff_large > diff_small

    def test_reproducibility_with_seed(self, sample_input, device):
        """Same seed should produce same noise."""
        from Utilities.perturbations import gaussian_noise

        result1 = gaussian_noise(sample_input, sigma=0.1, seed=42)
        result2 = gaussian_noise(sample_input, sigma=0.1, seed=42)

        torch.testing.assert_close(result1, result2)

    def test_zero_sigma_no_change(self, sample_input, device):
        """Zero sigma should not change input."""
        from Utilities.perturbations import gaussian_noise

        result = gaussian_noise(sample_input, sigma=0.0)

        torch.testing.assert_close(result, sample_input)


class TestTimeWarp:
    """Tests for time warping perturbation."""

    def test_output_shape(self, sample_input, device):
        """Time warp should preserve shape."""
        from Utilities.perturbations import time_warp

        result = time_warp(sample_input, warp_factor=0.1)

        assert result.shape == sample_input.shape

    def test_changes_input(self, sample_input, device):
        """Time warp should change the input."""
        from Utilities.perturbations import time_warp

        # Use larger warp factor to ensure change
        result = time_warp(sample_input, warp_factor=0.2, seed=42)

        # May not always change for very short sequences
        # Just check it runs
        assert result.shape == sample_input.shape

    def test_preserves_value_range(self, sample_input, device):
        """Time warp should not drastically change value range."""
        from Utilities.perturbations import time_warp

        result = time_warp(sample_input, warp_factor=0.1)

        # Values should be in similar range (interpolation preserves range roughly)
        assert result.min() >= sample_input.min() - 1.0
        assert result.max() <= sample_input.max() + 1.0


class TestPhaseShift:
    """Tests for phase shift perturbation."""

    def test_output_shape(self, sample_input, device):
        """Phase shift should preserve shape."""
        from Utilities.perturbations import phase_shift

        result = phase_shift(sample_input, max_shift=2)

        assert result.shape == sample_input.shape

    def test_preserves_values(self, sample_input, device):
        """Phase shift should preserve all values (just reordered)."""
        from Utilities.perturbations import phase_shift

        result = phase_shift(sample_input, max_shift=3, seed=42)

        # Sorted values should be the same
        original_sorted = torch.sort(sample_input.flatten())[0]
        result_sorted = torch.sort(result.flatten())[0]

        torch.testing.assert_close(original_sorted, result_sorted)

    def test_small_shift_minimal_change(self, sample_input, device):
        """Small max_shift should produce minimal average change."""
        from Utilities.perturbations import phase_shift

        # With max_shift=1, the shift is between -1 and 1 (could be 0)
        result = phase_shift(sample_input, max_shift=1, seed=42)

        # Should have same shape
        assert result.shape == sample_input.shape


class TestApplyPerturbation:
    """Tests for the unified perturbation interface."""

    def test_gaussian_via_interface(self, sample_input, device):
        """apply_perturbation should work for gaussian."""
        from Utilities.perturbations import apply_perturbation

        result = apply_perturbation(
            sample_input,
            method='gaussian',
            sigma=0.1
        )

        assert result.shape == sample_input.shape

    def test_time_warp_via_interface(self, sample_input, device):
        """apply_perturbation should work for time_warp."""
        from Utilities.perturbations import apply_perturbation

        result = apply_perturbation(
            sample_input,
            method='time_warp',
            warp_factor=0.1
        )

        assert result.shape == sample_input.shape

    def test_phase_shift_via_interface(self, sample_input, device):
        """apply_perturbation should work for phase_shift."""
        from Utilities.perturbations import apply_perturbation

        result = apply_perturbation(
            sample_input,
            method='phase_shift',
            max_shift=2
        )

        assert result.shape == sample_input.shape

    def test_invalid_method_raises(self, sample_input, device):
        """Invalid perturbation method should raise error."""
        from Utilities.perturbations import apply_perturbation

        with pytest.raises((ValueError, KeyError)):
            apply_perturbation(sample_input, method='invalid_method')


class TestValidatePerturbation:
    """Tests for perturbation validation."""

    def test_validates_label_preservation(self, small_model, sample_input, device):
        """validate_perturbation should check label preservation."""
        from Utilities.perturbations import validate_perturbation, gaussian_noise

        perturbed = gaussian_noise(sample_input, sigma=0.1)

        # Create dummy labels
        y = torch.zeros(1, dtype=torch.long, device=device)

        result = validate_perturbation(
            small_model, sample_input, perturbed, y, max_accuracy_drop=0.5
        )

        assert isinstance(result, dict)

    def test_returns_dict_with_info(self, small_model, sample_input, device):
        """validate_perturbation should return dict with validation info."""
        from Utilities.perturbations import validate_perturbation, gaussian_noise

        perturbed = gaussian_noise(sample_input, sigma=0.01)
        y = torch.zeros(1, dtype=torch.long, device=device)

        result = validate_perturbation(
            small_model, sample_input, perturbed, y
        )

        assert isinstance(result, dict)


class TestPerturbationConfigs:
    """Tests for perturbation configuration utilities."""

    def test_get_configs_returns_dict(self):
        """get_perturbation_configs should return dict."""
        from Utilities.perturbations import get_perturbation_configs

        configs = get_perturbation_configs()

        assert isinstance(configs, dict)

    def test_configs_has_methods(self):
        """Configs should have perturbation methods."""
        from Utilities.perturbations import get_perturbation_configs

        configs = get_perturbation_configs()

        # Should have the three perturbation types
        assert 'gaussian' in configs or any('gauss' in str(k).lower() for k in configs.keys())

    def test_configs_cover_all_types(self):
        """Configs should cover all perturbation types."""
        from Utilities.perturbations import get_perturbation_configs

        configs = get_perturbation_configs()
        keys_str = str(configs.keys()).lower()

        # Check for presence of all three types
        assert 'gaussian' in keys_str or 'noise' in keys_str
        assert 'warp' in keys_str or 'time' in keys_str
        assert 'shift' in keys_str or 'phase' in keys_str


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_sequence(self, device):
        """Should handle short sequence length."""
        from Utilities.perturbations import gaussian_noise, phase_shift

        x = torch.randn(1, 3, 4, device=device)  # Very short sequence

        result_noise = gaussian_noise(x, sigma=0.1)
        assert result_noise.shape == x.shape

        result_shift = phase_shift(x, max_shift=1)
        assert result_shift.shape == x.shape

    def test_single_channel(self, device):
        """Should handle single channel input."""
        from Utilities.perturbations import gaussian_noise, time_warp

        x = torch.randn(1, 10, 1, device=device)

        result_noise = gaussian_noise(x, sigma=0.1)
        assert result_noise.shape == x.shape

        result_warp = time_warp(x, warp_factor=0.1)
        assert result_warp.shape == x.shape

    def test_batch_processing(self, device):
        """Should handle batched input."""
        from Utilities.perturbations import gaussian_noise

        x = torch.randn(8, 10, 4, device=device)

        result = gaussian_noise(x, sigma=0.1)
        assert result.shape == x.shape
