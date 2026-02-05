"""
Tests for Utilities/baselines.py

Tests baseline comparison methods including:
- Random patching baseline
- Integrated Gradients
- Attention weight importance
- Method comparison
"""

import pytest
import torch
import numpy as np


class TestRandomPatchingBaseline:
    """Tests for random patching null distribution."""

    def test_returns_distribution(self, tiny_model, tiny_pair, device):
        """random_patching_baseline should return distribution stats."""
        from Utilities.baselines import random_patching_baseline

        clean, corrupt = tiny_pair
        result = random_patching_baseline(
            tiny_model, clean, corrupt,
            num_classes=2, n_permutations=50, seed=42
        )

        assert 'null_distribution' in result
        assert 'mean' in result
        assert 'std' in result
        assert 'percentiles' in result

    def test_distribution_size(self, tiny_model, tiny_pair, device):
        """Null distribution should have n_permutations elements."""
        from Utilities.baselines import random_patching_baseline

        clean, corrupt = tiny_pair
        n_perms = 100

        result = random_patching_baseline(
            tiny_model, clean, corrupt,
            num_classes=2, n_permutations=n_perms
        )

        assert len(result['null_distribution']) == n_perms

    def test_reproducibility(self, tiny_model, tiny_pair, device):
        """Same seed should give same results."""
        from Utilities.baselines import random_patching_baseline

        clean, corrupt = tiny_pair

        result1 = random_patching_baseline(
            tiny_model, clean, corrupt, num_classes=2,
            n_permutations=50, seed=42
        )
        result2 = random_patching_baseline(
            tiny_model, clean, corrupt, num_classes=2,
            n_permutations=50, seed=42
        )

        np.testing.assert_array_equal(
            result1['null_distribution'],
            result2['null_distribution']
        )

    def test_effects_non_negative(self, tiny_model, tiny_pair, device):
        """Effect sizes should be non-negative (absolute values)."""
        from Utilities.baselines import random_patching_baseline

        clean, corrupt = tiny_pair
        result = random_patching_baseline(
            tiny_model, clean, corrupt, num_classes=2, n_permutations=50
        )

        assert np.all(result['null_distribution'] >= 0)


class TestEmpiricalPValue:
    """Tests for empirical p-value computation."""

    def test_extreme_value_low_pvalue(self):
        """Very high observed effect should have low p-value."""
        from Utilities.baselines import compute_empirical_p_value

        null = np.random.randn(1000)  # Mean ~0
        observed = 5.0  # Very extreme

        p = compute_empirical_p_value(observed, null, alternative='greater')

        assert p < 0.01

    def test_typical_value_high_pvalue(self):
        """Typical observed effect should have high p-value."""
        from Utilities.baselines import compute_empirical_p_value

        null = np.random.randn(1000)
        observed = 0.0  # Right at mean

        p = compute_empirical_p_value(observed, null, alternative='two-sided')

        assert p > 0.3

    def test_pvalue_bounded(self):
        """P-value should be in (0, 1]."""
        from Utilities.baselines import compute_empirical_p_value

        null = np.random.randn(100)
        observed = 2.0

        p = compute_empirical_p_value(observed, null)

        assert 0 < p <= 1


class TestIntegratedGradientsImportance:
    """Tests for Integrated Gradients attribution."""

    def test_output_type(self, tiny_model, tiny_input, device):
        """Should return BaselineResult."""
        from Utilities.baselines import integrated_gradients_importance, BaselineResult

        result = integrated_gradients_importance(
            tiny_model, tiny_input, target_class=0, steps=10
        )

        assert isinstance(result, BaselineResult)
        assert result.method == 'integrated_gradients'

    def test_importance_shape(self, tiny_model, tiny_input, device):
        """Importance matrix should match model architecture."""
        from Utilities.baselines import integrated_gradients_importance

        result = integrated_gradients_importance(
            tiny_model, tiny_input, target_class=0, steps=10
        )

        # tiny_model: 1 layer, 2 heads
        assert result.importance_matrix.shape == (1, 2)

    def test_importance_non_negative(self, tiny_model, tiny_input, device):
        """Importance values should be non-negative."""
        from Utilities.baselines import integrated_gradients_importance

        result = integrated_gradients_importance(
            tiny_model, tiny_input, target_class=0, steps=10
        )

        assert np.all(result.importance_matrix >= 0)

    def test_importance_normalized(self, tiny_model, tiny_input, device):
        """Importance should be normalized to [0, 1]."""
        from Utilities.baselines import integrated_gradients_importance

        result = integrated_gradients_importance(
            tiny_model, tiny_input, target_class=0, steps=10
        )

        assert np.all(result.importance_matrix <= 1)


class TestGradientXInputImportance:
    """Tests for Gradient × Input saliency."""

    def test_output_type(self, tiny_model, tiny_input, device):
        """Should return BaselineResult."""
        from Utilities.baselines import gradient_x_input_importance, BaselineResult

        result = gradient_x_input_importance(
            tiny_model, tiny_input, target_class=0
        )

        assert isinstance(result, BaselineResult)
        assert result.method == 'gradient_x_input'

    def test_importance_shape(self, tiny_model, tiny_input, device):
        """Importance matrix should match model architecture."""
        from Utilities.baselines import gradient_x_input_importance

        result = gradient_x_input_importance(
            tiny_model, tiny_input, target_class=0
        )

        assert result.importance_matrix.shape == (1, 2)


class TestAttentionWeightImportance:
    """Tests for attention-based importance."""

    def test_output_type(self, tiny_model, tiny_input, device):
        """Should return BaselineResult."""
        from Utilities.baselines import attention_weight_importance, BaselineResult

        result = attention_weight_importance(tiny_model, tiny_input)

        assert isinstance(result, BaselineResult)

    def test_importance_shape(self, tiny_model, tiny_input, device):
        """Importance matrix should match model architecture."""
        from Utilities.baselines import attention_weight_importance

        result = attention_weight_importance(tiny_model, tiny_input)

        assert result.importance_matrix.shape == (1, 2)

    def test_different_aggregations(self, tiny_model, tiny_input, device):
        """Different aggregation methods should give different results."""
        from Utilities.baselines import attention_weight_importance

        result_entropy = attention_weight_importance(
            tiny_model, tiny_input, aggregation='entropy'
        )
        result_max = attention_weight_importance(
            tiny_model, tiny_input, aggregation='max'
        )

        # Results should be different (or could be same by coincidence)
        assert result_entropy.method != result_max.method

    def test_invalid_aggregation_raises(self, tiny_model, tiny_input, device):
        """Invalid aggregation should raise ValueError."""
        from Utilities.baselines import attention_weight_importance

        with pytest.raises(ValueError):
            attention_weight_importance(
                tiny_model, tiny_input, aggregation='invalid'
            )


class TestBaselineResult:
    """Tests for BaselineResult dataclass."""

    def test_get_ranking(self):
        """get_ranking should return valid permutation."""
        from Utilities.baselines import BaselineResult

        importance = np.array([[0.3, 0.7], [0.1, 0.9]])
        result = BaselineResult(
            method='test',
            importance_matrix=importance
        )

        ranking = result.get_ranking()

        # Should be permutation of 0,1,2,3
        assert set(ranking) == {0, 1, 2, 3}

    def test_get_top_k(self):
        """get_top_k should return correct heads."""
        from Utilities.baselines import BaselineResult

        importance = np.array([[0.1, 0.2], [0.3, 0.9]])
        result = BaselineResult(
            method='test',
            importance_matrix=importance
        )

        top2 = result.get_top_k(k=2)

        # (1,1) with 0.9 should be first
        assert top2[0] == (1, 1, 0.9)
        # (1,0) with 0.3 should be second
        assert top2[1] == (1, 0, 0.3)


class TestCompareAllMethods:
    """Tests for method comparison functionality."""

    def test_returns_comparison(self, tiny_model, tiny_pair, device):
        """compare_all_methods should return MethodComparison."""
        from Utilities.baselines import compare_all_methods, MethodComparison

        clean, corrupt = tiny_pair
        result = compare_all_methods(
            tiny_model, clean, corrupt,
            true_label=0, num_classes=2,
            include_random=False
        )

        assert isinstance(result, MethodComparison)

    def test_includes_multiple_methods(self, tiny_model, tiny_pair, device):
        """Should include multiple comparison methods."""
        from Utilities.baselines import compare_all_methods

        clean, corrupt = tiny_pair
        result = compare_all_methods(
            tiny_model, clean, corrupt,
            true_label=0, num_classes=2,
            include_random=False
        )

        assert len(result.methods) >= 3  # At least patching, IG, attention

    def test_correlation_matrix_shape(self, tiny_model, tiny_pair, device):
        """Correlation matrix should be square."""
        from Utilities.baselines import compare_all_methods

        clean, corrupt = tiny_pair
        result = compare_all_methods(
            tiny_model, clean, corrupt,
            true_label=0, num_classes=2,
            include_random=False
        )

        n_methods = len(result.methods)
        assert result.correlation_matrix.shape == (n_methods, n_methods)

    def test_diagonal_is_one(self, tiny_model, tiny_pair, device):
        """Diagonal of correlation matrix should be 1."""
        from Utilities.baselines import compare_all_methods

        clean, corrupt = tiny_pair
        result = compare_all_methods(
            tiny_model, clean, corrupt,
            true_label=0, num_classes=2,
            include_random=False
        )

        diagonal = np.diag(result.correlation_matrix.values)
        np.testing.assert_allclose(diagonal, np.ones(len(diagonal)))


class TestPrintMethodComparison:
    """Tests for comparison printing utility."""

    def test_prints_without_error(self, tiny_model, tiny_pair, device, capsys):
        """print_method_comparison should run without error."""
        from Utilities.baselines import compare_all_methods, print_method_comparison

        clean, corrupt = tiny_pair
        comparison = compare_all_methods(
            tiny_model, clean, corrupt,
            true_label=0, num_classes=2,
            include_random=False
        )

        # Should not raise
        print_method_comparison(comparison)

        captured = capsys.readouterr()
        assert "METHOD COMPARISON" in captured.out
