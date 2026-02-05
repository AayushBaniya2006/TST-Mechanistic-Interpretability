"""
Tests for Utilities/stability_metrics.py

Tests stability metrics including:
- Head importance computation
- Rank correlation
- Top-K overlap
- Composite stability score
"""

import pytest
import numpy as np
from scipy import stats


class TestHeadImportance:
    """Tests for head importance computation."""

    def test_output_shape(self):
        """get_head_importance should return correct shape."""
        from Utilities.stability_metrics import get_head_importance

        # Simulated patch_probs and baseline
        patch_probs = np.random.rand(3, 8, 5)  # 3 layers, 8 heads, 5 classes
        baseline = np.random.rand(5)
        baseline = baseline / baseline.sum()  # Valid probs

        result = get_head_importance(patch_probs, baseline, true_label=0)

        assert result.shape == (3, 8)

    def test_importance_reflects_delta(self):
        """Importance should reflect probability change."""
        from Utilities.stability_metrics import get_head_importance

        # Create patch_probs where one head has large effect
        patch_probs = np.ones((3, 8, 5)) * 0.2  # Baseline-like
        patch_probs[1, 3, 0] = 0.9  # Head (1,3) greatly increases class 0 prob

        baseline = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        importance = get_head_importance(patch_probs, baseline, true_label=0)

        # Head (1,3) should have highest importance
        assert importance[1, 3] == importance.max()


class TestHeadRanking:
    """Tests for head ranking computation."""

    def test_ranking_returns_list(self, importance_matrix):
        """get_head_ranking should return list of tuples."""
        from Utilities.stability_metrics import get_head_ranking

        ranking = get_head_ranking(importance_matrix)

        assert isinstance(ranking, list)
        assert len(ranking) > 0

    def test_ranking_tuple_structure(self, importance_matrix):
        """Each ranking entry should be (layer, head, value) tuple."""
        from Utilities.stability_metrics import get_head_ranking

        ranking = get_head_ranking(importance_matrix)

        for entry in ranking:
            assert len(entry) == 3
            layer, head, value = entry
            assert isinstance(layer, (int, np.integer))
            assert isinstance(head, (int, np.integer))
            assert isinstance(value, (float, np.floating))

    def test_ranking_sorted_by_importance(self):
        """Ranking should be sorted by absolute importance."""
        from Utilities.stability_metrics import get_head_ranking

        importance = np.array([[0.1, 0.5], [0.3, 0.8]])  # 2x2
        ranking = get_head_ranking(importance)

        # Values should be in descending order by absolute value
        values = [abs(entry[2]) for entry in ranking]
        assert values == sorted(values, reverse=True)


class TestRankCorrelation:
    """Tests for Spearman rank correlation."""

    def test_identical_matrices(self):
        """Identical matrices should have correlation 1.0."""
        from Utilities.stability_metrics import head_rank_correlation

        importance = np.random.rand(3, 8)
        rho = head_rank_correlation(importance, importance)

        np.testing.assert_allclose(rho, 1.0, atol=0.01)

    def test_opposite_rankings(self):
        """Opposite rankings should have correlation -1.0."""
        from Utilities.stability_metrics import head_rank_correlation

        importance1 = np.arange(24).reshape(3, 8).astype(float)
        importance2 = importance1.max() - importance1  # Reversed

        rho = head_rank_correlation(importance1, importance2)

        np.testing.assert_allclose(rho, -1.0, atol=0.01)

    def test_random_matrices_bounded(self):
        """Random matrices should have correlation in [-1, 1]."""
        from Utilities.stability_metrics import head_rank_correlation

        np.random.seed(42)
        imp1 = np.random.rand(3, 8)
        imp2 = np.random.rand(3, 8)

        rho = head_rank_correlation(imp1, imp2)

        assert -1 <= rho <= 1

    def test_similar_matrices_high_correlation(self, importance_pair):
        """Similar matrices should have high correlation."""
        from Utilities.stability_metrics import head_rank_correlation

        baseline, perturbed = importance_pair

        rho = head_rank_correlation(baseline, perturbed)

        # Should be positive since perturbed is baseline + small noise
        assert rho > 0.5


class TestTopKOverlap:
    """Tests for top-K overlap computation."""

    def test_identical_matrices(self):
        """Identical matrices should have overlap 1.0."""
        from Utilities.stability_metrics import topk_overlap

        importance = np.random.rand(3, 8)
        overlap = topk_overlap(importance, importance, k=5)

        np.testing.assert_allclose(overlap, 1.0)

    def test_no_overlap(self):
        """Completely different rankings should have overlap 0."""
        from Utilities.stability_metrics import topk_overlap

        # First matrix: first k heads are top
        imp1 = np.zeros((3, 8))
        imp1.flat[:5] = np.arange(5, 0, -1)  # Top 5 at indices 0-4

        # Second matrix: last k heads are top
        imp2 = np.zeros((3, 8))
        imp2.flat[-5:] = np.arange(5, 0, -1)  # Top 5 at indices 19-23

        overlap = topk_overlap(imp1, imp2, k=5)

        np.testing.assert_allclose(overlap, 0.0)

    def test_overlap_bounded(self, importance_matrix):
        """Overlap should be bounded [0, 1]."""
        from Utilities.stability_metrics import topk_overlap

        imp2 = importance_matrix + np.random.randn(*importance_matrix.shape) * 0.1

        for k in [1, 3, 5]:
            overlap = topk_overlap(importance_matrix, imp2, k=k)
            assert 0 <= overlap <= 1


class TestPatchRecoveryDelta:
    """Tests for patch recovery delta computation."""

    def test_returns_dict(self):
        """patch_recovery_delta should return dict."""
        from Utilities.stability_metrics import patch_recovery_delta

        imp1 = np.random.rand(3, 8)
        imp2 = np.random.rand(3, 8)

        delta = patch_recovery_delta(imp1, imp2)

        assert isinstance(delta, dict)

    def test_identical_matrices_small_delta(self):
        """Identical matrices should have small delta values."""
        from Utilities.stability_metrics import patch_recovery_delta

        importance = np.random.rand(3, 8)
        delta = patch_recovery_delta(importance, importance)

        # Should have some diff-related key with value near 0
        for key, value in delta.items():
            if 'diff' in key.lower() or 'delta' in key.lower():
                if isinstance(value, (int, float)):
                    assert abs(value) < 0.01


class TestMechanismStabilityScore:
    """Tests for composite stability score."""

    def test_identical_matrices(self):
        """Identical matrices should have score near 1.0."""
        from Utilities.stability_metrics import mechanism_stability_score

        importance = np.random.rand(3, 8)
        score = mechanism_stability_score(importance, importance)

        np.testing.assert_allclose(score, 1.0, atol=0.01)

    def test_score_bounded(self, importance_pair):
        """Score should be bounded [0, 1]."""
        from Utilities.stability_metrics import mechanism_stability_score

        baseline, perturbed = importance_pair
        score = mechanism_stability_score(baseline, perturbed)

        assert 0 <= score <= 1

    def test_worse_correlation_lower_score(self):
        """Worse correlation should give lower score."""
        from Utilities.stability_metrics import mechanism_stability_score

        baseline = np.random.rand(3, 8)

        # Similar matrix (small noise)
        similar = baseline + np.random.randn(3, 8) * 0.01
        score_similar = mechanism_stability_score(baseline, similar)

        # Different matrix (large noise)
        different = np.random.rand(3, 8)
        score_different = mechanism_stability_score(baseline, different)

        assert score_similar > score_different


class TestComputeAllMetrics:
    """Tests for comprehensive metrics computation."""

    def test_returns_dict(self, importance_pair):
        """compute_all_metrics should return dict."""
        from Utilities.stability_metrics import compute_all_metrics

        baseline, perturbed = importance_pair
        metrics = compute_all_metrics(baseline, perturbed)

        assert isinstance(metrics, dict)

    def test_has_correlation_metric(self, importance_pair):
        """Result should have some correlation metric."""
        from Utilities.stability_metrics import compute_all_metrics

        baseline, perturbed = importance_pair
        metrics = compute_all_metrics(baseline, perturbed)

        # Should have correlation-related key
        has_corr = any('corr' in k.lower() or 'rho' in k.lower()
                       for k in metrics.keys())
        assert has_corr

    def test_has_overlap_metric(self, importance_pair):
        """Result should have top-K overlap metrics."""
        from Utilities.stability_metrics import compute_all_metrics

        baseline, perturbed = importance_pair
        metrics = compute_all_metrics(baseline, perturbed)

        # Should have overlap-related key
        has_overlap = any('overlap' in k.lower() or 'topk' in k.lower()
                          for k in metrics.keys())
        assert has_overlap


class TestVisualizationFunctions:
    """Tests for visualization functions (smoke tests)."""

    def test_plot_importance_comparison(self, importance_pair):
        """plot_importance_comparison should run without error."""
        from Utilities.stability_metrics import plot_importance_comparison
        import matplotlib.pyplot as plt

        baseline, perturbed = importance_pair

        # Should not raise - may return figure or None
        result = plot_importance_comparison(baseline, perturbed, title="Test")

        # Close any created figures
        plt.close('all')

    def test_plot_correlation_decay(self):
        """plot_correlation_decay should run without error."""
        from Utilities.stability_metrics import plot_correlation_decay
        import matplotlib.pyplot as plt

        # Create sample data matching expected structure
        results = {
            'gaussian_0.05': {'rank_correlation': 0.95},
            'gaussian_0.10': {'rank_correlation': 0.88},
            'gaussian_0.20': {'rank_correlation': 0.75},
        }

        # Should not raise
        result = plot_correlation_decay(results, metric='rank_correlation', title="Test")

        plt.close('all')


class TestCreateSummaryTable:
    """Tests for summary table creation."""

    def test_creates_string(self):
        """create_summary_table should return string."""
        from Utilities.stability_metrics import create_summary_table

        results = {
            'dataset1': {
                'gaussian_0.1': {'rank_correlation': 0.9, 'topk_overlap_k5': 0.8},
            },
            'dataset2': {
                'gaussian_0.1': {'rank_correlation': 0.85, 'topk_overlap_k5': 0.75},
            }
        }

        table = create_summary_table(results)

        assert isinstance(table, str)
        assert len(table) > 0
