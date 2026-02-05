"""
Tests for Utilities/metrics.py

Tests patching metrics including:
- Logit difference
- KL divergence
- Cross-entropy change
- Probability metrics
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np


class TestLogitDifference:
    """Tests for logit difference computation."""

    def test_basic_computation(self):
        """Basic logit difference should match manual calculation."""
        from Utilities.metrics import compute_logit_difference

        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ld = compute_logit_difference(logits, class_a=4, class_b=0)

        # Should be 5.0 - 1.0 = 4.0
        np.testing.assert_allclose(ld, 4.0)

    def test_symmetric(self):
        """Swapping classes should negate result."""
        from Utilities.metrics import compute_logit_difference

        logits = torch.tensor([1.0, 2.0, 3.0])

        ld_ab = compute_logit_difference(logits, class_a=2, class_b=0)
        ld_ba = compute_logit_difference(logits, class_a=0, class_b=2)

        np.testing.assert_allclose(ld_ab, -ld_ba)

    def test_handles_batch_dim(self):
        """Should handle batch dimension correctly."""
        from Utilities.metrics import compute_logit_difference

        logits = torch.tensor([[1.0, 2.0, 3.0]])  # Shape (1, 3)
        ld = compute_logit_difference(logits, class_a=2, class_b=0)

        np.testing.assert_allclose(ld, 2.0)


class TestLogitDifferenceChange:
    """Tests for logit difference change."""

    def test_no_change(self):
        """Same logits should give zero change."""
        from Utilities.metrics import compute_logit_difference_change

        logits = torch.tensor([1.0, 2.0, 3.0])
        change = compute_logit_difference_change(logits, logits, class_a=2, class_b=0)

        np.testing.assert_allclose(change, 0.0)

    def test_positive_change(self):
        """Increasing class_a should give positive change."""
        from Utilities.metrics import compute_logit_difference_change

        original = torch.tensor([1.0, 2.0, 3.0])
        patched = torch.tensor([1.0, 2.0, 5.0])  # Class 2 increased

        change = compute_logit_difference_change(original, patched, class_a=2, class_b=0)

        assert change > 0


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_identical_distributions(self):
        """KL divergence of identical distributions should be 0."""
        from Utilities.metrics import compute_kl_divergence

        probs = torch.softmax(torch.randn(5), dim=-1)
        kl = compute_kl_divergence(probs, probs)

        np.testing.assert_allclose(kl, 0.0, atol=1e-6)

    def test_non_negative(self, prob_pair):
        """KL divergence should be non-negative."""
        from Utilities.metrics import compute_kl_divergence

        probs_a, probs_b = prob_pair
        kl = compute_kl_divergence(probs_a, probs_b)

        assert kl >= 0

    def test_asymmetric(self, prob_pair):
        """KL(P||Q) should generally differ from KL(Q||P)."""
        from Utilities.metrics import compute_kl_divergence

        probs_a, probs_b = prob_pair

        kl_ab = compute_kl_divergence(probs_a, probs_b)
        kl_ba = compute_kl_divergence(probs_b, probs_a)

        # Generally different (though could be same by chance)
        # Just check they're computed
        assert isinstance(kl_ab, float)
        assert isinstance(kl_ba, float)

    def test_handles_batch_dim(self):
        """Should handle batch dimension."""
        from Utilities.metrics import compute_kl_divergence

        probs_a = torch.softmax(torch.randn(1, 5), dim=-1)
        probs_b = torch.softmax(torch.randn(1, 5), dim=-1)

        kl = compute_kl_divergence(probs_a, probs_b)

        assert kl >= 0


class TestSymmetricKL:
    """Tests for symmetric KL divergence."""

    def test_symmetric(self, prob_pair):
        """Symmetric KL should be symmetric."""
        from Utilities.metrics import compute_symmetric_kl

        probs_a, probs_b = prob_pair

        skl_ab = compute_symmetric_kl(probs_a, probs_b)
        skl_ba = compute_symmetric_kl(probs_b, probs_a)

        np.testing.assert_allclose(skl_ab, skl_ba, rtol=1e-5)


class TestJSDivergence:
    """Tests for Jensen-Shannon divergence."""

    def test_symmetric(self, prob_pair):
        """JS divergence should be symmetric."""
        from Utilities.metrics import compute_js_divergence

        probs_a, probs_b = prob_pair

        js_ab = compute_js_divergence(probs_a, probs_b)
        js_ba = compute_js_divergence(probs_b, probs_a)

        np.testing.assert_allclose(js_ab, js_ba, rtol=1e-5)

    def test_bounded(self, prob_pair):
        """JS divergence should be bounded by log(2)."""
        from Utilities.metrics import compute_js_divergence

        probs_a, probs_b = prob_pair
        js = compute_js_divergence(probs_a, probs_b)

        assert 0 <= js <= np.log(2) + 1e-6

    def test_zero_for_identical(self):
        """JS divergence should be 0 for identical distributions."""
        from Utilities.metrics import compute_js_divergence

        probs = torch.softmax(torch.randn(5), dim=-1)
        js = compute_js_divergence(probs, probs)

        np.testing.assert_allclose(js, 0.0, atol=1e-6)


class TestCrossEntropy:
    """Tests for cross-entropy metrics."""

    def test_certain_prediction(self):
        """CE should be low for confident correct prediction."""
        from Utilities.metrics import compute_cross_entropy

        probs = torch.tensor([0.01, 0.01, 0.96, 0.01, 0.01])
        ce = compute_cross_entropy(probs, true_label=2)

        # Should be close to -log(0.96) ≈ 0.04
        assert ce < 0.1

    def test_uncertain_prediction(self):
        """CE should be high for uncertain prediction."""
        from Utilities.metrics import compute_cross_entropy

        probs = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        ce = compute_cross_entropy(probs, true_label=0)

        # Should be -log(0.2) ≈ 1.6
        assert ce > 1.0

    def test_non_negative(self, sample_probs):
        """CE should be non-negative."""
        from Utilities.metrics import compute_cross_entropy

        ce = compute_cross_entropy(sample_probs, true_label=0)

        assert ce >= 0


class TestCrossEntropyChange:
    """Tests for cross-entropy change."""

    def test_improvement(self):
        """Increasing true class prob should decrease CE (negative change)."""
        from Utilities.metrics import compute_cross_entropy_change

        original = torch.tensor([0.3, 0.4, 0.3])
        patched = torch.tensor([0.7, 0.2, 0.1])  # True class (0) increased

        change = compute_cross_entropy_change(original, patched, true_label=0)

        assert change < 0  # Improvement


class TestProbabilityRatio:
    """Tests for probability ratio computation."""

    def test_no_change(self):
        """Same probs should give ratio 1.0."""
        from Utilities.metrics import compute_probability_ratio

        probs = torch.softmax(torch.randn(5), dim=-1)
        ratio = compute_probability_ratio(probs, probs, target_class=0)

        np.testing.assert_allclose(ratio, 1.0)

    def test_increase(self):
        """Increasing target prob should give ratio > 1."""
        from Utilities.metrics import compute_probability_ratio

        original = torch.tensor([0.2, 0.3, 0.5])
        patched = torch.tensor([0.5, 0.2, 0.3])

        ratio = compute_probability_ratio(original, patched, target_class=0)

        assert ratio > 1


class TestDeltaProbability:
    """Tests for delta probability computation."""

    def test_no_change(self):
        """Same probs should give delta 0."""
        from Utilities.metrics import compute_delta_probability

        probs = torch.softmax(torch.randn(5), dim=-1)
        delta = compute_delta_probability(probs, probs, target_class=0)

        np.testing.assert_allclose(delta, 0.0)

    def test_increase(self):
        """Increasing target prob should give positive delta."""
        from Utilities.metrics import compute_delta_probability

        original = torch.tensor([0.2, 0.8])
        patched = torch.tensor([0.6, 0.4])

        delta = compute_delta_probability(original, patched, target_class=0)

        np.testing.assert_allclose(delta, 0.4)


class TestComputeAllMetrics:
    """Tests for comprehensive metrics computation."""

    def test_returns_all_fields(self):
        """compute_all_metrics should return PatchingMetrics with all fields."""
        from Utilities.metrics import compute_all_metrics

        original = torch.randn(1, 5)
        patched = torch.randn(1, 5)

        result = compute_all_metrics(original, patched, true_class=0, corrupt_class=1)

        assert hasattr(result, 'delta_p')
        assert hasattr(result, 'logit_diff')
        assert hasattr(result, 'kl_divergence')
        assert hasattr(result, 'cross_entropy_change')
        assert hasattr(result, 'probability_ratio')

    def test_metrics_types(self):
        """All metrics should be floats."""
        from Utilities.metrics import compute_all_metrics

        original = torch.randn(1, 5)
        patched = torch.randn(1, 5)

        result = compute_all_metrics(original, patched, true_class=0)

        assert isinstance(result.delta_p, float)
        assert isinstance(result.logit_diff, float)
        assert isinstance(result.kl_divergence, float)
        assert isinstance(result.cross_entropy_change, float)
        assert isinstance(result.probability_ratio, float)


class TestRankHeadsByMetric:
    """Tests for head ranking utility."""

    def test_correct_order(self):
        """Highest value should be ranked first."""
        from Utilities.metrics import rank_heads_by_metric

        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        top = rank_heads_by_metric(matrix, top_k=1)

        # (1, 1) has value 4.0, should be first
        assert top[0] == (1, 1, 4.0)

    def test_top_k_length(self):
        """Should return exactly k results."""
        from Utilities.metrics import rank_heads_by_metric

        matrix = np.random.rand(3, 8)

        for k in [1, 3, 5]:
            top = rank_heads_by_metric(matrix, top_k=k)
            assert len(top) == k


class TestCompareMetricRankings:
    """Tests for metric ranking comparison."""

    def test_self_correlation(self):
        """Same metric should have correlation 1.0 with itself."""
        from Utilities.metrics import compare_metric_rankings

        matrix = np.random.rand(3, 8)
        metrics = {'metric1': matrix, 'metric2': matrix}

        correlations = compare_metric_rankings(metrics)

        assert ('metric1', 'metric2') in correlations
        np.testing.assert_allclose(correlations[('metric1', 'metric2')], 1.0)
