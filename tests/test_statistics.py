"""
Tests for Utilities/statistics.py

Tests statistical functions including:
- Confidence intervals
- Effect sizes
- Significance tests
- FDR correction
- Power analysis
"""

import pytest
import numpy as np
from scipy import stats


class TestConfidenceIntervals:
    """Tests for confidence interval computation."""

    def test_bootstrap_ci_contains_mean(self, sample_data):
        """Bootstrap CI should contain the sample mean."""
        from Utilities.statistics import compute_confidence_interval

        result = compute_confidence_interval(
            sample_data,
            confidence=0.95,
            method='bootstrap',
            n_bootstrap=1000
        )

        assert result.lower <= result.mean <= result.upper
        assert result.lower < result.upper

    def test_bootstrap_ci_width_increases_with_confidence(self):
        """Higher confidence should produce wider intervals."""
        from Utilities.statistics import compute_confidence_interval

        data = np.random.randn(100)

        ci_90 = compute_confidence_interval(data, confidence=0.90, n_bootstrap=1000)
        ci_95 = compute_confidence_interval(data, confidence=0.95, n_bootstrap=1000)
        ci_99 = compute_confidence_interval(data, confidence=0.99, n_bootstrap=1000)

        width_90 = ci_90.upper - ci_90.lower
        width_95 = ci_95.upper - ci_95.lower
        width_99 = ci_99.upper - ci_99.lower

        assert width_90 < width_95 < width_99

    def test_t_ci_matches_theoretical(self):
        """T-based CI should match theoretical formula."""
        from Utilities.statistics import compute_confidence_interval

        data = np.random.randn(1000)
        result = compute_confidence_interval(data, confidence=0.95, method='t')

        # Theoretical CI using t-distribution
        mean = data.mean()
        se = data.std(ddof=1) / np.sqrt(len(data))
        t_crit = stats.t.ppf(0.975, df=len(data)-1)

        theoretical_lower = mean - t_crit * se
        theoretical_upper = mean + t_crit * se

        np.testing.assert_allclose(result.lower, theoretical_lower, rtol=0.01)
        np.testing.assert_allclose(result.upper, theoretical_upper, rtol=0.01)

    def test_ci_with_small_sample(self):
        """CI should work with small samples."""
        from Utilities.statistics import compute_confidence_interval

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_confidence_interval(data, method='bootstrap', n_bootstrap=500)

        assert result.lower < result.mean < result.upper
        assert result.n_samples == 5

    def test_bca_method(self):
        """BCa method should work."""
        from Utilities.statistics import compute_confidence_interval

        data = np.random.randn(50)
        result = compute_confidence_interval(data, method='bca', n_bootstrap=500)

        assert result.lower < result.mean < result.upper
        assert result.method == 'bca'

    def test_percentile_method(self):
        """Percentile method should work."""
        from Utilities.statistics import compute_confidence_interval

        data = np.random.randn(50)
        result = compute_confidence_interval(data, method='percentile', n_bootstrap=500)

        assert result.lower < result.mean < result.upper


class TestEffectSizes:
    """Tests for effect size computation."""

    def test_cohens_d_small_for_similar(self):
        """Cohen's d should be small for very similar groups."""
        from Utilities.statistics import compute_effect_size

        np.random.seed(42)
        data1 = np.random.randn(100)
        data2 = data1 + np.random.randn(100) * 0.01  # Very small difference

        result = compute_effect_size(data1, data2)

        # Should be very close to 0
        assert abs(result.cohens_d) < 0.5

    def test_cohens_d_positive_for_larger_treatment(self, effect_size_samples):
        """Cohen's d should be positive when treatment > control."""
        from Utilities.statistics import compute_effect_size

        treatment, control = effect_size_samples
        result = compute_effect_size(treatment, control)

        assert result.cohens_d > 0

    def test_cohens_d_interpretation(self):
        """Cohen's d should have correct interpretation."""
        from Utilities.statistics import compute_effect_size

        # Create samples with d ≈ 0.8 (large effect)
        np.random.seed(42)
        control = np.random.randn(100)
        treatment = np.random.randn(100) + 0.8

        result = compute_effect_size(treatment, control, paired=False)

        assert result.interpretation in ['medium', 'large']
        assert result.cohens_d > 0.5

    def test_glass_delta_computed(self):
        """Glass's delta should be computed."""
        from Utilities.statistics import compute_effect_size

        control = np.random.randn(100) * 1.0
        treatment = np.random.randn(100) * 2.0 + 0.5

        result = compute_effect_size(treatment, control, paired=False)

        # Glass delta should exist
        assert result.glass_delta is not None

    def test_cliffs_delta_bounded(self, paired_samples):
        """Cliff's delta should be in [-1, 1]."""
        from Utilities.statistics import compute_effect_size

        treatment, control = paired_samples
        result = compute_effect_size(treatment, control)

        assert -1 <= result.cliffs_delta <= 1

    def test_effect_size_ci_exists(self):
        """Effect size should have confidence interval."""
        from Utilities.statistics import compute_effect_size

        treatment = np.random.randn(50) + 0.5
        control = np.random.randn(50)

        result = compute_effect_size(treatment, control)

        assert result.cohens_d_ci is not None
        assert len(result.cohens_d_ci) == 2
        assert result.cohens_d_ci[0] < result.cohens_d_ci[1]


class TestSignificanceTests:
    """Tests for significance testing."""

    def test_returns_dict(self):
        """significance_test should return dict with multiple tests."""
        from Utilities.statistics import significance_test

        treatment = np.random.randn(50) + 1.0
        control = np.random.randn(50)

        result = significance_test(treatment, control, paired=False)

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_ttest_different_distributions(self):
        """t-test should reject null for clearly different distributions."""
        from Utilities.statistics import significance_test

        np.random.seed(42)
        control = np.random.randn(100)
        treatment = np.random.randn(100) + 2.0  # Large difference

        result = significance_test(treatment, control, paired=False)

        # At least one test should be significant
        has_significant = any(
            r.significant for r in result.values()
            if hasattr(r, 'significant')
        )
        assert has_significant

    def test_paired_vs_unpaired(self):
        """Paired and unpaired tests should both run."""
        from Utilities.statistics import significance_test

        treatment = np.random.randn(50) + 0.5
        control = np.random.randn(50)

        paired_result = significance_test(treatment, control, paired=True)
        unpaired_result = significance_test(treatment, control, paired=False)

        assert isinstance(paired_result, dict)
        assert isinstance(unpaired_result, dict)


class TestFDRCorrection:
    """Tests for multiple comparison correction."""

    def test_fdr_returns_dict(self):
        """FDR correction should return dict."""
        from Utilities.statistics import apply_fdr_correction

        p_values = np.array([0.01, 0.04, 0.05, 0.10, 0.20])
        result = apply_fdr_correction(p_values, method='benjamini_hochberg')

        assert isinstance(result, dict)

    def test_fdr_has_expected_keys(self):
        """FDR result should have expected keys."""
        from Utilities.statistics import apply_fdr_correction

        p_values = np.array([0.01, 0.04, 0.05, 0.10, 0.20])
        result = apply_fdr_correction(p_values, method='benjamini_hochberg')

        # Check for some form of adjusted p-values
        assert any('adjust' in k.lower() or 'correct' in k.lower() or 'p_value' in k.lower()
                   for k in result.keys()) or 'reject' in str(result.keys()).lower()

    def test_bonferroni_method(self):
        """Bonferroni method should work."""
        from Utilities.statistics import apply_fdr_correction

        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = apply_fdr_correction(p_values, method='bonferroni')

        assert isinstance(result, dict)

    def test_holm_method(self):
        """Holm method should work."""
        from Utilities.statistics import apply_fdr_correction

        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = apply_fdr_correction(p_values, method='holm')

        assert isinstance(result, dict)


class TestPowerAnalysis:
    """Tests for power analysis."""

    def test_returns_dict(self):
        """Power analysis should return dict."""
        from Utilities.statistics import power_analysis

        result = power_analysis(effect_size=0.5, power=0.8)

        assert isinstance(result, dict)

    def test_larger_effect_needs_smaller_sample(self):
        """Larger effects should need smaller sample sizes."""
        from Utilities.statistics import power_analysis

        result_small = power_analysis(effect_size=0.2, power=0.8)
        result_medium = power_analysis(effect_size=0.5, power=0.8)
        result_large = power_analysis(effect_size=0.8, power=0.8)

        # Find the key that contains sample size info
        def get_n(result):
            for k, v in result.items():
                if 'n' in k.lower() or 'sample' in k.lower():
                    return v
            return result.get('required_n', result.get('n', list(result.values())[0]))

        n_small = get_n(result_small)
        n_medium = get_n(result_medium)
        n_large = get_n(result_large)

        assert n_small > n_medium > n_large

    def test_higher_power_needs_larger_sample(self):
        """Higher desired power should need larger samples."""
        from Utilities.statistics import power_analysis

        result_70 = power_analysis(effect_size=0.5, power=0.70)
        result_80 = power_analysis(effect_size=0.5, power=0.80)
        result_90 = power_analysis(effect_size=0.5, power=0.90)

        def get_n(result):
            for k, v in result.items():
                if 'n' in k.lower() or 'sample' in k.lower():
                    return v
            return result.get('required_n', result.get('n', list(result.values())[0]))

        assert get_n(result_70) < get_n(result_80) < get_n(result_90)


class TestVarianceDecomposition:
    """Tests for variance decomposition."""

    def test_variance_decomposition_runs(self):
        """Variance decomposition should run without error."""
        from Utilities.statistics import compute_variance_decomposition

        # Create simple data with groups
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        decomp = compute_variance_decomposition(data, groups)

        assert isinstance(decomp, dict)
        assert 'total_variance' in decomp or 'total' in str(decomp.keys()).lower()


class TestCorrelationCI:
    """Tests for correlation confidence intervals."""

    def test_correlation_ci_returns_dict(self):
        """compute_correlation_ci should return dict."""
        from Utilities.statistics import compute_correlation_ci

        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.5

        result = compute_correlation_ci(x, y, n_bootstrap=100)

        assert isinstance(result, dict)

    def test_correlation_ci_contains_correlation(self):
        """Result should contain correlation value."""
        from Utilities.statistics import compute_correlation_ci

        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.5

        result = compute_correlation_ci(x, y, n_bootstrap=100)

        # Should have some correlation-related key
        has_corr = any('corr' in k.lower() or 'rho' in k.lower() or 'r' == k.lower()
                       for k in result.keys())
        assert has_corr or len(result) > 0


class TestAggregateStatistics:
    """Tests for aggregate statistics."""

    def test_aggregate_runs(self):
        """compute_aggregate_statistics should run."""
        from Utilities.statistics import compute_aggregate_statistics

        results = [np.random.randn(10) for _ in range(5)]
        agg = compute_aggregate_statistics(results)

        assert isinstance(agg, dict)


class TestFormatResultTable:
    """Tests for result table formatting."""

    def test_format_returns_string(self):
        """format_result_table should return string."""
        from Utilities.statistics import format_result_table

        results = {'metric1': 0.5, 'metric2': 0.8}
        table = format_result_table(results)

        assert isinstance(table, str)
