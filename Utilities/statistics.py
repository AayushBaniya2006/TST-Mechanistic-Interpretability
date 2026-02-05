"""
Statistical infrastructure for rigorous analysis of mechanistic interpretability experiments.

This module provides comprehensive statistical tools including:
- Bootstrap confidence intervals (no distributional assumptions)
- Effect size calculations (Cohen's d, Glass's delta, Cliff's delta)
- Significance tests (t-test, Wilcoxon, permutation)
- Multiple comparison correction (FDR)
- Power analysis
- Variance decomposition

References:
- Bootstrap CIs: https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html
- FDR correction: Benjamini & Hochberg (1995)
- Effect sizes: Cohen (1988), Cliff (1993)
"""

from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import spearmanr, pearsonr, wilcoxon, ttest_rel, ttest_ind, mannwhitneyu
import warnings


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class ConfidenceInterval:
    """Container for confidence interval results."""
    mean: float
    lower: float
    upper: float
    confidence: float
    method: str
    n_samples: int

    def __repr__(self) -> str:
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}] ({self.confidence*100:.0f}% CI, {self.method})"

    def contains(self, value: float) -> bool:
        """Check if a value falls within the CI."""
        return self.lower <= value <= self.upper


@dataclass
class EffectSize:
    """Container for effect size calculations."""
    cohens_d: float
    cohens_d_ci: Tuple[float, float]
    glass_delta: Optional[float]
    cliffs_delta: float
    interpretation: str  # 'negligible', 'small', 'medium', 'large'

    def __repr__(self) -> str:
        return f"Cohen's d={self.cohens_d:.3f} [{self.cohens_d_ci[0]:.3f}, {self.cohens_d_ci[1]:.3f}] ({self.interpretation})"


@dataclass
class SignificanceResult:
    """Container for significance test results."""
    test_name: str
    statistic: float
    p_value: float
    p_value_corrected: Optional[float]
    significant: bool
    alpha: float

    def __repr__(self) -> str:
        sig_str = "significant" if self.significant else "not significant"
        return f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.6f} ({sig_str} at α={self.alpha})"


# =============================================================================
# CONFIDENCE INTERVALS
# =============================================================================

def compute_confidence_interval(
    data: NDArray[np.float64],
    confidence: float = 0.95,
    method: Literal['bootstrap', 'percentile', 'bca', 't'] = 'bootstrap',
    n_bootstrap: int = 10000,
    seed: Optional[int] = None
) -> ConfidenceInterval:
    """
    Compute confidence interval for the mean of data.

    Args:
        data: 1D array of observations
        confidence: Confidence level (default 0.95 for 95% CI)
        method: CI method - 'bootstrap' (BCa), 'percentile', 'bca', or 't' (parametric)
        n_bootstrap: Number of bootstrap samples (default 10000 for stable estimates)
        seed: Random seed for reproducibility

    Returns:
        ConfidenceInterval dataclass with mean, lower, upper bounds

    Example:
        >>> data = np.array([0.1, 0.2, 0.15, 0.18, 0.22])
        >>> ci = compute_confidence_interval(data)
        >>> print(ci)
        0.1700 [0.1200, 0.2100] (95% CI, bootstrap)
    """
    data = np.asarray(data).flatten()
    n = len(data)

    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    mean = np.mean(data)

    if method == 't':
        # Parametric t-distribution CI
        se = stats.sem(data)
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
        lower = mean - t_crit * se
        upper = mean + t_crit * se
    else:
        # Bootstrap methods
        if seed is not None:
            np.random.seed(seed)

        # Generate bootstrap samples
        bootstrap_means = np.array([
            np.mean(np.random.choice(data, size=n, replace=True))
            for _ in range(n_bootstrap)
        ])

        alpha = 1 - confidence

        if method == 'percentile':
            lower = np.percentile(bootstrap_means, alpha/2 * 100)
            upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

        elif method in ['bootstrap', 'bca']:
            # BCa (Bias-Corrected and Accelerated) bootstrap
            # More accurate for skewed distributions

            # Bias correction
            z0 = stats.norm.ppf(np.mean(bootstrap_means < mean))

            # Acceleration (jackknife estimate)
            jackknife_means = np.array([
                np.mean(np.delete(data, i)) for i in range(n)
            ])
            jack_mean = np.mean(jackknife_means)
            num = np.sum((jack_mean - jackknife_means) ** 3)
            denom = 6 * (np.sum((jack_mean - jackknife_means) ** 2) ** 1.5)

            if denom == 0:
                a = 0
            else:
                a = num / denom

            # Adjusted percentiles
            z_alpha_lower = stats.norm.ppf(alpha / 2)
            z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

            def adjusted_percentile(z_alpha):
                num = z0 + z_alpha
                denom = 1 - a * num
                if denom == 0:
                    return 0.5
                return stats.norm.cdf(z0 + num / denom)

            p_lower = adjusted_percentile(z_alpha_lower) * 100
            p_upper = adjusted_percentile(z_alpha_upper) * 100

            # Clip to valid range
            p_lower = np.clip(p_lower, 0, 100)
            p_upper = np.clip(p_upper, 0, 100)

            lower = np.percentile(bootstrap_means, p_lower)
            upper = np.percentile(bootstrap_means, p_upper)
        else:
            raise ValueError(f"Unknown method: {method}")

    return ConfidenceInterval(
        mean=float(mean),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        method=method,
        n_samples=n
    )


def compute_correlation_ci(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    confidence: float = 0.95,
    method: Literal['spearman', 'pearson'] = 'spearman',
    n_bootstrap: int = 10000,
    seed: Optional[int] = None
) -> Dict:
    """
    Compute correlation with confidence interval via bootstrap.

    Args:
        x, y: Arrays to correlate
        confidence: Confidence level
        method: 'spearman' or 'pearson'
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Dict with 'rho', 'p_value', 'ci_lower', 'ci_upper'
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: {len(x)} vs {len(y)}")

    n = len(x)

    # Compute observed correlation
    if method == 'spearman':
        rho, p_value = spearmanr(x, y)
    else:
        rho, p_value = pearsonr(x, y)

    # Bootstrap CI
    if seed is not None:
        np.random.seed(seed)

    corr_func = spearmanr if method == 'spearman' else pearsonr

    bootstrap_rhos = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        r, _ = corr_func(x[idx], y[idx])
        if not np.isnan(r):
            bootstrap_rhos.append(r)

    bootstrap_rhos = np.array(bootstrap_rhos)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_rhos, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_rhos, (1 - alpha/2) * 100)

    return {
        'rho': float(rho),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'confidence': confidence,
        'method': method,
        'n': n
    }


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================

def compute_effect_size(
    treatment: NDArray[np.float64],
    control: NDArray[np.float64],
    paired: bool = True,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None
) -> EffectSize:
    """
    Compute effect sizes with confidence intervals.

    Args:
        treatment: Treatment group data
        control: Control group data
        paired: Whether samples are paired (same subjects)
        n_bootstrap: Bootstrap iterations for CI on Cohen's d
        seed: Random seed

    Returns:
        EffectSize dataclass with Cohen's d, Glass's delta, Cliff's delta

    Interpretation of Cohen's d:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    treatment = np.asarray(treatment).flatten()
    control = np.asarray(control).flatten()

    # Cohen's d
    if paired:
        # For paired data, use difference scores
        if len(treatment) != len(control):
            raise ValueError("Paired data must have same length")
        diff = treatment - control
        d = np.mean(diff) / np.std(diff, ddof=1)
    else:
        # Pooled standard deviation
        n1, n2 = len(treatment), len(control)
        var1 = np.var(treatment, ddof=1)
        var2 = np.var(control, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        d = (np.mean(treatment) - np.mean(control)) / pooled_std

    # Bootstrap CI for Cohen's d
    if seed is not None:
        np.random.seed(seed)

    bootstrap_ds = []
    n = len(treatment)
    for _ in range(n_bootstrap):
        if paired:
            idx = np.random.choice(n, size=n, replace=True)
            t_boot = treatment[idx]
            c_boot = control[idx]
            diff_boot = t_boot - c_boot
            d_boot = np.mean(diff_boot) / np.std(diff_boot, ddof=1)
        else:
            t_boot = np.random.choice(treatment, size=len(treatment), replace=True)
            c_boot = np.random.choice(control, size=len(control), replace=True)
            pooled = np.sqrt((np.var(t_boot, ddof=1) + np.var(c_boot, ddof=1)) / 2)
            d_boot = (np.mean(t_boot) - np.mean(c_boot)) / pooled

        if not np.isnan(d_boot) and not np.isinf(d_boot):
            bootstrap_ds.append(d_boot)

    d_ci = (np.percentile(bootstrap_ds, 2.5), np.percentile(bootstrap_ds, 97.5))

    # Glass's delta (uses control group SD only)
    control_std = np.std(control, ddof=1)
    glass_delta = (np.mean(treatment) - np.mean(control)) / control_std if control_std > 0 else None

    # Cliff's delta (non-parametric)
    # Proportion of pairs where treatment > control minus proportion where treatment < control
    cliffs_delta = _compute_cliffs_delta(treatment, control)

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    return EffectSize(
        cohens_d=float(d),
        cohens_d_ci=d_ci,
        glass_delta=float(glass_delta) if glass_delta is not None else None,
        cliffs_delta=float(cliffs_delta),
        interpretation=interpretation
    )


def _compute_cliffs_delta(x: NDArray, y: NDArray) -> float:
    """Compute Cliff's delta (non-parametric effect size)."""
    n_x, n_y = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


# =============================================================================
# SIGNIFICANCE TESTS
# =============================================================================

def significance_test(
    treatment: NDArray[np.float64],
    control: NDArray[np.float64],
    paired: bool = True,
    alpha: float = 0.05,
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
) -> Dict[str, SignificanceResult]:
    """
    Run multiple significance tests and return results.

    Args:
        treatment: Treatment group data
        control: Control group data
        paired: Whether samples are paired
        alpha: Significance level
        alternative: Alternative hypothesis direction

    Returns:
        Dict mapping test name to SignificanceResult

    Tests performed:
        - Parametric: paired/independent t-test
        - Non-parametric: Wilcoxon signed-rank / Mann-Whitney U
        - Permutation test (assumption-free)
    """
    treatment = np.asarray(treatment).flatten()
    control = np.asarray(control).flatten()

    results = {}

    if paired:
        if len(treatment) != len(control):
            raise ValueError("Paired data must have same length")

        # Paired t-test
        t_stat, t_pval = ttest_rel(treatment, control, alternative=alternative)
        results['paired_t_test'] = SignificanceResult(
            test_name='Paired t-test',
            statistic=float(t_stat),
            p_value=float(t_pval),
            p_value_corrected=None,
            significant=t_pval < alpha,
            alpha=alpha
        )

        # Wilcoxon signed-rank test
        diff = treatment - control
        # Remove zeros for Wilcoxon
        diff_nonzero = diff[diff != 0]
        if len(diff_nonzero) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w_stat, w_pval = wilcoxon(diff_nonzero, alternative=alternative)
            results['wilcoxon'] = SignificanceResult(
                test_name='Wilcoxon signed-rank',
                statistic=float(w_stat),
                p_value=float(w_pval),
                p_value_corrected=None,
                significant=w_pval < alpha,
                alpha=alpha
            )
    else:
        # Independent t-test
        t_stat, t_pval = ttest_ind(treatment, control, alternative=alternative)
        results['independent_t_test'] = SignificanceResult(
            test_name='Independent t-test',
            statistic=float(t_stat),
            p_value=float(t_pval),
            p_value_corrected=None,
            significant=t_pval < alpha,
            alpha=alpha
        )

        # Mann-Whitney U test
        u_stat, u_pval = mannwhitneyu(treatment, control, alternative=alternative)
        results['mann_whitney'] = SignificanceResult(
            test_name='Mann-Whitney U',
            statistic=float(u_stat),
            p_value=float(u_pval),
            p_value_corrected=None,
            significant=u_pval < alpha,
            alpha=alpha
        )

    # Permutation test (always valid)
    perm_pval = _permutation_test(treatment, control, paired=paired, n_permutations=10000)
    results['permutation'] = SignificanceResult(
        test_name='Permutation test',
        statistic=float(np.mean(treatment) - np.mean(control)),
        p_value=float(perm_pval),
        p_value_corrected=None,
        significant=perm_pval < alpha,
        alpha=alpha
    )

    return results


def _permutation_test(
    treatment: NDArray,
    control: NDArray,
    paired: bool = True,
    n_permutations: int = 10000,
    seed: Optional[int] = None
) -> float:
    """
    Permutation test for difference in means.

    Returns two-sided p-value.
    """
    if seed is not None:
        np.random.seed(seed)

    observed_diff = np.mean(treatment) - np.mean(control)

    if paired:
        # For paired data, randomly flip signs
        diff = treatment - control
        n = len(diff)

        count = 0
        for _ in range(n_permutations):
            signs = np.random.choice([-1, 1], size=n)
            perm_diff = np.mean(diff * signs)
            if abs(perm_diff) >= abs(observed_diff):
                count += 1
    else:
        # For independent data, shuffle group assignments
        combined = np.concatenate([treatment, control])
        n1 = len(treatment)

        count = 0
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
            if abs(perm_diff) >= abs(observed_diff):
                count += 1

    return (count + 1) / (n_permutations + 1)  # Add 1 for continuity correction


# =============================================================================
# MULTIPLE COMPARISON CORRECTION
# =============================================================================

def apply_fdr_correction(
    p_values: Union[List[float], NDArray[np.float64]],
    method: Literal['benjamini_hochberg', 'bonferroni', 'holm'] = 'benjamini_hochberg',
    alpha: float = 0.05
) -> Dict:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: Array of p-values
        method: Correction method
            - 'benjamini_hochberg': Controls False Discovery Rate (recommended)
            - 'bonferroni': Most conservative, controls Family-Wise Error Rate
            - 'holm': Step-down Bonferroni, less conservative
        alpha: Significance level

    Returns:
        Dict with:
            - 'p_corrected': Corrected p-values
            - 'significant': Boolean array of which tests are significant
            - 'n_significant': Number of significant tests
            - 'method': Method used
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    if method == 'bonferroni':
        p_corrected = np.minimum(p_values * n, 1.0)

    elif method == 'holm':
        # Holm step-down procedure
        sorted_idx = np.argsort(p_values)
        p_sorted = p_values[sorted_idx]

        p_corrected = np.zeros(n)
        for i, (idx, p) in enumerate(zip(sorted_idx, p_sorted)):
            p_corrected[idx] = min(p * (n - i), 1.0)

        # Enforce monotonicity
        p_corrected = np.maximum.accumulate(p_corrected[sorted_idx])[np.argsort(sorted_idx)]

    elif method == 'benjamini_hochberg':
        # Benjamini-Hochberg FDR procedure
        sorted_idx = np.argsort(p_values)
        p_sorted = p_values[sorted_idx]

        # Compute adjusted p-values
        cummin_input = p_sorted * n / (np.arange(n) + 1)
        # Reverse cumulative minimum
        p_adjusted_sorted = np.minimum.accumulate(cummin_input[::-1])[::-1]
        p_adjusted_sorted = np.minimum(p_adjusted_sorted, 1.0)

        # Put back in original order
        p_corrected = np.zeros(n)
        p_corrected[sorted_idx] = p_adjusted_sorted

    else:
        raise ValueError(f"Unknown method: {method}")

    significant = p_corrected < alpha

    return {
        'p_original': p_values,
        'p_corrected': p_corrected,
        'significant': significant,
        'n_significant': int(np.sum(significant)),
        'n_tests': n,
        'method': method,
        'alpha': alpha
    }


# =============================================================================
# POWER ANALYSIS
# =============================================================================

def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    n: Optional[int] = None,
    test_type: Literal['paired', 'independent'] = 'paired'
) -> Dict:
    """
    Perform power analysis for t-tests.

    Provide either:
        - effect_size + n: Calculate achieved power
        - effect_size + power: Calculate required n

    Args:
        effect_size: Expected Cohen's d
        alpha: Significance level
        power: Desired power (if calculating n)
        n: Sample size (if calculating power)
        test_type: 'paired' or 'independent'

    Returns:
        Dict with power, n, effect_size, and recommendations
    """
    from scipy.stats import norm

    # For paired t-test, df = n-1; for independent, df = 2n-2
    # Using normal approximation for simplicity

    z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed

    if n is not None:
        # Calculate power given n
        if test_type == 'paired':
            se = 1 / np.sqrt(n)
        else:
            se = np.sqrt(2 / n)

        ncp = effect_size / se  # Non-centrality parameter
        z_beta = ncp - z_alpha
        calculated_power = norm.cdf(z_beta)

        return {
            'n': n,
            'effect_size': effect_size,
            'alpha': alpha,
            'power': float(calculated_power),
            'adequate': calculated_power >= 0.80,
            'recommendation': f"Power={calculated_power:.2%}. {'Adequate' if calculated_power >= 0.80 else 'Consider increasing n'}."
        }
    else:
        # Calculate required n given power
        z_beta = norm.ppf(power)

        if test_type == 'paired':
            required_n = ((z_alpha + z_beta) / effect_size) ** 2
        else:
            required_n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        required_n = int(np.ceil(required_n))

        return {
            'n': required_n,
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'test_type': test_type,
            'recommendation': f"Need n≥{required_n} for {power:.0%} power to detect d={effect_size}"
        }


# =============================================================================
# VARIANCE DECOMPOSITION
# =============================================================================

def compute_variance_decomposition(
    data: NDArray[np.float64],
    groups: NDArray,
    group_names: Optional[List[str]] = None
) -> Dict:
    """
    Decompose total variance into between-group and within-group components.

    Args:
        data: 1D array of observations
        groups: Group labels for each observation
        group_names: Optional names for groups

    Returns:
        Dict with variance components, ICC, and ANOVA-style results
    """
    data = np.asarray(data)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    k = len(unique_groups)
    n = len(data)

    # Grand mean
    grand_mean = np.mean(data)

    # Total sum of squares
    ss_total = np.sum((data - grand_mean) ** 2)

    # Between-group sum of squares
    ss_between = 0
    group_stats = {}
    for g in unique_groups:
        mask = groups == g
        group_data = data[mask]
        n_g = len(group_data)
        group_mean = np.mean(group_data)
        ss_between += n_g * (group_mean - grand_mean) ** 2

        g_name = str(g) if group_names is None else group_names[list(unique_groups).index(g)]
        group_stats[g_name] = {
            'n': n_g,
            'mean': float(group_mean),
            'std': float(np.std(group_data, ddof=1)),
            'min': float(np.min(group_data)),
            'max': float(np.max(group_data))
        }

    # Within-group sum of squares
    ss_within = ss_total - ss_between

    # Degrees of freedom
    df_between = k - 1
    df_within = n - k
    df_total = n - 1

    # Mean squares
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0

    # F-statistic and p-value
    if ms_within > 0:
        f_stat = ms_between / ms_within
        p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)
    else:
        f_stat = np.inf
        p_value = 0.0

    # Variance components
    var_total = ss_total / df_total if df_total > 0 else 0
    var_between = (ms_between - ms_within) / (n / k) if ms_between > ms_within else 0
    var_within = ms_within

    # Intraclass correlation coefficient (ICC)
    if var_between + var_within > 0:
        icc = var_between / (var_between + var_within)
    else:
        icc = 0

    # Proportion of variance explained
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    omega_squared = max(0, omega_squared)  # Can be negative for small effects

    return {
        'ss_total': float(ss_total),
        'ss_between': float(ss_between),
        'ss_within': float(ss_within),
        'var_total': float(var_total),
        'var_between': float(var_between),
        'var_within': float(var_within),
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'eta_squared': float(eta_squared),
        'omega_squared': float(omega_squared),
        'icc': float(icc),
        'n_groups': k,
        'n_total': n,
        'group_stats': group_stats
    }


# =============================================================================
# AGGREGATE STATISTICS FOR EXPERIMENTS
# =============================================================================

def compute_aggregate_statistics(
    results_per_pair: List[NDArray[np.float64]],
    pair_ids: Optional[List[str]] = None,
    confidence: float = 0.95
) -> Dict:
    """
    Aggregate statistics across multiple pairs/samples.

    Args:
        results_per_pair: List of result arrays (one per pair)
        pair_ids: Optional identifiers for each pair
        confidence: Confidence level for CIs

    Returns:
        Dict with:
            - Overall statistics (mean, CI, effect sizes)
            - Per-pair summaries
            - Heterogeneity metrics
    """
    # Flatten results
    all_results = np.concatenate([np.asarray(r).flatten() for r in results_per_pair])

    # Overall CI
    overall_ci = compute_confidence_interval(all_results, confidence=confidence)

    # Per-pair statistics
    pair_stats = []
    for i, result in enumerate(results_per_pair):
        result = np.asarray(result).flatten()
        pair_id = pair_ids[i] if pair_ids else f"pair_{i}"

        pair_stats.append({
            'id': pair_id,
            'n': len(result),
            'mean': float(np.mean(result)),
            'std': float(np.std(result, ddof=1)) if len(result) > 1 else 0,
            'min': float(np.min(result)),
            'max': float(np.max(result))
        })

    # Heterogeneity (I² statistic)
    pair_means = np.array([ps['mean'] for ps in pair_stats])
    pair_vars = np.array([ps['std']**2 for ps in pair_stats])
    pair_ns = np.array([ps['n'] for ps in pair_stats])

    # Weighted mean
    weights = 1 / (pair_vars + 1e-10)  # Inverse variance weighting
    weighted_mean = np.average(pair_means, weights=weights)

    # Q statistic for heterogeneity
    Q = np.sum(weights * (pair_means - weighted_mean)**2)
    df = len(pair_means) - 1

    # I² = (Q - df) / Q
    if Q > df:
        I_squared = (Q - df) / Q
    else:
        I_squared = 0

    return {
        'overall': {
            'mean': overall_ci.mean,
            'ci_lower': overall_ci.lower,
            'ci_upper': overall_ci.upper,
            'std': float(np.std(all_results, ddof=1)),
            'n_total': len(all_results),
            'n_pairs': len(results_per_pair)
        },
        'per_pair': pair_stats,
        'heterogeneity': {
            'Q_statistic': float(Q),
            'df': df,
            'I_squared': float(I_squared),
            'interpretation': _interpret_i_squared(I_squared)
        }
    }


def _interpret_i_squared(i2: float) -> str:
    """Interpret I² heterogeneity statistic."""
    if i2 < 0.25:
        return "low heterogeneity"
    elif i2 < 0.50:
        return "moderate heterogeneity"
    elif i2 < 0.75:
        return "substantial heterogeneity"
    else:
        return "considerable heterogeneity"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_result_table(
    results: Dict,
    metrics: List[str] = None,
    precision: int = 3
) -> str:
    """
    Format results as a markdown table for papers.

    Args:
        results: Dict with metric names as keys
        metrics: Which metrics to include (default: all)
        precision: Decimal places

    Returns:
        Markdown table string
    """
    if metrics is None:
        metrics = list(results.keys())

    lines = ["| Metric | Value | 95% CI | p-value |",
             "|--------|-------|--------|---------|"]

    for metric in metrics:
        val = results.get(metric, {})
        if isinstance(val, ConfidenceInterval):
            ci_str = f"[{val.lower:.{precision}f}, {val.upper:.{precision}f}]"
            lines.append(f"| {metric} | {val.mean:.{precision}f} | {ci_str} | - |")
        elif isinstance(val, dict):
            mean = val.get('mean', val.get('rho', 'N/A'))
            ci_lower = val.get('ci_lower', '-')
            ci_upper = val.get('ci_upper', '-')
            p_val = val.get('p_value', '-')

            if isinstance(mean, float):
                mean_str = f"{mean:.{precision}f}"
            else:
                mean_str = str(mean)

            if isinstance(ci_lower, float) and isinstance(ci_upper, float):
                ci_str = f"[{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"
            else:
                ci_str = "-"

            if isinstance(p_val, float):
                p_str = f"{p_val:.{precision+1}f}" if p_val >= 0.001 else "<0.001"
            else:
                p_str = str(p_val)

            lines.append(f"| {metric} | {mean_str} | {ci_str} | {p_str} |")

    return "\n".join(lines)
