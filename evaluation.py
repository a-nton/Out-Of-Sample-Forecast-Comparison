"""
evaluation.py - Model evaluation and statistical testing functions.
Includes forecast accuracy metrics, statistical tests, and cross-sectional analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import ANALYSIS_CONFIG, PRESENTATION_CONFIG, format_percentage, format_number

# === ANNUALIZATION UTILITIES ===

def calculate_annualized_metrics(daily_alpha: float,
                                 daily_returns: pd.Series = None) -> dict:
    """Convert daily metrics to annualized values."""

    trading_days_per_year = 252

    # Simple annualization (approximation)
    annual_alpha_simple = daily_alpha * trading_days_per_year

    # Compounded annualization (more accurate)
    annual_alpha_compound = (1 + daily_alpha) ** trading_days_per_year - 1

    metrics = {
        'daily_alpha_pct': daily_alpha * 100,
        'annual_alpha_simple_pct': annual_alpha_simple * 100,
        'annual_alpha_compound_pct': annual_alpha_compound * 100,
    }

    if daily_returns is not None and len(daily_returns) > 0:
        annual_return = (1 + daily_returns.mean()) ** trading_days_per_year - 1
        annual_vol = daily_returns.std() * np.sqrt(trading_days_per_year)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        metrics.update({
            'annual_return_pct': annual_return * 100,
            'annual_volatility_pct': annual_vol * 100,
            'sharpe_ratio': sharpe
        })

    return metrics

# === SECTION 1: ERROR METRICS ===

def calculate_forecast_errors(results: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract forecast errors from results list.
    
    Args:
        results: List of dictionaries with 'error_alpha' and 'error_zero'
    
    Returns:
        errors_alpha: Array of errors with intercept
        errors_zero: Array of errors without intercept
    """
    errors_alpha = np.array([r['error_alpha'] for r in results])
    errors_zero = np.array([r['error_zero'] for r in results])
    
    # Remove any NaN values (both arrays simultaneously to maintain pairing)
    valid_mask = ~(np.isnan(errors_alpha) | np.isnan(errors_zero))
    
    if not valid_mask.all():
        n_removed = (~valid_mask).sum()
        warnings.warn(f"Removed {n_removed} observations with NaN errors")
    
    return errors_alpha[valid_mask], errors_zero[valid_mask]


def calculate_rmse(errors: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    if len(errors) == 0:
        return np.nan
    return np.sqrt(np.mean(errors ** 2))


def calculate_mae(errors: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    if len(errors) == 0:
        return np.nan
    return np.mean(np.abs(errors))


def calculate_mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        return np.nan
    
    ape = np.abs((actual[mask] - forecast[mask]) / actual[mask])
    return np.mean(ape) * 100


def calculate_directional_accuracy(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Calculate percentage of correct directional forecasts.
    Useful for understanding if model predicts direction correctly.
    """
    if len(actual) == 0:
        return np.nan
    
    # Check if sign matches (both positive or both negative)
    correct_direction = np.sign(actual) == np.sign(forecast)
    return np.mean(correct_direction) * 100


# === SECTION 2: STATISTICAL TESTS ===

def paired_t_test(errors1: np.ndarray, errors2: np.ndarray, 
                 alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    Perform paired t-test on forecast errors.
    Tests H0: E[errors1] = E[errors2] vs H1: E[errors1] ≠ E[errors2]
    
    Args:
        errors1: First set of errors (e.g., with alpha)
        errors2: Second set of errors (e.g., without alpha)
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        t_statistic: Test statistic
        p_value: Two-tailed p-value
    """
    if len(errors1) != len(errors2):
        raise ValueError("Error arrays must have same length")
    
    if len(errors1) < 2:
        return np.nan, np.nan
    
    # Calculate differences
    differences = errors1 - errors2
    
    # Remove NaN differences
    valid_diff = differences[~np.isnan(differences)]
    
    if len(valid_diff) < 2:
        return np.nan, np.nan
    
    # Perform t-test
    # Note: scipy's ttest_rel tests if the mean of differences is zero
    # This is equivalent to testing if means are equal
    result = stats.ttest_rel(errors1, errors2, alternative=alternative, nan_policy='omit')
    t_stat, p_val = result.statistic, result.pvalue
    # Interpretation: t_stat > 0 means errors1 > errors2 on average
    # Small p_val (e.g. <0.05) rejects equal means; large p_val implies no significant difference
    return t_stat, p_val


def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray, 
                        horizon: int = 1,
                        loss_function: str = 'squared',
                        alternative: str = 'two-sided') -> Tuple[float, float, dict]:
    """
    Diebold-Mariano test for predictive accuracy.
    Appropriate for large samples (n > 100) as it relies on asymptotic normality.
    
    Args:
        errors1: Forecast errors from model 1 (e.g., with alpha)
        errors2: Forecast errors from model 2 (e.g., without alpha)
        horizon: Forecast horizon (affects HAC standard error calculation)
        loss_function: 'squared' or 'absolute'
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        dm_statistic: Test statistic (asymptotically N(0,1))
        p_value: P-value based on normal distribution
        details: Dictionary with additional information
    """
    if len(errors1) != len(errors2):
        raise ValueError("Error arrays must have same length")
    
    n = len(errors1)
    
    if n < 30:
        warnings.warn(f"Sample size ({n}) is small for DM test. Results may be unreliable.")
    
    # Calculate loss differential
    if loss_function == 'squared':
        loss1 = errors1 ** 2
        loss2 = errors2 ** 2
    elif loss_function == 'absolute':
        loss1 = np.abs(errors1)
        loss2 = np.abs(errors2)
    else:
        raise ValueError("loss_function must be 'squared' or 'absolute'")
    
    d = loss1 - loss2  # Positive d means model 1 is worse
    d_mean = np.mean(d)
    
    # Calculate HAC variance using Newey-West
    # Use maxlag = max(1, horizon-1) to handle serial correlation
    maxlag = max(1, horizon - 1)
    gamma = [np.var(d, ddof=1)]  # Variance at lag 0

    for h in range(1, min(maxlag + 1, len(d))):
        # Calculate autocovariance at lag h
        cov_h = np.cov(d[:-h], d[h:])[0, 1]
        # Newey-West weight with truncation parameter maxlag
        weight = 1 - h / (maxlag + 1)
        gamma.append(weight * cov_h)
    
    # Long-run variance
    lr_variance = gamma[0] + 2 * sum(gamma[1:])
    
    # Standard error
    se = np.sqrt(lr_variance / n)
    
    # DM statistic
    if se > 0:
        dm_stat = d_mean / se
    else:
        dm_stat = 0 if abs(d_mean) < 1e-10 else np.inf * np.sign(d_mean)
    
    # P-value calculation (asymptotically standard normal)
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == 'less':
        p_value = stats.norm.cdf(dm_stat)
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(dm_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    
    # Additional details
    details = {
        'n_obs': n,
        'mean_loss_diff': d_mean,
        'se': se,
        'horizon': horizon,
        'loss_function': loss_function,
        'mean_loss1': np.mean(loss1),
        'mean_loss2': np.mean(loss2),
        'percent_model1_better': np.mean(loss1 < loss2) * 100,
        'autocorrelation_adjusted': horizon > 1
    }
    
    # Interpretation for large samples
    if n >= 100:
        details['interpretation'] = 'DM test is asymptotically valid for this sample size'
        if p_value < 0.05:
            if d_mean > 0:
                details['conclusion'] = 'Model 2 (without alpha) significantly better at 5% level'
            else:
                details['conclusion'] = 'Model 1 (with alpha) significantly better at 5% level'
        else:
            details['conclusion'] = 'No significant difference in predictive accuracy'
    else:
        details['interpretation'] = 'Sample size may be too small for reliable DM test'
    # Interpretation: dm_stat > 0 implies model 1 has higher loss than model 2
    # Small p_value (e.g. <0.05) indicates predictive accuracy differs; large p_value suggests no difference
    return dm_stat, p_value, details


def sign_test(errors1: np.ndarray, errors2: np.ndarray) -> Tuple[float, float]:
    """
    Non-parametric sign test for paired forecast comparisons.
    Tests if one model tends to have smaller absolute errors.
    
    Returns:
        statistic: Number of times |errors1| < |errors2|
        p_value: Two-tailed p-value from binomial test
    """
    abs_errors1 = np.abs(errors1)
    abs_errors2 = np.abs(errors2)
    
    # Count wins for model 1
    wins = np.sum(abs_errors1 < abs_errors2)
    ties = np.sum(abs_errors1 == abs_errors2)
    n_comparisons = len(errors1) - ties
    
    if n_comparisons == 0:
        return np.nan, np.nan
    
    # Binomial test with p=0.5
    p_value = stats.binomtest(wins, n_comparisons, p=0.5, alternative='two-sided').pvalue
    # Return win rate for model 1 and associated p_value
    # Low p_value (<0.05) -> model 1's win rate differs from 50%; high p_value -> no clear advantage
    return wins / n_comparisons, p_value


# === SECTION 3: BOOTSTRAP ANALYSIS ===

def bootstrap_rmse(errors: np.ndarray, n_bootstrap: int = 1000,
                  confidence_level: float = 0.95,
                  random_seed: Optional[int] = None) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence intervals for RMSE.
    
    Returns:
        lower_ci: Lower confidence bound
        upper_ci: Upper confidence bound
    """
    rng = np.random.default_rng(random_seed)

    n = len(errors)
    rmse_bootstrap = []

    for _ in range(n_bootstrap):
        sample_idx = rng.choice(n, n, replace=True)
        sample_errors = errors[sample_idx]
        rmse_bootstrap.append(calculate_rmse(sample_errors))
    
    # Calculate percentiles
    alpha = (1 - confidence_level) / 2
    lower_ci = np.percentile(rmse_bootstrap, alpha * 100)
    upper_ci = np.percentile(rmse_bootstrap, (1 - alpha) * 100)
    
    return lower_ci, upper_ci


def bootstrap_comparison(errors1: np.ndarray, errors2: np.ndarray,
                        metric: str = 'rmse',
                        n_bootstrap: int = 1000,
                        random_seed: Optional[int] = None) -> Dict[str, float]:
    """
    Bootstrap test for comparing two models.
    
    Args:
        errors1: Errors from model 1
        errors2: Errors from model 2
        metric: 'rmse' or 'mae'
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed
    
    Returns:
        Dictionary with bootstrap results
    """
    rng = np.random.default_rng(random_seed)

    n = len(errors1)
    metric_diffs = []
    
    # Metric function
    if metric == 'rmse':
        metric_func = calculate_rmse
    elif metric == 'mae':
        metric_func = calculate_mae
    else:
        raise ValueError("metric must be 'rmse' or 'mae'")
    
    for _ in range(n_bootstrap):
        # Resample pairs to maintain correlation
        idx = rng.choice(n, n, replace=True)
        
        metric1 = metric_func(errors1[idx])
        metric2 = metric_func(errors2[idx])
        
        metric_diffs.append(metric1 - metric2)
    
    metric_diffs = np.array(metric_diffs)
    
    # Calculate statistics
    results = {
        'mean_diff': np.mean(metric_diffs),
        'std_diff': np.std(metric_diffs),
        'ci_lower': np.percentile(metric_diffs, 2.5),
        'ci_upper': np.percentile(metric_diffs, 97.5),
        'p_value': np.mean(metric_diffs > 0) * 2,  # Two-sided test; low p_value (<0.05) => metric1 differs from metric2
        'prob_model1_better': np.mean(metric_diffs < 0),
        'significant': False
    }
    
    # Check if CI excludes zero
    results['significant'] = (results['ci_lower'] > 0) or (results['ci_upper'] < 0)
    
    return results


# === SECTION 4: CROSS-SECTIONAL ANALYSIS ===

def analyze_by_characteristic(results_df: pd.DataFrame, 
                            characteristic: str = 'market_cap',
                            n_groups: int = 5,
                            group_method: str = 'quintile') -> pd.DataFrame:
    """
    Analyze forecast performance by stock characteristics.
    
    Args:
        results_df: DataFrame with results
        characteristic: Column to group by
        n_groups: Number of groups
        group_method: 'quintile' or 'ntile'
    
    Returns:
        DataFrame with performance metrics by group
    """
    if characteristic not in results_df.columns:
        raise ValueError(f"Characteristic '{characteristic}' not in results")
    
    # Remove missing values
    clean_df = results_df.dropna(subset=[characteristic, 'error_alpha', 'error_zero'])
    
    if len(clean_df) < n_groups:
        warnings.warn(f"Not enough data for {n_groups} groups")
        n_groups = max(2, len(clean_df) // 10)
    
    # Create groups
    if group_method == 'quintile':
        labels = [f'Q{i+1}' for i in range(n_groups)]
        clean_df['group'] = pd.qcut(
            clean_df[characteristic], q=n_groups, labels=labels, duplicates='drop'
        )
    else:
        labels = [f'G{i+1}' for i in range(n_groups)]
        clean_df['group'] = pd.cut(clean_df[characteristic], bins=n_groups, labels=labels)
    
    # Calculate metrics by group
    group_metrics = []
    
    for group in clean_df['group'].cat.categories:
        group_data = clean_df[clean_df['group'] == group]
        
        if len(group_data) < 2:
            continue
        
        # Calculate metrics
        metrics = {
            'group': group,
            'n_obs': len(group_data),
            f'mean_{characteristic}': group_data[characteristic].mean(),
            f'median_{characteristic}': group_data[characteristic].median(),
            
            # Forecast accuracy
            'rmse_alpha': calculate_rmse(group_data['error_alpha'].values),
            'rmse_zero': calculate_rmse(group_data['error_zero'].values),
            'mae_alpha': calculate_mae(group_data['error_alpha'].values),
            'mae_zero': calculate_mae(group_data['error_zero'].values),
        }
        
        # Improvement metrics
        if metrics['rmse_zero'] > 0:
            metrics['rmse_improvement_pct'] = (
                (metrics['rmse_zero'] - metrics['rmse_alpha']) / 
                metrics['rmse_zero'] * 100
            )
        else:
            metrics['rmse_improvement_pct'] = 0
        
        # Statistical test within group
        if len(group_data) > 1:
            t_stat, p_val = paired_t_test(
                group_data['error_alpha'].values,
                group_data['error_zero'].values
            )
            metrics['t_statistic'] = t_stat
            metrics['p_value'] = p_val
        
        # Model parameters if available
        if 'alpha' in group_data.columns:
            metrics['mean_alpha'] = group_data['alpha'].mean() * 100  # Convert to %
            metrics['std_alpha'] = group_data['alpha'].std() * 100
        
        if 'beta' in group_data.columns:
            metrics['mean_beta'] = group_data['beta'].mean()
            metrics['std_beta'] = group_data['beta'].std()
        
        group_metrics.append(metrics)
    
    return pd.DataFrame(group_metrics)


def analyze_alpha_subset(results_df: pd.DataFrame, 
                        percentile_cutoff: int = 50,
                        use_absolute: bool = True) -> Dict[str, any]:
    """
    Analyze forecast performance for high-alpha subset.
    
    Args:
        results_df: DataFrame with results including 'alpha' column
        percentile_cutoff: Percentile cutoff for high alpha
        use_absolute: Whether to use absolute value of alpha
    
    Returns:
        Dictionary with subset analysis results
    """
    if 'alpha' not in results_df.columns:
        raise ValueError("Results must include 'alpha' column")
    
    # Calculate alpha measure
    if use_absolute:
        alpha_measure = results_df['alpha'].abs()
    else:
        alpha_measure = results_df['alpha']
    
    # Get threshold
    threshold = alpha_measure.quantile(percentile_cutoff / 100)
    
    # Split data
    high_alpha_mask = alpha_measure >= threshold
    high_alpha = results_df[high_alpha_mask].copy()
    low_alpha = results_df[~high_alpha_mask].copy()
    
    # Calculate metrics for each group
    def calc_metrics(df, label):
        if len(df) == 0:
            return None
        
        errors_alpha = df['error_alpha'].values
        errors_zero = df['error_zero'].values
        
        # Basic metrics
        rmse_alpha = calculate_rmse(errors_alpha)
        rmse_zero = calculate_rmse(errors_zero)
        mae_alpha = calculate_mae(errors_alpha)
        mae_zero = calculate_mae(errors_zero)
        
        # Test difference
        if len(df) > 1:
            t_stat, p_val = paired_t_test(errors_alpha, errors_zero)
            dm_stat, dm_pval, dm_details = diebold_mariano_test(
                errors_alpha, errors_zero, horizon=1
            )
        else:
            t_stat, p_val = np.nan, np.nan
            dm_stat, dm_pval = np.nan, np.nan
            dm_details = {}
        
        return {
            'label': label,
            'n': len(df),
            'mean_alpha_pct': df['alpha'].mean() * 100,
            'mean_abs_alpha_pct': df['alpha'].abs().mean() * 100,
            'rmse_alpha': rmse_alpha,
            'rmse_zero': rmse_zero,
            'rmse_improvement_pct': (rmse_zero - rmse_alpha) / rmse_zero * 100 if rmse_zero > 0 else 0,
            'mae_alpha': mae_alpha,
            'mae_zero': mae_zero,
            'mae_improvement_pct': (mae_zero - mae_alpha) / mae_zero * 100 if mae_zero > 0 else 0,
            't_statistic': t_stat,
            'p_value': p_val,
            'dm_statistic': dm_stat,
            'dm_pvalue': dm_pval,
            'significant_improvement': p_val < 0.05 and rmse_alpha < rmse_zero
        }
    
    # Get metrics for both groups
    high_metrics = calc_metrics(high_alpha, f'High Alpha (≥{percentile_cutoff}th pct)')
    low_metrics = calc_metrics(low_alpha, f'Low Alpha (<{percentile_cutoff}th pct)')
    
    # Test difference in improvements
    if high_metrics and low_metrics and len(high_alpha) > 10 and len(low_alpha) > 10:
        # Bootstrap test for difference
        boot_results = bootstrap_comparison_groups(
            high_alpha[['error_alpha', 'error_zero']].values,
            low_alpha[['error_alpha', 'error_zero']].values,
            n_bootstrap=1000
        )
    else:
        boot_results = None
    
    return {
        'threshold': threshold,
        'percentile_cutoff': percentile_cutoff,
        'high_alpha_group': high_metrics,
        'low_alpha_group': low_metrics,
        'bootstrap_comparison': boot_results,
        'conclusion': generate_subset_conclusion(high_metrics, low_metrics, boot_results)
    }


def bootstrap_comparison_groups(group1_errors: np.ndarray, group2_errors: np.ndarray,
                              n_bootstrap: int = 1000,
                              random_seed: Optional[int] = None) -> Dict[str, float]:
    """
    Bootstrap comparison of improvement between two groups.
    
    Args:
        group1_errors: Array with columns [error_alpha, error_zero] for group 1
        group2_errors: Array with columns [error_alpha, error_zero] for group 2
    
    Returns:
        Dictionary with bootstrap comparison results
    """
    rng = np.random.default_rng(random_seed)

    improvements_diff = []

    for _ in range(n_bootstrap):
        idx1 = rng.choice(len(group1_errors), len(group1_errors), replace=True)
        g1_alpha = group1_errors[idx1, 0]
        g1_zero = group1_errors[idx1, 1]
        g1_rmse_alpha = calculate_rmse(g1_alpha)
        g1_rmse_zero = calculate_rmse(g1_zero)
        g1_improvement = (g1_rmse_zero - g1_rmse_alpha) / g1_rmse_zero * 100 if g1_rmse_zero > 0 else 0

        idx2 = rng.choice(len(group2_errors), len(group2_errors), replace=True)
        g2_alpha = group2_errors[idx2, 0]
        g2_zero = group2_errors[idx2, 1]
        g2_rmse_alpha = calculate_rmse(g2_alpha)
        g2_rmse_zero = calculate_rmse(g2_zero)
        g2_improvement = (g2_rmse_zero - g2_rmse_alpha) / g2_rmse_zero * 100 if g2_rmse_zero > 0 else 0
        
        improvements_diff.append(g1_improvement - g2_improvement)
    
    improvements_diff = np.array(improvements_diff)
    
    return {
        'mean_diff': np.mean(improvements_diff),
        'std_diff': np.std(improvements_diff),
        'ci_lower': np.percentile(improvements_diff, 2.5),
        'ci_upper': np.percentile(improvements_diff, 97.5),
        'p_value': 2 * min(np.mean(improvements_diff > 0), np.mean(improvements_diff < 0)),
        'significant': (np.percentile(improvements_diff, 2.5) > 0) or (np.percentile(improvements_diff, 97.5) < 0)
    }


def generate_subset_conclusion(high_metrics: dict, low_metrics: dict, 
                             boot_results: Optional[dict]) -> str:
    """Generate conclusion text for subset analysis."""
    if not high_metrics or not low_metrics:
        return "Insufficient data for comparison"
    
    high_imp = high_metrics['rmse_improvement_pct']
    low_imp = low_metrics['rmse_improvement_pct']
    
    if boot_results and boot_results['significant']:
        if boot_results['mean_diff'] > 0:
            return (f"High-alpha stocks benefit significantly more from including alpha. "
                   f"Improvement difference: {boot_results['mean_diff']:.2f}% "
                   f"(95% CI: [{boot_results['ci_lower']:.2f}%, {boot_results['ci_upper']:.2f}%])")
        else:
            return (f"Low-alpha stocks benefit more from including alpha. "
                   f"This surprising result suggests alpha estimation adds noise.")
    else:
        return (f"No significant difference between high and low alpha groups. "
               f"High: {high_imp:.2f}% improvement, Low: {low_imp:.2f}% improvement")


# === SECTION 5: FORECAST HORIZON ANALYSIS ===

def analyze_horizon_effects(results_by_horizon: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze how forecast performance changes with horizon.
    
    Args:
        results_by_horizon: Dictionary mapping horizon to results DataFrame
    
    Returns:
        DataFrame with metrics by horizon
    """
    horizon_metrics = []
    
    for horizon in sorted(results_by_horizon.keys()):
        results = results_by_horizon[horizon]
        
        if len(results) == 0:
            continue
        
        # Extract errors
        errors_alpha = results['error_alpha'].values
        errors_zero = results['error_zero'].values
        
        # Calculate metrics
        metrics = {
            'horizon': horizon,
            'n_obs': len(results),
            
            # RMSE
            'rmse_alpha': calculate_rmse(errors_alpha),
            'rmse_zero': calculate_rmse(errors_zero),
            
            # MAE
            'mae_alpha': calculate_mae(errors_alpha),
            'mae_zero': calculate_mae(errors_zero),
            
            # Mean errors (bias)
            'mean_error_alpha': np.mean(errors_alpha),
            'mean_error_zero': np.mean(errors_zero),
            
            # Standard deviation of errors
            'std_error_alpha': np.std(errors_alpha),
            'std_error_zero': np.std(errors_zero),
        }
        
        # Improvement metrics
        if metrics['rmse_zero'] > 0:
            metrics['rmse_improvement_pct'] = (
                (metrics['rmse_zero'] - metrics['rmse_alpha']) / 
                metrics['rmse_zero'] * 100
            )
        
        # Statistical tests
        if len(results) > 1:
            # Paired t-test on absolute error differences for this horizon
            diff = np.abs(errors_alpha) - np.abs(errors_zero)
            t_stat, p_val = stats.ttest_1samp(diff, 0.0, nan_policy='omit')
            metrics['t_statistic'] = t_stat
            metrics['p_value'] = p_val

            # Diebold-Mariano test
            dm_stat, dm_pval, dm_details = diebold_mariano_test(
                errors_alpha, errors_zero, horizon=horizon
            )
            metrics['dm_statistic'] = dm_stat
            metrics['dm_pvalue'] = dm_pval

            # Sign test
            sign_stat, sign_pval = sign_test(errors_alpha, errors_zero)
            metrics['sign_test_stat'] = sign_stat
            metrics['sign_test_pval'] = sign_pval
        
        # Model parameters summary
        if 'alpha' in results.columns:
            metrics['mean_alpha_pct'] = results['alpha'].mean() * 100
            metrics['pct_positive_alpha'] = (results['alpha'] > 0).mean() * 100
        
        horizon_metrics.append(metrics)
    
    return pd.DataFrame(horizon_metrics)


# === SECTION 6: COMPREHENSIVE EVALUATION REPORT ===

def create_evaluation_report(results_df: pd.DataFrame, 
                           horizon: int = 1,
                           save_path: Optional[str] = None) -> str:
    """
    Create comprehensive evaluation report with all tests and metrics.
    
    Args:
        results_df: DataFrame with forecast results
        horizon: Forecast horizon
        save_path: Optional path to save report
    
    Returns:
        Formatted report string
    """
    report = f"\n{'='*70}\n"
    report += f"FORECAST EVALUATION REPORT - {horizon}-Day Horizon\n"
    report += f"{'='*70}\n\n"
    
    # Extract errors
    errors_alpha, errors_zero = calculate_forecast_errors(results_df.to_dict('records'))
    n_obs = len(errors_alpha)
    
    # 1. Basic Statistics
    report += "1. BASIC STATISTICS\n"
    report += "-" * 30 + "\n"
    report += f"Number of observations: {n_obs}\n"
    report += f"Number of unique stocks: {results_df['permno'].nunique()}\n"
    report += f"Date range: {results_df['forecast_date'].min()} to {results_df['forecast_date'].max()}\n"
    
    # 2. Forecast Accuracy Metrics
    report += "\n2. FORECAST ACCURACY METRICS\n"
    report += "-" * 30 + "\n"
    
    # RMSE
    rmse_alpha = calculate_rmse(errors_alpha)
    rmse_zero = calculate_rmse(errors_zero)
    rmse_improvement = (rmse_zero - rmse_alpha) / rmse_zero * 100 if rmse_zero > 0 else 0
    
    report += f"Root Mean Squared Error (RMSE):\n"
    report += f"  With alpha:    {format_percentage(rmse_alpha)}\n"
    report += f"  Without alpha: {format_percentage(rmse_zero)}\n"
    report += f"  Improvement:   {rmse_improvement:.2f}%\n\n"
    
    # MAE
    mae_alpha = calculate_mae(errors_alpha)
    mae_zero = calculate_mae(errors_zero)
    mae_improvement = (mae_zero - mae_alpha) / mae_zero * 100 if mae_zero > 0 else 0
    
    report += f"Mean Absolute Error (MAE):\n"
    report += f"  With alpha:    {format_percentage(mae_alpha)}\n"
    report += f"  Without alpha: {format_percentage(mae_zero)}\n"
    report += f"  Improvement:   {mae_improvement:.2f}%\n"
    
    # 3. Statistical Tests
    report += "\n3. STATISTICAL TESTS\n"
    report += "-" * 30 + "\n"
    
    # Paired t-test
    t_stat, p_val = paired_t_test(errors_alpha, errors_zero)
    report += f"Paired t-test:\n"
    report += f"  H0: E[error_alpha] = E[error_zero]\n"
    report += f"  t-statistic: {t_stat:.4f}\n"
    report += f"  p-value: {p_val:.4f}\n"
    report += f"  Significant at 5%: {'Yes' if p_val < 0.05 else 'No'}\n\n"
    
    # Diebold-Mariano test
    dm_stat, dm_pval, dm_details = diebold_mariano_test(errors_alpha, errors_zero, horizon=horizon)
    report += f"Diebold-Mariano test:\n"
    report += f"  DM statistic: {dm_stat:.4f}\n"
    report += f"  p-value: {dm_pval:.4f}\n"
    report += f"  {dm_details['interpretation']}\n"
    report += f"  {dm_details['conclusion']}\n\n"
    
    # Sign test
    sign_stat, sign_pval = sign_test(errors_alpha, errors_zero)
    report += f"Sign test:\n"
    report += f"  Proportion where |error_alpha| < |error_zero|: {sign_stat:.3f}\n"
    report += f"  p-value: {sign_pval:.4f}\n"
    
    # 4. Bootstrap Analysis
    report += "\n4. BOOTSTRAP CONFIDENCE INTERVALS (95%)\n"
    report += "-" * 30 + "\n"
    
    ci_alpha = bootstrap_rmse(errors_alpha, n_bootstrap=1000)
    ci_zero = bootstrap_rmse(errors_zero, n_bootstrap=1000)
    
    report += f"RMSE with alpha:    {format_percentage(rmse_alpha)} "
    report += f"[{format_percentage(ci_alpha[0])}, {format_percentage(ci_alpha[1])}]\n"
    report += f"RMSE without alpha: {format_percentage(rmse_zero)} "
    report += f"[{format_percentage(ci_zero[0])}, {format_percentage(ci_zero[1])}]\n"
    
    # 5. Error Distribution Analysis
    report += "\n5. ERROR DISTRIBUTION ANALYSIS\n"
    report += "-" * 30 + "\n"
    
    # Normality tests
    _, norm_p_alpha = stats.jarque_bera(errors_alpha)
    _, norm_p_zero = stats.jarque_bera(errors_zero)
    
    report += f"Jarque-Bera normality test:\n"
    report += f"  With alpha p-value:    {norm_p_alpha:.4f} "
    report += f"({'Normal' if norm_p_alpha > 0.05 else 'Non-normal'})\n"
    report += f"  Without alpha p-value: {norm_p_zero:.4f} "
    report += f"({'Normal' if norm_p_zero > 0.05 else 'Non-normal'})\n\n"
    
    # Error statistics
    report += f"Error statistics:\n"
    report += f"  Mean error (bias):\n"
    report += f"    With alpha:    {format_percentage(np.mean(errors_alpha))}\n"
    report += f"    Without alpha: {format_percentage(np.mean(errors_zero))}\n"
    report += f"  Skewness:\n"
    report += f"    With alpha:    {stats.skew(errors_alpha):.3f}\n"
    report += f"    Without alpha: {stats.skew(errors_zero):.3f}\n"
    report += f"  Kurtosis:\n"
    report += f"    With alpha:    {stats.kurtosis(errors_alpha):.3f}\n"
    report += f"    Without alpha: {stats.kurtosis(errors_zero):.3f}\n"
    
    # 6. Model Parameters Summary
    if 'alpha' in results_df.columns and 'beta' in results_df.columns:
        report += "\n6. MODEL PARAMETERS SUMMARY\n"
        report += "-" * 30 + "\n"
        report += f"Alpha:\n"
        report += f"  Mean:  {format_percentage(results_df['alpha'].mean())}\n"
        report += f"  Std:   {format_percentage(results_df['alpha'].std())}\n"
        report += f"  % > 0: {(results_df['alpha'] > 0).mean() * 100:.1f}%\n"
        
        if 'p_alpha' in results_df.columns:
            sig_alpha = (results_df['p_alpha'] < 0.05).mean() * 100
            report += f"  % significant at 5%: {sig_alpha:.1f}%\n"
        
        report += f"\nBeta:\n"
        report += f"  Mean: {results_df['beta'].mean():.3f}\n"
        report += f"  Std:  {results_df['beta'].std():.3f}\n"
        report += f"  % > 1: {(results_df['beta'] > 1).mean() * 100:.1f}%\n"

    if 'alpha' in results_df.columns:
        report += "\n7. ANNUALIZED ALPHA ANALYSIS\n"
        report += "-" * 30 + "\n"

        mean_daily_alpha = results_df['alpha'].mean()
        median_daily_alpha = results_df['alpha'].median()

        mean_annual_simple = mean_daily_alpha * 252
        median_annual_simple = median_daily_alpha * 252

        mean_annual_compound = (1 + mean_daily_alpha) ** 252 - 1
        median_annual_compound = (1 + median_daily_alpha) ** 252 - 1

        report += f"Daily Alpha:\n"
        report += f"  Mean:   {mean_daily_alpha*100:>7.4f}%\n"
        report += f"  Median: {median_daily_alpha*100:>7.4f}%\n\n"

        report += f"Annualized Alpha (Simple):\n"
        report += f"  Mean:   {mean_annual_simple*100:>7.2f}%\n"
        report += f"  Median: {median_annual_simple*100:>7.2f}%\n\n"

        report += f"Annualized Alpha (Compound):\n"
        report += f"  Mean:   {mean_annual_compound*100:>7.2f}%\n"
        report += f"  Median: {median_annual_compound*100:>7.2f}%\n\n"

        report += f"Distribution of Annualized Alphas:\n"
        annual_alphas = results_df['alpha'] * 252 * 100
        report += f"  < -10%:  {(annual_alphas < -10).sum():>3d} ({(annual_alphas < -10).mean()*100:>5.1f}%)\n"
        report += f"  -10 to 0%: {((annual_alphas >= -10) & (annual_alphas < 0)).sum():>3d} ({((annual_alphas >= -10) & (annual_alphas < 0)).mean()*100:>5.1f}%)\n"
        report += f"  0 to 10%:  {((annual_alphas >= 0) & (annual_alphas < 10)).sum():>3d} ({((annual_alphas >= 0) & (annual_alphas < 10)).mean()*100:>5.1f}%)\n"
        report += f"  > 10%:   {(annual_alphas >= 10).sum():>3d} ({(annual_alphas >= 10).mean()*100:>5.1f}%)\n"

        if 'p_alpha' in results_df.columns:
            sig_5pct = (results_df['p_alpha'] < 0.05).sum()
            sig_1pct = (results_df['p_alpha'] < 0.01).sum()
            report += f"\nStatistical Significance:\n"
            report += f"  Significant at 5%: {sig_5pct} ({sig_5pct/len(results_df)*100:.1f}%)\n"
            report += f"  Significant at 1%: {sig_1pct} ({sig_1pct/len(results_df)*100:.1f}%)\n"

    report += "\n" + "="*70 + "\n"
    
    # Save if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    
    return report


# === SECTION 7: VALUE-WEIGHTED ANALYSIS ===

def calculate_vw_statistics(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate value-weighted statistics for beta and forecast errors.
    
    Args:
        results_df: DataFrame with results including market_cap
    
    Returns:
        Dictionary with VW statistics
    """
    if 'market_cap' not in results_df.columns:
        return {'error': 'Market cap data not available'}
    
    # Remove missing market caps
    vw_data = results_df.dropna(subset=['market_cap']).copy()
    
    if len(vw_data) == 0:
        return {'error': 'No valid market cap data'}
    
    # Calculate weights
    total_cap = vw_data['market_cap'].sum()
    vw_data['weight'] = vw_data['market_cap'] / total_cap
    
    # Basic VW statistics
    vw_stats = {
        'n_stocks': len(vw_data),
        'total_market_cap': total_cap,
        'weight_concentration': vw_data.nlargest(10, 'weight')['weight'].sum(),
    }
    
    # VW model parameters
    if 'alpha' in vw_data.columns:
        vw_stats['vw_alpha'] = (vw_data['alpha'] * vw_data['weight']).sum()
        vw_stats['ew_alpha'] = vw_data['alpha'].mean()
        vw_stats['alpha_diff'] = vw_stats['vw_alpha'] - vw_stats['ew_alpha']
    
    if 'beta' in vw_data.columns:
        vw_stats['vw_beta'] = (vw_data['beta'] * vw_data['weight']).sum()
        vw_stats['ew_beta'] = vw_data['beta'].mean()
        vw_stats['beta_diff'] = vw_stats['vw_beta'] - vw_stats['ew_beta']
        
        # Check if VW beta closer to 1
        vw_stats['vw_beta_distance_from_1'] = abs(vw_stats['vw_beta'] - 1)
        vw_stats['ew_beta_distance_from_1'] = abs(vw_stats['ew_beta'] - 1)
        vw_stats['vw_closer_to_1'] = (vw_stats['vw_beta_distance_from_1'] < 
                                     vw_stats['ew_beta_distance_from_1'])
    
    # VW forecast errors
    if 'error_alpha' in vw_data.columns and 'error_zero' in vw_data.columns:
        # VW RMSE (weighted average of squared errors)
        vw_mse_alpha = (vw_data['error_alpha']**2 * vw_data['weight']).sum()
        vw_mse_zero = (vw_data['error_zero']**2 * vw_data['weight']).sum()
        
        vw_stats['vw_rmse_alpha'] = np.sqrt(vw_mse_alpha)
        vw_stats['vw_rmse_zero'] = np.sqrt(vw_mse_zero)
        vw_stats['vw_rmse_improvement'] = (
            (vw_stats['vw_rmse_zero'] - vw_stats['vw_rmse_alpha']) / 
            vw_stats['vw_rmse_zero'] * 100
        )
        
        # Compare to EW
        ew_rmse_alpha = calculate_rmse(vw_data['error_alpha'].values)
        ew_rmse_zero = calculate_rmse(vw_data['error_zero'].values)
        
        vw_stats['ew_rmse_alpha'] = ew_rmse_alpha
        vw_stats['ew_rmse_zero'] = ew_rmse_zero
        vw_stats['ew_rmse_improvement'] = (
            (ew_rmse_zero - ew_rmse_alpha) / ew_rmse_zero * 100
        )
    
    return vw_stats


# === SECTION 8: SUMMARY TABLE CREATION ===

def create_forecast_comparison_table(results_by_horizon: Dict[int, pd.DataFrame],
                                   format_type: str = 'latex') -> str:
    """
    Create publication-ready table comparing forecast performance.
    
    Args:
        results_by_horizon: Dictionary mapping horizon to results
        format_type: 'latex' or 'markdown'
    
    Returns:
        Formatted table string
    """
    # Analyze each horizon
    horizon_analysis = analyze_horizon_effects(results_by_horizon)
    
    if format_type == 'latex':
        # Create LaTeX table
        table = "\\begin{table}[ht]\n"
        table += "\\centering\n"
        table += f"\\caption{{{PRESENTATION_CONFIG['forecast_comparison_table']['caption']}}}\n"
        table += "\\label{tab:forecast_comparison}\n"
        table += "\\begin{tabular}{lcccccc}\n"
        table += "\\hline\\hline\n"
        table += "Horizon & N & RMSE ($\\alpha$) & RMSE (0) & Improvement & t-stat & DM stat \\\\\n"
        table += "\\hline\n"
        
        for _, row in horizon_analysis.iterrows():
            horizon = int(row['horizon'])
            n = int(row['n_obs'])
            rmse_a = format_percentage(row['rmse_alpha'])
            rmse_0 = format_percentage(row['rmse_zero'])
            imp = f"{row['rmse_improvement_pct']:.2f}\\%"
            
            # Add significance stars
            t_stat = f"{row['t_statistic']:.3f}"
            if row['p_value'] < 0.01:
                t_stat += "$^{***}$"
            elif row['p_value'] < 0.05:
                t_stat += "$^{**}$"
            elif row['p_value'] < 0.10:
                t_stat += "$^{*}$"
            
            dm_stat = f"{row['dm_statistic']:.3f}"
            if row['dm_pvalue'] < 0.01:
                dm_stat += "$^{***}$"
            elif row['dm_pvalue'] < 0.05:
                dm_stat += "$^{**}$"
            elif row['dm_pvalue'] < 0.10:
                dm_stat += "$^{*}$"
            
            table += f"{horizon} & {n} & {rmse_a} & {rmse_0} & {imp} & {t_stat} & {dm_stat} \\\\\n"
        
        table += "\\hline\\hline\n"
        table += "\\end{tabular}\n"
        table += "\\begin{tablenotes}\n"
        table += "\\small\n"
        table += f"\\item {PRESENTATION_CONFIG['forecast_comparison_table']['notes']}\n"
        table += "\\end{tablenotes}\n"
        table += "\\end{table}\n"
        
    else:  # Markdown format
        # Create header
        table = "| Horizon | N | RMSE (α) | RMSE (0) | Improvement | t-stat | DM stat |\n"
        table += "|---------|---|----------|----------|-------------|--------|----------|\n"
        
        for _, row in horizon_analysis.iterrows():
            table += f"| {int(row['horizon'])} "
            table += f"| {int(row['n_obs'])} "
            table += f"| {format_percentage(row['rmse_alpha'])} "
            table += f"| {format_percentage(row['rmse_zero'])} "
            table += f"| {row['rmse_improvement_pct']:.2f}% "
            table += f"| {row['t_statistic']:.3f}"
            if row['p_value'] < 0.01:
                table += "***"
            elif row['p_value'] < 0.05:
                table += "**"
            elif row['p_value'] < 0.10:
                table += "*"
            table += f" | {row['dm_statistic']:.3f}"
            if row['dm_pvalue'] < 0.01:
                table += "***"
            elif row['dm_pvalue'] < 0.05:
                table += "**"
            elif row['dm_pvalue'] < 0.10:
                table += "*"
            table += " |\n"
        
        table += "\n" + PRESENTATION_CONFIG['forecast_comparison_table']['notes']

    return table


def create_multi_horizon_summary(horizon_results: Dict[int, dict],
                                 save_path: Optional[str] = None) -> pd.DataFrame:
    """Create comprehensive summary table comparing all horizons."""

    summary_data = []

    for horizon in sorted(horizon_results.keys()):
        h_data = horizon_results[horizon]
        results_df = h_data['results_df']

        daily_alpha_mean = results_df['alpha'].mean()

        horizon_alpha = daily_alpha_mean * horizon
        horizon_alpha_compound = (1 + daily_alpha_mean) ** horizon - 1

        annual_alpha = daily_alpha_mean * 252
        annual_alpha_compound = (1 + daily_alpha_mean) ** 252 - 1

        row = {
            'Horizon (days)': horizon,
            'N': h_data['n_samples'],
            'RMSE (α) %': h_data['rmse_alpha'] * 100,
            'RMSE (0) %': h_data['rmse_zero'] * 100,
            'Improvement %': h_data['rmse_improvement_pct'],
            't-stat': h_data['t_statistic'],
            'p-value': h_data['p_value'],
            'DM-stat': h_data['dm_statistic'],
            'DM p-value': h_data['dm_pvalue'],
            'Mean α (daily) %': daily_alpha_mean * 100,
            'Median α (daily) %': results_df['alpha'].median() * 100,
            'Std α (daily) %': results_df['alpha'].std() * 100,
            f'α over {horizon}d (simple) %': horizon_alpha * 100,
            f'α over {horizon}d (compound) %': horizon_alpha_compound * 100,
            'α (annual simple) %': annual_alpha * 100,
            'α (annual compound) %': annual_alpha_compound * 100,
            'Mean β': results_df['beta'].mean(),
            'Std β': results_df['beta'].std(),
            'Mean R²': results_df['r_squared'].mean() if 'r_squared' in results_df.columns else np.nan,
            '% |α| > 0': (results_df['alpha'] != 0).mean() * 100,
            '% α > 0': (results_df['alpha'] > 0).mean() * 100,
        }

        if row['p-value'] < 0.01:
            row['Significance'] = '***'
        elif row['p-value'] < 0.05:
            row['Significance'] = '**'
        elif row['p-value'] < 0.10:
            row['Significance'] = '*'
        else:
            row['Significance'] = ''

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    if save_path:
        summary_df.to_csv(save_path, index=False)
        latex_path = save_path.replace('.csv', '.tex')
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write("\\caption{Forecast Performance Across All Horizons}\n")
            f.write("\\label{tab:all_horizons}\n")
            f.write("\\resizebox{\\textwidth}{!}{%\n")
            f.write(summary_df.to_latex(index=False, float_format='%.3f'))
            f.write("}\n")
            f.write("\\end{table}\n")

    return summary_df


# === SECTION 9: DIAGNOSTIC FUNCTIONS ===

def check_forecast_quality(results_df: pd.DataFrame) -> Dict[str, any]:
    """
    Run diagnostic checks on forecast quality.
    
    Returns:
        Dictionary with diagnostic results and warnings
    """
    diagnostics = {
        'n_obs': len(results_df),
        'warnings': [],
        'issues': []
    }
    
    # Check for systematic biases
    if 'error_alpha' in results_df.columns:
        mean_error = results_df['error_alpha'].mean()
        if abs(mean_error) > 0.001:  # 0.1% daily
            diagnostics['warnings'].append(
                f"Systematic bias detected: mean error = {mean_error*100:.3f}%"
            )
    
    # Check for outliers
    if 'error_alpha' in results_df.columns:
        errors = results_df['error_alpha'].values
        q1, q3 = np.percentile(errors, [25, 75])
        iqr = q3 - q1
        outlier_mask = (errors < q1 - 3*iqr) | (errors > q3 + 3*iqr)
        n_outliers = outlier_mask.sum()
        
        if n_outliers > len(errors) * 0.05:  # More than 5% outliers
            diagnostics['warnings'].append(
                f"High proportion of outliers: {n_outliers} ({n_outliers/len(errors)*100:.1f}%)"
            )
    
    # Check for autocorrelation in errors
    if 'error_alpha' in results_df.columns and 'forecast_date' in results_df.columns:
        # Sort by date
        sorted_df = results_df.sort_values('forecast_date')
        errors = sorted_df['error_alpha'].values
        
        if len(errors) > 20:
            # Ljung-Box test
            lb_result = acorr_ljungbox(errors, lags=min(10, len(errors)//5), return_df=True)
            if (lb_result['lb_pvalue'] < 0.05).any():
                diagnostics['warnings'].append(
                    "Significant autocorrelation detected in forecast errors"
                )
    
    # Check model parameters
    if 'beta' in results_df.columns:
        beta_issues = (results_df['beta'] < 0) | (results_df['beta'] > 3)
        if beta_issues.any():
            diagnostics['issues'].append(
                f"{beta_issues.sum()} observations with unusual betas (<0 or >3)"
            )
    
    # Check R-squared distribution
    if 'r_squared' in results_df.columns:
        low_r2 = (results_df['r_squared'] < 0.01).sum()
        if low_r2 > len(results_df) * 0.10:
            diagnostics['issues'].append(
                f"{low_r2} observations ({low_r2/len(results_df)*100:.1f}%) with R² < 0.01"
            )

    return diagnostics


# === SECTION 7: MODEL COMPARISON UTILITIES ===

def compare_models_performance(results_df: pd.DataFrame) -> Dict[str, float]:
    """Compare CAPM and FF3 forecast performance without re-estimation.

    Expects CAPM errors to be stored in ``error_capm_alpha`` and
    ``error_capm_zero``. If FF3 error columns are missing, the function
    returns a CAPM-only comparison instead of raising an exception.
    """

    # CAPM errors are expected under fixed column names
    capm_alpha_col = 'error_capm_alpha'
    capm_zero_col = 'error_capm_zero'

    # If the CAPM error columns are missing, we cannot compute any comparison
    if capm_alpha_col not in results_df.columns or capm_zero_col not in results_df.columns:
        print("\nCAPM errors not found; skipping model comparison.")
        return {}

    # If FF3 errors are missing, fall back to CAPM-only comparison
    if 'error_ff3_alpha' not in results_df.columns or 'error_ff3_zero' not in results_df.columns:
        print("\nFF3 errors not found; returning CAPM comparison only.")
        return {
            'CAPM with α': calculate_rmse(results_df[capm_alpha_col]),
            'CAPM no α': calculate_rmse(results_df[capm_zero_col]),
        }

    comparisons = {
        'CAPM with α': calculate_rmse(results_df[capm_alpha_col]),
        'CAPM no α': calculate_rmse(results_df[capm_zero_col]),
        'FF3 with α': calculate_rmse(results_df['error_ff3_alpha']),
        'FF3 no α': calculate_rmse(results_df['error_ff3_zero']),
    }

    capm_improvement = (
        (comparisons['CAPM no α'] - comparisons['CAPM with α']) /
        comparisons['CAPM no α'] * 100 if comparisons['CAPM no α'] else np.nan
    )
    ff3_improvement = (
        (comparisons['FF3 no α'] - comparisons['FF3 with α']) /
        comparisons['FF3 no α'] * 100 if comparisons['FF3 no α'] else np.nan
    )
    ff3_over_capm = (
        (comparisons['CAPM with α'] - comparisons['FF3 with α']) /
        comparisons['CAPM with α'] * 100 if comparisons['CAPM with α'] else np.nan
    )

    print("\n" + "="*60)
    print("MODEL COMPARISON MATRIX")
    print("="*60)
    print(f"{'Model':<20} {'RMSE (%)':<12} {'vs CAPM+α':<15}")
    print("-"*47)

    baseline = comparisons['CAPM with α']
    for model, rmse in comparisons.items():
        diff = (baseline - rmse) / baseline * 100 if baseline else np.nan
        print(f"{model:<20} {rmse*100:<12.4f} {diff:+.2f}%")

    print("\nKey Insights:")
    print(f"1. Alpha helps CAPM: {capm_improvement:+.2f}%")
    print(f"2. Alpha helps FF3:  {ff3_improvement:+.2f}%")
    print(f"3. FF3 beats CAPM:   {ff3_over_capm:+.2f}%")

    # Statistical test: does FF3 beat CAPM?
    t_stat, p_val = stats.ttest_rel(
        results_df['error_ff3_alpha'].abs(),
        results_df[capm_alpha_col].abs()
    )
    print(f"\nFF3 vs CAPM paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    if 'r2_increment' in results_df.columns:
        print(f"Average R² improvement from factors: {results_df['r2_increment'].mean():.3f}")
    else:
        print("Average R² improvement from factors: N/A")

    return comparisons
