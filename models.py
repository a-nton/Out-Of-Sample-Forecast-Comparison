"""
models.py - Financial models for return prediction.
Includes CAPM and multi-factor model implementations with comprehensive validation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import warnings

import statsmodels.api as sm
from scipy import stats

from config import MODEL_CONFIG, PRESENTATION_CONFIG

# === SECTION 1: DATA VALIDATION AND COLUMN CHECKING ===

def validate_columns(data: pd.DataFrame, required_cols: List[str], 
                    model_name: str = "Model") -> Dict[str, str]:
    """
    Validate that required columns exist and check their data types.
    
    Args:
        data: DataFrame to validate
        required_cols: List of required column names
        model_name: Name of model for error messages
    
    Returns:
        Dictionary mapping column names to their dtypes
    
    Raises:
        ValueError: If required columns are missing
    """
    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        available = list(data.columns)
        raise ValueError(f"{model_name} requires columns {missing_cols}, but they are missing. "
                        f"Available columns: {available}")
    
    # Get data types and validate
    col_types = {}
    for col in required_cols:
        col_types[col] = str(data[col].dtype)
        
        # Check if numeric where expected
        numeric_cols = ['RET', 'RF', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'market_cap']
        if col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                warnings.warn(
                    f"Column '{col}' has type {data[col].dtype} but numeric type expected."
                )
                try:
                    pd.to_numeric(data[col], errors='coerce')
                    col_types[col] = str(data[col].dtype) + " (coerce attempted)"
                except Exception as exc:
                    raise TypeError(f"Cannot convert column '{col}' to numeric type") from exc
    
    # Print validation summary if verbose
    if MODEL_CONFIG.get('validate_estimations', True):
        print(f"\n{model_name} Column Validation:")
        print("-" * 50)
        for col, dtype in col_types.items():
            non_missing = data[col].notna().sum()
            missing_pct = (1 - non_missing/len(data)) * 100
            print(f"{col:15} | Type: {dtype:10} | Missing: {missing_pct:5.1f}%")
    
    return col_types


def check_estimation_data_quality(data: pd.DataFrame) -> Dict[str, any]:
    """
    Check quality of estimation window data.
    
    Returns:
        Dictionary with quality metrics and warnings
    """
    quality = {
        'n_obs': len(data),
        'n_complete': data.dropna().shape[0],
        'warnings': []
    }
    
    if 'RET' in data.columns:
        ret_stats = data['RET'].dropna()
        quality['ret_mean'] = ret_stats.mean()
        quality['ret_std'] = ret_stats.std()
        
        # Check for extreme returns
        extreme_rets = (ret_stats.abs() > 0.5).sum()
        if extreme_rets > 0:
            quality['warnings'].append(f"{extreme_rets} extreme returns (>50%)")
        
        # Check for too many identical returns (possible data error)
        if len(ret_stats) > 0:
            mode_count = ret_stats.value_counts().iloc[0]
            if mode_count / len(ret_stats) > 0.1:  # More than 10% identical
                quality['warnings'].append("Many identical returns detected")
    
    # Check market data
    if 'Mkt-RF' in data.columns:
        mkt_stats = data['Mkt-RF'].dropna()
        quality['mkt_mean'] = mkt_stats.mean()
        quality['mkt_std'] = mkt_stats.std()
        
        # Market should have reasonable volatility
        if mkt_stats.std() < 0.001:
            quality['warnings'].append("Market returns have suspiciously low volatility")
    
    return quality


# === SECTION 2: MARKET CAP WEIGHTED CALCULATIONS ===

def calculate_vw_beta(results_df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate value-weighted and equal-weighted betas with diagnostics.
    Tests if VW beta is closer to 1.0 as theory suggests.
    
    Args:
        results_df: DataFrame with columns 'beta' and 'market_cap'
    
    Returns:
        Dictionary with VW/EW betas, diagnostics, and sanity checks
    """
    # Validate inputs
    if 'beta' not in results_df.columns or 'market_cap' not in results_df.columns:
        return {'error': 'Missing required columns: beta and/or market_cap'}
    
    # Remove invalid observations
    valid_mask = (
        results_df['beta'].notna() & 
        results_df['market_cap'].notna() & 
        (results_df['market_cap'] > 0) &
        (results_df['beta'] > -2) &  # Reasonable beta range
        (results_df['beta'] < 5)
    )
    
    clean_data = results_df[valid_mask].copy()
    
    if len(clean_data) == 0:
        return {'error': 'No valid beta/market cap pairs'}
    
    # Calculate equal-weighted statistics
    ew_beta = clean_data['beta'].mean()
    ew_beta_std = clean_data['beta'].std()
    
    # Calculate value-weighted statistics
    total_cap = clean_data['market_cap'].sum()
    if total_cap > 0:
        weights = clean_data['market_cap'] / total_cap
        vw_beta = (clean_data['beta'] * weights).sum()
        vw_variance = (weights * (clean_data['beta'] - vw_beta) ** 2).sum()
        vw_beta_std = np.sqrt(vw_variance)
    else:
        weights = np.zeros(len(clean_data))
        vw_beta = np.nan
        vw_beta_std = np.nan

    if clean_data['market_cap'].var() > 0 and clean_data['beta'].var() > 0:
        log_cap = np.log(clean_data['market_cap'])
        size_beta_corr = np.corrcoef(log_cap, clean_data['beta'])[0, 1]
    else:
        size_beta_corr = np.nan
    
    # Beta by size quintiles
    clean_data['size_quintile'] = pd.qcut(
        clean_data['market_cap'],
        q=5,
        labels=['Q1_Small', 'Q2', 'Q3', 'Q4', 'Q5_Large'],
        duplicates='drop'
    )
    
    beta_by_size = clean_data.groupby('size_quintile')['beta'].agg(['mean', 'std', 'count'])
    
    # Prepare results
    results = {
        # Basic statistics
        'ew_beta': ew_beta,
        'ew_beta_std': ew_beta_std,
        'vw_beta': vw_beta,
        'vw_beta_std': vw_beta_std,
        'beta_difference': ew_beta - vw_beta,
        
        # Diagnostics
        'size_beta_correlation': size_beta_corr,
        'n_valid': len(clean_data),
        'n_excluded': len(results_df) - len(clean_data),
        'beta_by_size': beta_by_size,
        
        # Market cap distribution
        'cap_concentration': (
            clean_data.nlargest(10, 'market_cap')['market_cap'].sum() / total_cap
            if total_cap > 0 else np.nan
        ),
        'median_cap': clean_data['market_cap'].median(),
        'mean_cap': clean_data['market_cap'].mean(),
        
        # Sanity checks
        'small_firm_bias': ew_beta > vw_beta,
        'vw_closer_to_one': abs(vw_beta - 1.0) < abs(ew_beta - 1.0),
    }
    
    # Detailed sanity check message
    if results['vw_closer_to_one']:
        results['sanity_check'] = 'PASS: VW beta closer to 1.0 than EW beta (as expected)'
        results['sanity_details'] = (f"VW beta = {vw_beta:.3f} (distance from 1: {abs(vw_beta-1):.3f}), "
                                    f"EW beta = {ew_beta:.3f} (distance from 1: {abs(ew_beta-1):.3f})")
    else:
        results['sanity_check'] = 'WARNING: EW beta closer to 1.0 than VW beta (unexpected)'
        results['sanity_details'] = "This might indicate data issues or unusual sample composition"
    
    # Additional diagnostics
    if size_beta_corr < -0.3:
        results['size_effect'] = "Strong negative size-beta relation (small firms have higher betas)"
    elif size_beta_corr > 0.3:
        results['size_effect'] = "Unusual positive size-beta relation"
    else:
        results['size_effect'] = "Weak size-beta relation"
    
    return results


# === SECTION 3: MODEL ESTIMATION WITH VALIDATION ===

def estimate_capm(data: pd.DataFrame, 
                  min_obs: int = 150,
                  validate_data: bool = None) -> Tuple[float, float, dict]:
    """
    Estimate CAPM model with comprehensive validation.
    R_i - R_f = alpha + beta * (R_m - R_f) + epsilon
    
    Args:
        data: DataFrame with columns RET, RF, Mkt-RF
        min_obs: Minimum observations required
        validate_data: Whether to validate (None = use config setting)
    
    Returns:
        alpha: Intercept (Jensen's alpha) 
        beta: Market beta
        stats: Comprehensive statistics and diagnostics
    """
    if validate_data is None:
        validate_data = MODEL_CONFIG.get('validate_estimations', True)
    
    # Validate columns if requested
    if validate_data:
        required_cols = ['RET', 'RF', 'Mkt-RF']
        col_types = validate_columns(data, required_cols, "CAPM")
        data_quality = check_estimation_data_quality(data)
    
    # Prepare data - ensure we have numeric arrays
    y = pd.to_numeric(data['RET'] - data['RF'], errors='coerce').values
    X = pd.to_numeric(data['Mkt-RF'], errors='coerce').values.reshape(-1, 1)
    
    # Remove missing values
    mask = ~(np.isnan(y) | np.isnan(X.flatten()))
    y_clean = y[mask]
    X_clean = X[mask]
    
    n_obs = len(y_clean)
    n_missing = len(y) - n_obs
    
    if n_obs < min_obs:
        raise ValueError(f"Insufficient observations: {n_obs} < {min_obs} "
                        f"(dropped {n_missing} missing values)")
    
    # Estimate model with HAC standard errors
    X_with_const = sm.add_constant(X_clean)
    ols = sm.OLS(y_clean, X_with_const)
    maxlags = int(np.ceil(n_obs ** (1 / 4)))  # Newey-West rule-of-thumb
    results = ols.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags, 'use_correction': True})

    alpha, beta = results.params
    residuals = results.resid
    r_squared = results.rsquared
    adj_r_squared = results.rsquared_adj

    se_alpha, se_beta = results.bse
    t_alpha = results.tvalues[0]
    t_beta = (beta - 1) / se_beta if se_beta > 0 else np.nan
    p_alpha = results.pvalues[0]
    p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), n_obs - 2)) if se_beta > 0 else np.nan

    ss_res = np.sum(residuals ** 2)

    metrics = {
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'n_obs': n_obs,
        'n_missing': n_missing,
        'se_alpha': se_alpha,
        'se_beta': se_beta,
        't_alpha': t_alpha,
        't_beta': t_beta,
        'p_alpha': p_alpha,
        'p_beta': p_beta,
        'residual_std': np.std(residuals, ddof=2),
        'residual_skew': stats.skew(residuals),
        'residual_kurtosis': stats.kurtosis(residuals),
        'aic': results.aic,
        'bic': results.bic,
        'mean_excess_return': np.mean(y_clean),
        'market_excess_mean': np.mean(X_clean),
        'market_excess_std': np.std(X_clean),
        'realized_sharpe': np.mean(y_clean) / np.std(y_clean) if np.std(y_clean) > 0 else 0,
        'market_sharpe': np.mean(X_clean) / np.std(X_clean) if np.std(X_clean) > 0 else 0,
    }

    if validate_data:
        metrics['data_quality'] = data_quality
        if data_quality['warnings']:
            metrics['quality_warnings'] = data_quality['warnings']

    if n_obs > 10:
        dw = sm.stats.stattools.durbin_watson(residuals)
        metrics['durbin_watson'] = dw
        if dw < 1.5 or dw > 2.5:
            metrics['autocorrelation_warning'] = f"Potential autocorrelation (DW = {dw:.2f})"

    return alpha, beta, metrics


def estimate_ff3(data: pd.DataFrame, 
                 min_obs: int = 150,
                 validate_data: bool = None) -> Tuple[np.ndarray, dict]:
    """
    Estimate Fama-French 3-factor model with validation.
    R_i - R_f = alpha + beta_MKT*(R_m - R_f) + beta_SMB*SMB + beta_HML*HML + epsilon
    
    Returns:
        coefficients: [alpha, beta_mkt, beta_smb, beta_hml]
        stats: Model statistics including incremental R² over CAPM
    """
    if validate_data is None:
        validate_data = MODEL_CONFIG.get('validate_estimations', True)
    
    # Validate columns
    if validate_data:
        required_cols = ['RET', 'RF', 'Mkt-RF', 'SMB', 'HML']
        col_types = validate_columns(data, required_cols, "FF3")
    
    # Prepare data
    y = pd.to_numeric(data['RET'] - data['RF'], errors='coerce').values
    factor_cols = ['Mkt-RF', 'SMB', 'HML']
    X = data[factor_cols].apply(pd.to_numeric, errors='coerce').values
    
    # Remove missing values
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y_clean = y[mask]
    X_clean = X[mask]
    
    n_obs = len(y_clean)
    if n_obs < min_obs:
        raise ValueError(f"Insufficient observations: {n_obs} < {min_obs}")
    
    # Estimate model with HAC standard errors
    X_with_const = sm.add_constant(X_clean)
    ols = sm.OLS(y_clean, X_with_const)
    maxlags = int(np.ceil(n_obs ** (1 / 4)))
    results = ols.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags, 'use_correction': True})

    coefficients = results.params
    residuals = results.resid
    r_squared = results.rsquared

    # Model comparison with CAPM
    X_capm = X_clean[:, 0].reshape(-1, 1)
    capm_results = sm.OLS(y_clean, sm.add_constant(X_capm)).fit(
        cov_type='HAC', cov_kwds={'maxlags': maxlags, 'use_correction': True}
    )
    r_squared_capm = capm_results.rsquared

    se_coefficients = results.bse
    t_statistics = results.tvalues
    p_values = results.pvalues

    n = len(y_clean)
    k = len(coefficients)
    ss_res = np.sum(residuals ** 2)
    
    # Factor correlations and VIF
    factor_corr = np.corrcoef(X_clean.T)
    
    # Variance Inflation Factors
    vif = {}
    for i, factor in enumerate(factor_cols):
        X_others = np.delete(X_clean, i, axis=1)
        reg = sm.OLS(X_clean[:, i], sm.add_constant(X_others)).fit()
        r2 = reg.rsquared
        vif[factor] = 1 / (1 - r2) if r2 < 0.999 else np.inf
    
    metrics = {
        'r_squared': r_squared,
        'adj_r_squared': 1 - (1 - r_squared) * (n - 1) / (n - k),
        'r_squared_capm': r_squared_capm,
        'incremental_r_squared': r_squared - r_squared_capm,
        'n_obs': n_obs,
        'n_missing': len(y) - n_obs,
        'factor_names': ['alpha', 'beta_mkt', 'beta_smb', 'beta_hml'],
        'coefficients': coefficients,
        'se_coefficients': se_coefficients,
        't_statistics': t_statistics,
        'p_values': p_values,
        'residual_std': np.std(residuals, ddof=k),
        'aic': results.aic,
        'bic': results.bic,
        'factor_correlations': factor_corr,
        'vif': vif,
        'max_vif': max(vif.values()),
        'factor_risk_premia': {
            'mkt': coefficients[1] * np.mean(X_clean[:, 0]),
            'smb': coefficients[2] * np.mean(X_clean[:, 1]),
            'hml': coefficients[3] * np.mean(X_clean[:, 2]),
        },
    }

    if metrics['max_vif'] > 10:
        metrics['multicollinearity_warning'] = f"High VIF detected: {metrics['max_vif']:.1f}"

    return coefficients, metrics


# === SECTION 4: FORECAST GENERATION ===

def forecast_capm_return(alpha: float, beta: float, mkt_excess: float, 
                        rf: float, use_alpha: bool = True) -> float:
    """
    Generate return forecast using CAPM.
    
    Returns:
        Forecasted return (in same units as input)
    """
    if use_alpha:
        return alpha + beta * mkt_excess + rf
    else:
        return beta * mkt_excess + rf


def forecast_ff3_return(coefficients: np.ndarray, factors: dict, 
                       rf: float, use_alpha: bool = True) -> float:
    """
    Generate return forecast using FF3 model.
    
    Args:
        coefficients: [alpha, beta_mkt, beta_smb, beta_hml]
        factors: Dict with 'Mkt-RF', 'SMB', 'HML' values
        rf: Risk-free rate
        use_alpha: Whether to include alpha
    
    Returns:
        Forecasted return
    """
    required_factors = ['Mkt-RF', 'SMB', 'HML']
    missing = [f for f in required_factors if f not in factors]
    if missing:
        raise ValueError(f"Missing factors: {missing}")
    
    alpha = coefficients[0] if use_alpha else 0
    forecast = alpha + rf
    
    for i, factor in enumerate(required_factors):
        forecast += coefficients[i + 1] * factors[factor]

    return forecast

# === SECTION 5: ROLLING ESTIMATION ===

def rolling_capm(data: pd.DataFrame, window: int, step: int = 1) -> pd.DataFrame:
    """Generate rolling CAPM estimates over the provided data."""
    results = []
    for start in range(0, len(data) - window + 1, step):
        window_df = data.iloc[start:start + window]
        try:
            alpha, beta, metrics = estimate_capm(window_df, min_obs=window, validate_data=False)
        except ValueError:
            continue
        metrics.update({
            'start': window_df['date'].iloc[0],
            'end': window_df['date'].iloc[-1],
            'alpha': alpha,
            'beta': beta,
        })
        results.append(metrics)
    return pd.DataFrame(results)

# === SECTION 6: ALPHA ANALYSIS ===

def analyze_alpha_persistence(results_df: pd.DataFrame, 
                            percentile_cutoff: int = 50) -> Dict[str, any]:
    """
    Analyze whether high-alpha stocks benefit more from including alpha in forecasts.
    Tests the hypothesis that alpha estimation is more valuable when true alpha is large.
    
    Args:
        results_df: DataFrame with 'alpha', 'error_alpha', 'error_zero' columns
        percentile_cutoff: Percentile for high-alpha classification
    
    Returns:
        Comprehensive analysis of alpha effects
    """
    # Ensure we have required columns
    required = ['alpha', 'error_alpha', 'error_zero']
    missing = [col for col in required if col not in results_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate absolute alpha for ranking
    results_df = results_df.copy()
    results_df['abs_alpha'] = results_df['alpha'].abs()
    
    # Split by alpha magnitude
    threshold = results_df['abs_alpha'].quantile(percentile_cutoff / 100)
    high_alpha_mask = results_df['abs_alpha'] >= threshold
    
    high_alpha = results_df[high_alpha_mask]
    low_alpha = results_df[~high_alpha_mask]
    
    # Calculate metrics for each group
    def calc_group_metrics(df, label):
        if len(df) == 0:
            return None
            
        rmse_alpha = np.sqrt(np.mean(df['error_alpha']**2))
        rmse_zero = np.sqrt(np.mean(df['error_zero']**2))
        mae_alpha = np.mean(np.abs(df['error_alpha']))
        mae_zero = np.mean(np.abs(df['error_zero']))
        
        # Paired differences
        diff = df['error_alpha'].abs() - df['error_zero'].abs()
        
        # Test if including alpha helps (negative difference = improvement)
        if len(diff) > 1:
            t_stat, p_val = stats.ttest_1samp(diff, 0)
        else:
            t_stat, p_val = np.nan, np.nan
        
        return {
            'label': label,
            'n': len(df),
            'mean_abs_alpha': df['abs_alpha'].mean(),
            'mean_alpha': df['alpha'].mean(),
            'std_alpha': df['alpha'].std(),
            'rmse_alpha': rmse_alpha,
            'rmse_zero': rmse_zero,
            'rmse_improvement': (rmse_zero - rmse_alpha) / rmse_zero * 100 if rmse_zero > 0 else 0,
            'mae_alpha': mae_alpha,
            'mae_zero': mae_zero,
            'mae_improvement': (mae_zero - mae_alpha) / mae_zero * 100 if mae_zero > 0 else 0,
            'mean_improvement': -np.mean(diff),  # Negative because lower error is better
            't_stat': t_stat,
            'p_value': p_val,
            'helps': p_val < 0.05 and np.mean(diff) < 0
        }
    
    # Get metrics for both groups
    high_metrics = calc_group_metrics(high_alpha, f'High Alpha (>{percentile_cutoff}th percentile)')
    low_metrics = calc_group_metrics(low_alpha, f'Low Alpha (<{percentile_cutoff}th percentile)')
    
    # Test difference in improvements
    if high_metrics and low_metrics and len(high_alpha) > 1 and len(low_alpha) > 1:
        # Bootstrap test for difference in improvements
        n_boot = 1000
        boot_diffs = []
        
        for _ in range(n_boot):
            # Bootstrap high alpha group
            h_idx = np.random.choice(len(high_alpha), len(high_alpha), replace=True)
            h_rmse_a = np.sqrt(np.mean(high_alpha.iloc[h_idx]['error_alpha']**2))
            h_rmse_0 = np.sqrt(np.mean(high_alpha.iloc[h_idx]['error_zero']**2))
            h_imp = (h_rmse_0 - h_rmse_a) / h_rmse_0 * 100 if h_rmse_0 > 0 else 0
            
            # Bootstrap low alpha group
            l_idx = np.random.choice(len(low_alpha), len(low_alpha), replace=True)
            l_rmse_a = np.sqrt(np.mean(low_alpha.iloc[l_idx]['error_alpha']**2))
            l_rmse_0 = np.sqrt(np.mean(low_alpha.iloc[l_idx]['error_zero']**2))
            l_imp = (l_rmse_0 - l_rmse_a) / l_rmse_0 * 100 if l_rmse_0 > 0 else 0
            
            boot_diffs.append(h_imp - l_imp)
        
        # Calculate confidence interval for difference
        diff_mean = np.mean(boot_diffs)
        diff_ci = np.percentile(boot_diffs, [2.5, 97.5])
        diff_significant = (diff_ci[0] > 0) or (diff_ci[1] < 0)
    else:
        diff_mean = np.nan
        diff_ci = [np.nan, np.nan]
        diff_significant = False
    
    # Create summary
    summary = {
        'threshold': threshold,
        'percentile_cutoff': percentile_cutoff,
        'high_alpha_group': high_metrics,
        'low_alpha_group': low_metrics,
        'improvement_difference': diff_mean,
        'improvement_diff_ci': diff_ci,
        'difference_significant': diff_significant,
        'conclusion': None
    }
    
    # Generate conclusion
    if high_metrics and low_metrics:
        if diff_significant and diff_mean > 0:
            summary['conclusion'] = (f"High-alpha stocks benefit significantly more from including alpha. "
                                   f"Improvement difference: {diff_mean:.2f}% "
                                   f"(95% CI: [{diff_ci[0]:.2f}%, {diff_ci[1]:.2f}%])")
        elif diff_significant and diff_mean < 0:
            summary['conclusion'] = "Surprisingly, low-alpha stocks benefit more from including alpha."
        else:
            summary['conclusion'] = f"No significant difference in benefit between high and low alpha stocks (p > 0.05)"
    
    return summary


# === SECTION 6: MODEL DIAGNOSTICS ===

def diagnose_estimation_quality(alpha: float, beta: float, stats: dict) -> Dict[str, str]:
    """
    Provide comprehensive diagnostics about estimation quality.
    Identifies potential issues that could affect forecast accuracy.
    
    Returns:
        Dictionary with diagnostic messages and severity levels
    """
    diagnostics = {}
    
    # R-squared check
    r2 = stats.get('r_squared', 0)
    if r2 < 0.01:
        diagnostics['r_squared'] = {
            'severity': 'HIGH',
            'message': f'Extremely low R² ({r2:.3f}), model explains <1% of variation'
        }
    elif r2 < 0.05:
        diagnostics['r_squared'] = {
            'severity': 'MEDIUM',
            'message': f'Very low R² ({r2:.3f}), model has weak explanatory power'
        }
    
    # Beta reasonableness
    if beta < -0.5:
        diagnostics['beta_range'] = {
            'severity': 'HIGH',
            'message': f'Strongly negative beta ({beta:.3f}), unusual except for gold/defensive assets'
        }
    elif beta < 0:
        diagnostics['beta_range'] = {
            'severity': 'MEDIUM',
            'message': f'Negative beta ({beta:.3f}), uncommon for equity securities'
        }
    elif beta > 3:
        diagnostics['beta_range'] = {
            'severity': 'MEDIUM',
            'message': f'Very high beta ({beta:.3f}), suggests high leverage or extreme volatility'
        }
    elif beta > 5:
        diagnostics['beta_range'] = {
            'severity': 'HIGH',
            'message': f'Extreme beta ({beta:.3f}), likely data error or distressed security'
        }
    
    # Alpha significance and magnitude
    if 't_alpha' in stats and abs(stats['t_alpha']) > 3:
        alpha_pct = alpha * 100  # Convert to percentage
        diagnostics['alpha_significance'] = {
            'severity': 'INFO',
            'message': f'Highly significant alpha ({alpha_pct:.3f}% daily, t={stats["t_alpha"]:.2f})'
        }
    
    # Sample size concerns
    n_obs = stats.get('n_obs', 0)
    if n_obs < 200:
        diagnostics['sample_size'] = {
            'severity': 'MEDIUM',
            'message': f'Limited observations ({n_obs}), estimates may be imprecise'
        }
    elif n_obs == 150:  # Exactly at minimum
        diagnostics['sample_size'] = {
            'severity': 'LOW',
            'message': f'Minimum observations ({n_obs}), consider requiring more data'
        }
    
    # Missing data concerns
    missing_pct = stats.get('n_missing', 0) / (stats.get('n_obs', 1) + stats.get('n_missing', 0)) * 100
    if missing_pct > 40:
        diagnostics['missing_data'] = {
            'severity': 'HIGH',
            'message': f'Excessive missing data ({missing_pct:.1f}%), results may be biased'
        }
    elif missing_pct > 20:
        diagnostics['missing_data'] = {
            'severity': 'MEDIUM',
            'message': f'Substantial missing data ({missing_pct:.1f}%)'
        }
    
    # Residual diagnostics
    if 'residual_skew' in stats:
        skew = stats['residual_skew']
        if abs(skew) > 2:
            diagnostics['residual_skew'] = {
                'severity': 'MEDIUM',
                'message': f'Highly skewed residuals ({skew:.2f}), violates normality assumption'
            }
    
    if 'residual_kurtosis' in stats:
        kurt = stats['residual_kurtosis']
        if kurt > 5:
            diagnostics['residual_kurtosis'] = {
                'severity': 'MEDIUM',
                'message': f'Heavy-tailed residuals (kurtosis={kurt:.1f}), more extreme events than normal'
            }
    
    # Autocorrelation check
    if 'durbin_watson' in stats:
        dw = stats['durbin_watson']
        if dw < 1.5:
            diagnostics['autocorrelation'] = {
                'severity': 'MEDIUM',
                'message': f'Positive autocorrelation detected (DW={dw:.2f}), may underestimate standard errors'
            }
        elif dw > 2.5:
            diagnostics['autocorrelation'] = {
                'severity': 'MEDIUM',
                'message': f'Negative autocorrelation detected (DW={dw:.2f})'
            }
    
    # Data quality warnings
    if 'quality_warnings' in stats:
        for i, warning in enumerate(stats['quality_warnings']):
            diagnostics[f'data_quality_{i}'] = {
                'severity': 'MEDIUM',
                'message': warning
            }
    
    # Noise-to-signal ratio
    if 'residual_std' in stats and 'market_excess_std' in stats and stats['market_excess_std'] > 0:
        if stats['market_excess_std'] > 0 and beta != 0:
            noise_ratio = stats['residual_std'] / (beta * stats['market_excess_std'])
            if noise_ratio > 5:
                diagnostics['noise_ratio'] = {
                    'severity': 'HIGH',
                    'message': (
                        f'Very high idiosyncratic risk (noise ratio={noise_ratio:.1f}), '
                        'alpha estimates unreliable'
                    ),
                }
            elif noise_ratio > 3:
                diagnostics['noise_ratio'] = {
                    'severity': 'MEDIUM',
                    'message': (
                        f'Very high idiosyncratic risk (noise ratio={noise_ratio:.1f}), '
                        'alpha estimates unreliable'
                    ),
                }
    
    # Multicollinearity (for multi-factor models)
    if 'max_vif' in stats and stats['max_vif'] > 10:
        diagnostics['multicollinearity'] = {
            'severity': 'HIGH',
            'message': f'Severe multicollinearity (max VIF={stats["max_vif"]:.1f}), estimates unstable'
        }
    
    return diagnostics


def create_diagnostics_summary(all_diagnostics: List[Dict[str, dict]]) -> pd.DataFrame:
    """
    Create summary of all diagnostic issues across estimations.
    
    Args:
        all_diagnostics: List of diagnostic dictionaries from multiple estimations
    
    Returns:
        DataFrame summarizing diagnostic issues by type and severity
    """
    # Flatten all diagnostics
    issues = []
    for i, diag_dict in enumerate(all_diagnostics):
        for issue_type, issue_info in diag_dict.items():
            issues.append({
                'estimation': i,
                'issue_type': issue_type,
                'severity': issue_info['severity'],
                'message': issue_info['message']
            })
    
    if not issues:
        return pd.DataFrame(columns=['issue_type', 'severity', 'count', 'pct_affected'])
    
    issues_df = pd.DataFrame(issues)
    
    # Summarize by issue type and severity
    summary = issues_df.groupby(['issue_type', 'severity']).size().reset_index(name='count')
    summary['pct_affected'] = summary['count'] / len(all_diagnostics) * 100
    
    # Sort by severity and frequency
    severity_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'INFO': 0}
    summary['severity_rank'] = summary['severity'].map(severity_order)
    summary = summary.sort_values(['severity_rank', 'count'], ascending=[False, False])
    summary = summary.drop('severity_rank', axis=1)
    
    return summary


# === SECTION 7: FORECAST ERROR ANALYSIS ===

def decompose_forecast_error(actual: float, forecast_alpha: float, forecast_zero: float,
                           alpha: float, beta: float, mkt_excess: float, rf: float) -> dict:
    """
    Decompose forecast errors to understand sources.
    
    This helps understand why including alpha might not improve forecasts:
    1. Alpha estimation error (noise)
    2. Beta estimation error  
    3. Model misspecification
    4. Idiosyncratic shocks
    
    Returns:
        Dictionary with error decomposition
    """
    # Basic errors
    error_alpha = actual - forecast_alpha
    error_zero = actual - forecast_zero
    
    # Theoretical perfect forecast (if we knew true parameters)
    # We don't know true values, but we can analyze the structure
    
    # The difference in errors
    error_diff = error_alpha - error_zero
    
    # By construction: error_diff = -alpha (the estimated alpha)
    # This shows that including alpha helps only if the estimated alpha
    # has predictive power for the next period
    
    decomposition = {
        'error_with_alpha': error_alpha,
        'error_without_alpha': error_zero,
        'error_difference': error_diff,
        'estimated_alpha': alpha,
        'alpha_contribution': -alpha,  # How much alpha contributed to forecast
        
        # Diagnostics
        'alpha_helped': abs(error_alpha) < abs(error_zero),
        'improvement_pct': (abs(error_zero) - abs(error_alpha)) / abs(error_zero) * 100 
                          if error_zero != 0 else 0,
        
        # Sources of error (approximate)
        'market_surprise': mkt_excess - 0,  # Deviation from expected market return
        'idiosyncratic_shock': actual - (rf + beta * mkt_excess),  # What market model missed
    }
    
    return decomposition


# === SECTION 8: REPORTING UTILITIES ===

def format_model_output(alpha: float, beta: float, stats: dict, 
                       model_name: str = "CAPM") -> str:
    """
    Format model estimation results for reporting.
    
    Returns:
        Formatted string suitable for console output or reports
    """
    from config import format_percentage, format_number, get_significance_stars
    
    # Header
    output = f"\n{model_name} Estimation Results\n"
    output += "=" * 50 + "\n"
    
    # Main parameters
    alpha_pct = alpha * 100  # Convert to percentage
    output += f"Alpha: {alpha_pct:>8.4f}% {get_significance_stars(stats.get('p_alpha', 1))}\n"
    output += f"       ({format_number(stats.get('se_alpha', 0)*100, 'coefficients')})\n"
    output += f"       [t = {format_number(stats.get('t_alpha', 0), 'statistics')}]\n\n"
    
    output += f"Beta:  {beta:>8.4f} {get_significance_stars(stats.get('p_beta', 1))}\n"
    output += f"       ({format_number(stats.get('se_beta', 0), 'coefficients')})\n"
    output += f"       [t = {format_number(stats.get('t_beta', 0), 'statistics')}]\n\n"
    
    # Model fit
    output += f"R-squared:     {stats.get('r_squared', 0):>6.3f}\n"
    output += f"Adj R-squared: {stats.get('adj_r_squared', 0):>6.3f}\n"
    output += f"Observations:  {stats.get('n_obs', 0):>6d}\n"
    
    # Additional statistics if available
    if 'residual_std' in stats:
        output += f"\nResidual Std:  {stats['residual_std']*100:>6.3f}%\n"
    
    if 'durbin_watson' in stats:
        output += f"Durbin-Watson: {stats['durbin_watson']:>6.2f}\n"
    
    # Warnings
    if 'quality_warnings' in stats and stats['quality_warnings']:
        output += "\nWarnings:\n"
        for warning in stats['quality_warnings']:
            output += f"  - {warning}\n"
    
    output += "=" * 50 + "\n"
    
    return output


def create_model_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table comparing models with and without alpha.
    
    Args:
        results_df: DataFrame with estimation results
    
    Returns:
        Summary DataFrame suitable for paper tables
    """
    # Calculate summary statistics
    summary = {
        'Statistic': [
            'Mean Alpha (%)',
            'Std Alpha (%)',
            'Mean Beta',
            'Std Beta',
            'Mean R²',
            'Median R²',
            'Avg Observations',
            '% Significant Alpha (5%)',
            '% Negative Alpha'
        ],
        'Value': []
    }
    
    # Calculate values
    summary['Value'].append(f"{results_df['alpha'].mean() * 100:.4f}")
    summary['Value'].append(f"{results_df['alpha'].std() * 100:.4f}")
    summary['Value'].append(f"{results_df['beta'].mean():.3f}")
    summary['Value'].append(f"{results_df['beta'].std():.3f}")
    
    if 'r_squared' in results_df.columns:
        summary['Value'].append(f"{results_df['r_squared'].mean():.3f}")
        summary['Value'].append(f"{results_df['r_squared'].median():.3f}")
    else:
        summary['Value'].extend(['N/A', 'N/A'])
    
    if 'n_obs' in results_df.columns:
        summary['Value'].append(f"{results_df['n_obs'].mean():.0f}")
    else:
        summary['Value'].append('N/A')
    
    # Significance testing
    if 'p_alpha' in results_df.columns:
        pct_sig = (results_df['p_alpha'] < 0.05).mean() * 100
        summary['Value'].append(f"{pct_sig:.1f}")
    else:
        summary['Value'].append('N/A')
    
    pct_neg = (results_df['alpha'] < 0).mean() * 100
    summary['Value'].append(f"{pct_neg:.1f}")
    
    return pd.DataFrame(summary)
