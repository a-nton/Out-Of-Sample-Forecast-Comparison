"""
sampling.py - Random sampling and window management for event studies.
Ensures non-overlapping estimation windows and proper data quality.
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
import warnings

from config import SAMPLING_CONFIG, DATA_FILTERS, MODEL_CONFIG

# === SECTION 1: WINDOW MANAGEMENT ===

class WindowTracker:
    """
    Tracks used estimation windows to prevent overlap.
    Ensures each stock-date combination is used at most once.
    """
    
    def __init__(self):
        self.used_windows: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
        self.used_forecast_dates: Dict[int, Set[pd.Timestamp]] = {}
        self.rejection_reasons: Dict[str, int] = {}
        
    def check_overlap(self, permno: int, start: pd.Timestamp, 
                     end: pd.Timestamp, forecast_date: pd.Timestamp) -> Tuple[bool, str]:
        """
        Check if proposed window overlaps with any previously used.
        
        Returns:
            (has_overlap, reason)
        """
        # Check estimation window overlap
        if permno in self.used_windows:
            for used_start, used_end in self.used_windows[permno]:
                if start <= used_end and end >= used_start:
                    return True, "estimation_window_overlap"
        
        # Check if forecast date already used for this stock
        if permno in self.used_forecast_dates:
            if forecast_date in self.used_forecast_dates[permno]:
                return True, "forecast_date_reused"
        
        return False, ""
    
    def add_window(self, permno: int, start: pd.Timestamp, 
                  end: pd.Timestamp, forecast_date: pd.Timestamp):
        """Record a used window."""
        if permno not in self.used_windows:
            self.used_windows[permno] = []
            self.used_forecast_dates[permno] = set()
            
        self.used_windows[permno].append((start, end))
        self.used_forecast_dates[permno].add(forecast_date)
    
    def record_rejection(self, reason: str):
        """Track why samples are rejected."""
        if reason not in self.rejection_reasons:
            self.rejection_reasons[reason] = 0
        self.rejection_reasons[reason] += 1
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of window usage and rejections."""
        total_windows = sum(len(windows) for windows in self.used_windows.values())
        
        return {
            'n_stocks_used': len(self.used_windows),
            'total_windows': total_windows,
            'avg_windows_per_stock': total_windows / len(self.used_windows) if self.used_windows else 0,
            'rejection_reasons': dict(self.rejection_reasons),
            'total_rejections': sum(self.rejection_reasons.values())
        }


# === SECTION 2: SAMPLING UTILITIES ===

def get_valid_estimation_indices(stock_data: pd.DataFrame, 
                               estimation_window: int,
                               forecast_horizons: List[int]) -> np.ndarray:
    """
    Get indices where we can form valid estimation windows and forecasts.
    
    Args:
        stock_data: DataFrame for a single stock, sorted by date
        estimation_window: Number of trading days for estimation
        forecast_horizons: List of forecast horizons to accommodate
    
    Returns:
        Array of valid indices for estimation end dates
    """
    n_obs = len(stock_data)
    max_horizon = max(forecast_horizons)
    
    # Need estimation_window days before and max_horizon days after
    valid_start = estimation_window - 1  # -1 because we include the end date
    valid_end = n_obs - max_horizon
    
    if valid_end <= valid_start:
        return np.array([])
    
    return np.arange(valid_start, valid_end)


def validate_estimation_window(data: pd.DataFrame, config: dict) -> Tuple[bool, str]:
    """
    Validate that estimation window meets all requirements.
    
    Args:
        data: Estimation window data
        config: Configuration dictionary
    
    Returns:
        (is_valid, rejection_reason)
    """
    # Check minimum observations
    if len(data) < config.get('min_observations', 150):
        return False, "insufficient_observations"
    
    # Check for enough non-missing returns
    non_missing_returns = data['RET'].notna().sum()
    if non_missing_returns < config.get('min_observations', 150):
        return False, "insufficient_returns"
    
    # Check date continuity (approximate check for gaps)
    date_diff = data['date'].diff()
    max_gap = date_diff.max()
    if max_gap > pd.Timedelta(days=10):  # More than 10 calendar days gap
        return False, "date_discontinuity"
    
    # Check for penny stocks
    if 'min_price' in config and config['min_price'] > 0:
        if (data['abs_prc'] < config['min_price']).any():
            return False, "penny_stock"
    
    # Check for extreme returns
    if 'max_return' in config:
        if (data['RET'].abs() > config['max_return']).any():
            return False, "extreme_returns"
    
    # Check for too many consecutive missing returns
    if data['RET'].isna().any():
        # Find maximum consecutive NaN values
        is_nan = data['RET'].isna()
        nan_groups = (is_nan != is_nan.shift()).cumsum()
        nan_streaks = is_nan.groupby(nan_groups).sum()
        max_consecutive_missing = nan_streaks.max() if len(nan_streaks) > 0 else 0
        
        if max_consecutive_missing > 5:
            return False, "consecutive_missing"
    
    # Check for sufficient price variation (not stuck/halted)
    price_variation = data['abs_prc'].std() / data['abs_prc'].mean()
    if price_variation < 0.01:  # Less than 1% coefficient of variation
        return False, "insufficient_price_variation"
    
    # Check market cap if required
    if 'min_market_cap' in config and config['min_market_cap'] > 0:
        if 'market_cap' in data.columns:
            if (data['market_cap'] < config['min_market_cap']).any():
                return False, "low_market_cap"
    
    # Additional data quality checks if enabled
    if MODEL_CONFIG.get('validate_estimations', True):
        # Check for suspicious return patterns
        returns = data['RET'].dropna()
        if len(returns) > 10:
            # Check for too many zero returns
            zero_return_pct = (returns == 0).sum() / len(returns)
            if zero_return_pct > 0.5:
                return False, "excessive_zero_returns"
    
    return True, ""


def calculate_actual_trading_days(start_date: pd.Timestamp,
                                end_date: pd.Timestamp,
                                all_dates: pd.Series) -> int:
    """
    Calculate actual number of trading days between dates.
    
    Args:
        start_date: Start date
        end_date: End date (inclusive)
        all_dates: Series of all dates in the dataset
    
    Returns:
        Number of trading days
    """
    mask = (all_dates >= start_date) & (all_dates <= end_date)
    return mask.sum()


def compute_cross_sectional_weights(merged_data: pd.DataFrame,
                                    min_market_cap: float = 0) -> pd.DataFrame:
    """Compute market-cap weights for each stock-date pair from the
    broad market universe.

    Args:
        merged_data: DataFrame containing at least PERMNO, date and
            market_cap columns.
        min_market_cap: Minimum market cap (same units as market_cap) to
            include in the universe. Defaults to 0 meaning all stocks.

    Returns:
        DataFrame with columns PERMNO, date and weight where weight is
        the firm's share of total market cap on that date.
    """

    weights = merged_data[['PERMNO', 'date', 'market_cap']].dropna().copy()

    if min_market_cap > 0:
        weights = weights[weights['market_cap'] >= min_market_cap]

    weights['total_cap'] = weights.groupby('date')['market_cap'].transform('sum')
    weights['weight'] = weights['market_cap'] / weights['total_cap']
    return weights[['PERMNO', 'date', 'weight']]


def validate_weight_distribution(weights: pd.DataFrame,
                                 verbose: bool = True) -> None:
    """Validate that cross-sectional weights form a proper market
    distribution.

    Checks that weights sum to 1 for each date and reports concentration
    metrics. Raises a warning if any date's weights do not sum to 1.

    Args:
        weights: DataFrame produced by ``compute_cross_sectional_weights``
        verbose: If True, print summary statistics.
    """

    totals = weights.groupby('date')['weight'].sum()
    mismatched = totals[np.abs(totals - 1) > 1e-6]
    if not mismatched.empty:
        warnings.warn(
            f"Weight sums differ from 1 for {len(mismatched)} dates;"
            " check market universe coverage."
        )

    if verbose:
        # Average weights across time to avoid double counting stocks
        # with longer histories. This yields a time-normalized distribution.
        by_stock = weights.groupby('PERMNO')['weight'].mean()
        top10 = by_stock.nlargest(min(10, len(by_stock))).sum()
        top50 = by_stock.nlargest(min(50, len(by_stock))).sum()
        print(
            "\nWeight distribution summary:"
            f" {len(by_stock)} stocks, top 10 {top10:.1%}, top 50 {top50:.1%}"
        )


# === SECTION 3: MAIN SAMPLING FUNCTION ===

def sample_events(merged_data: pd.DataFrame,
                 n_samples: int,
                 estimation_window: int,
                 forecast_horizons: List[int],
                 config: dict,
                 random_seed: Optional[int] = None,
                 verbose: bool = True) -> Dict[int, List[dict]]:
    """
    Main sampling function for event study.
    Samples random stock-date pairs ensuring all requirements are met.
    
    Args:
        merged_data: Merged CRSP and FF data
        n_samples: Number of samples to collect
        estimation_window: Number of trading days for estimation
        forecast_horizons: List of forecast horizons
        config: Configuration dictionary
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress
    
    Returns:
        Dictionary mapping horizon to list of samples
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Initialize
    window_tracker = WindowTracker()
    samples = {h: [] for h in forecast_horizons}
    all_trading_dates = merged_data['date'].unique()

    # --- Pre-group data by PERMNO to avoid repeated slicing ---
    grouped_data = {}
    valid_permnos = []
    min_required_obs = estimation_window + max(forecast_horizons) + 10

    for permno, stock_df in merged_data.groupby('PERMNO'):
        stock_df = stock_df.sort_values('date').reset_index(drop=True)
        if len(stock_df) < min_required_obs:
            continue

        valid_indices = get_valid_estimation_indices(stock_df,
                                                    estimation_window,
                                                    forecast_horizons)
        if len(valid_indices) == 0:
            continue

        grouped_data[permno] = {
            'data': stock_df,
            'valid_indices': valid_indices
        }
        valid_permnos.append(permno)

    if verbose:
        print(f"\nSampling from {len(valid_permnos)} stocks with sufficient data")
        print(f"Target: {n_samples} samples for {len(forecast_horizons)} horizons")
    
    attempts = 0
    max_attempts = n_samples * config.get('max_attempts_multiplier', 100)
    last_progress = 0
    
    while len(samples[forecast_horizons[0]]) < n_samples and attempts < max_attempts:
        attempts += 1

        # Random stock selection
        permno = random.choice(valid_permnos)
        stock_info = grouped_data[permno]
        stock_data = stock_info['data']
        valid_indices = stock_info['valid_indices']

        if len(valid_indices) == 0:
            window_tracker.record_rejection("no_valid_indices")
            continue

        # Random date selection
        est_end_idx = random.choice(valid_indices)
        est_start_idx = est_end_idx - estimation_window + 1
        
        # Extract estimation window
        est_data = stock_data.iloc[est_start_idx:est_end_idx + 1].copy()
        
        # Validate estimation window
        is_valid, rejection_reason = validate_estimation_window(est_data, config)
        if not is_valid:
            window_tracker.record_rejection(rejection_reason)
            continue
        
        # Check all forecast horizons
        valid_for_all_horizons = True
        forecast_data_by_horizon = {}
        
        for horizon in forecast_horizons:
            forecast_idx = est_end_idx + horizon
            
            if forecast_idx >= len(stock_data):
                valid_for_all_horizons = False
                window_tracker.record_rejection(f"no_data_h{horizon}")
                break
            
            forecast_data = stock_data.iloc[forecast_idx]
            
            # Validate forecast day
            if pd.isna(forecast_data['RET']) or pd.isna(forecast_data['MKT']):
                valid_for_all_horizons = False
                window_tracker.record_rejection(f"missing_forecast_h{horizon}")
                break
            
            forecast_data_by_horizon[horizon] = forecast_data
        
        if not valid_for_all_horizons:
            continue
        
        # Check overlap for base forecast date
        base_forecast_date = stock_data.iloc[est_end_idx + 1]['date']
        has_overlap, overlap_reason = window_tracker.check_overlap(
            permno, est_data.iloc[0]['date'], est_data.iloc[-1]['date'], base_forecast_date
        )
        
        if has_overlap:
            window_tracker.record_rejection(overlap_reason)
            continue
        
        # All checks passed - create samples for each horizon
        base_sample = {
            'permno': permno,
            'estimation_start': est_data.iloc[0]['date'],
            'estimation_end': est_data.iloc[-1]['date'],
            'estimation_data': est_data,
            'n_obs': len(est_data),
            'n_returns': est_data['RET'].notna().sum(),
            'mean_market_cap': est_data['market_cap'].mean() if 'market_cap' in est_data.columns else None,
        }
        
        # Add sample for each horizon
        for horizon in forecast_horizons:
            sample = base_sample.copy()
            sample['horizon'] = horizon
            sample['forecast_date'] = forecast_data_by_horizon[horizon]['date']
            sample['forecast_data'] = forecast_data_by_horizon[horizon]
            sample['calendar_days'] = (sample['forecast_date'] - sample['estimation_end']).days
            sample['trading_days'] = calculate_actual_trading_days(
                sample['estimation_end'] + timedelta(days=1),
                sample['forecast_date'],
                all_trading_dates
            )
            
            samples[horizon].append(sample)
        
        # Record window usage
        window_tracker.add_window(permno, est_data.iloc[0]['date'], 
                                 est_data.iloc[-1]['date'], base_forecast_date)
        
        # Progress update
        current_progress = len(samples[forecast_horizons[0]])
        if verbose and current_progress % 10 == 0 and current_progress > last_progress:
            print(f"Progress: {current_progress}/{n_samples} samples collected "
                  f"({attempts} attempts, {attempts/current_progress:.1f} attempts per sample)")
            last_progress = current_progress
    
    # Final summary
    if verbose:
        print(f"\nSampling complete: {len(samples[forecast_horizons[0]])} samples collected in {attempts} attempts")
        
        # Print rejection summary
        summary = window_tracker.get_summary()
        if summary['rejection_reasons']:
            print("\nRejection reasons:")
            total_rejections = summary['total_rejections']
            for reason, count in sorted(summary['rejection_reasons'].items(), 
                                      key=lambda x: x[1], reverse=True):
                print(f"  {reason:30} {count:6d} ({count/total_rejections*100:5.1f}%)")
        
        print(f"\nStocks used: {summary['n_stocks_used']}")
        print(f"Average samples per stock: {summary['avg_windows_per_stock']:.2f}")

    return samples


def sample_events_value_weighted(merged_data: pd.DataFrame,
                                n_samples: int,
                                estimation_window: int,
                                forecast_horizons: List[int],
                                config: dict,
                                random_seed: Optional[int] = None,
                                verbose: bool = True) -> Dict[int, List[dict]]:
    """Sample events using market-cap weights from the full market."""

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # --- Compute and validate cross-sectional weights ---
    min_cap = config.get('min_market_cap', 0)
    weight_table = compute_cross_sectional_weights(merged_data, min_market_cap=min_cap)
    validate_weight_distribution(weight_table, verbose=verbose)

    # Average weight by stock for sampling
    stock_weights = weight_table.groupby('PERMNO')['weight'].mean()
    stock_weights = stock_weights / stock_weights.sum()
    valid_permnos = stock_weights.index.tolist()
    sampling_weights = stock_weights.values

    # Pre-compute lookup for individual date weights
    weight_lookup = weight_table.set_index(['PERMNO', 'date'])['weight'].to_dict()

    if verbose:
        top10 = stock_weights.nlargest(min(10, len(stock_weights))).sum()
        top50 = stock_weights.nlargest(min(50, len(stock_weights))).sum()
        print(f"\nValue-weighted sampling from {len(valid_permnos)} stocks")
        print(f"Top 10 stocks weight: {top10:.1%}")
        print(f"Top 50 stocks weight: {top50:.1%}")

    window_tracker = WindowTracker()
    samples = {h: [] for h in forecast_horizons}
    attempts = 0
    max_attempts = n_samples * config.get('max_attempts_multiplier', 100)
    all_trading_dates = merged_data['date'].unique()

    while len(samples[forecast_horizons[0]]) < n_samples and attempts < max_attempts:
        attempts += 1

        # Weighted selection by market cap
        permno = np.random.choice(valid_permnos, p=sampling_weights)
        stock_data = merged_data[merged_data['PERMNO'] == permno].sort_values('date').reset_index(drop=True)

        valid_indices = get_valid_estimation_indices(stock_data, estimation_window, forecast_horizons)
        if len(valid_indices) == 0:
            window_tracker.record_rejection("no_valid_indices")
            continue

        # Random date selection
        est_end_idx = random.choice(valid_indices)
        est_start_idx = est_end_idx - estimation_window + 1
        est_data = stock_data.iloc[est_start_idx:est_end_idx + 1].copy()

        is_valid, rejection_reason = validate_estimation_window(est_data, config)
        if not is_valid:
            window_tracker.record_rejection(rejection_reason)
            continue

        valid_for_all_horizons = True
        forecast_data_by_horizon = {}
        for horizon in forecast_horizons:
            forecast_idx = est_end_idx + horizon
            if forecast_idx >= len(stock_data):
                valid_for_all_horizons = False
                window_tracker.record_rejection(f"no_data_h{horizon}")
                break
            forecast_data = stock_data.iloc[forecast_idx]
            if pd.isna(forecast_data['RET']) or pd.isna(forecast_data['MKT']):
                valid_for_all_horizons = False
                window_tracker.record_rejection(f"missing_forecast_h{horizon}")
                break
            forecast_data_by_horizon[horizon] = forecast_data

        if not valid_for_all_horizons:
            continue

        base_forecast_date = stock_data.iloc[est_end_idx + 1]['date']
        has_overlap, overlap_reason = window_tracker.check_overlap(
            permno, est_data.iloc[0]['date'], est_data.iloc[-1]['date'], base_forecast_date
        )
        if has_overlap:
            window_tracker.record_rejection(overlap_reason)
            continue

        base_sample = {
            'permno': permno,
            'estimation_start': est_data.iloc[0]['date'],
            'estimation_end': est_data.iloc[-1]['date'],
            'estimation_data': est_data,
            'n_obs': len(est_data),
            'n_returns': est_data['RET'].notna().sum(),
            'mean_market_cap': est_data['market_cap'].mean() if 'market_cap' in est_data.columns else None,
        }

        for horizon in forecast_horizons:
            sample = base_sample.copy()
            sample['horizon'] = horizon
            sample['forecast_date'] = forecast_data_by_horizon[horizon]['date']
            sample['forecast_data'] = forecast_data_by_horizon[horizon]
            sample['calendar_days'] = (sample['forecast_date'] - sample['estimation_end']).days
            sample['trading_days'] = calculate_actual_trading_days(
                sample['estimation_end'] + timedelta(days=1),
                sample['forecast_date'],
                all_trading_dates,
            )
            sample['market_weight'] = weight_lookup.get(
                (permno, sample['forecast_date']), np.nan
            )
            samples[horizon].append(sample)

        window_tracker.add_window(permno, est_data.iloc[0]['date'], est_data.iloc[-1]['date'], base_forecast_date)

        current_progress = len(samples[forecast_horizons[0]])
        if verbose and current_progress % 10 == 0 and current_progress > 0:
            print(
                f"Progress: {current_progress}/{n_samples} samples collected "
                f"({attempts} attempts, {attempts/current_progress:.1f} attempts per sample)"
            )

    if verbose:
        print(f"\nSampling complete: {len(samples[forecast_horizons[0]])} samples collected in {attempts} attempts")
        summary = window_tracker.get_summary()
        if summary['rejection_reasons']:
            print("\nRejection reasons:")
            total_rejections = summary['total_rejections']
            for reason, count in sorted(summary['rejection_reasons'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason:30} {count:6d} ({count/total_rejections*100:5.1f}%)")
        print(f"\nStocks used: {summary['n_stocks_used']}")
        print(f"Average samples per stock: {summary['avg_windows_per_stock']:.2f}")

    return samples


# === SECTION 4: SAMPLE ANALYSIS AND VALIDATION ===

def analyze_sample_characteristics(samples: Dict[int, List[dict]]) -> pd.DataFrame:
    """
    Analyze characteristics of the collected samples.
    
    Args:
        samples: Dictionary mapping horizon to list of samples
    
    Returns:
        DataFrame with sample characteristics summary
    """
    all_characteristics = []
    
    for horizon, horizon_samples in samples.items():
        if not horizon_samples:
            continue
            
        chars = {
            'horizon': horizon,
            'n_samples': len(horizon_samples),
            'n_unique_stocks': len(set(s['permno'] for s in horizon_samples)),
            'avg_obs_per_window': np.mean([s['n_obs'] for s in horizon_samples]),
            'avg_returns_per_window': np.mean([s['n_returns'] for s in horizon_samples]),
            'pct_missing_returns': np.mean([
                (s['n_obs'] - s['n_returns']) / s['n_obs'] * 100 
                for s in horizon_samples
            ]),
        }
        
        # Market cap statistics
        market_caps = [s['mean_market_cap'] for s in horizon_samples if s['mean_market_cap'] is not None]
        if market_caps:
            chars['avg_market_cap'] = np.mean(market_caps)
            chars['median_market_cap'] = np.median(market_caps)
            chars['p10_market_cap'] = np.percentile(market_caps, 10)
            chars['p90_market_cap'] = np.percentile(market_caps, 90)
        
        # Date coverage
        all_dates = []
        for s in horizon_samples:
            all_dates.extend([s['estimation_start'], s['estimation_end'], s['forecast_date']])
        all_dates = pd.to_datetime(all_dates)
        
        chars['earliest_date'] = all_dates.min()
        chars['latest_date'] = all_dates.max()
        chars['date_span_years'] = (all_dates.max() - all_dates.min()).days / 365.25
        
        # Forecast horizon validation
        trading_days = [s['trading_days'] for s in horizon_samples]
        chars['mean_trading_days'] = np.mean(trading_days)
        chars['std_trading_days'] = np.std(trading_days)
        
        all_characteristics.append(chars)
    
    return pd.DataFrame(all_characteristics)


def validate_sampling_randomness(samples: Dict[int, List[dict]]) -> Dict[str, any]:
    """
    Validate that sampling is properly random and unbiased.
    
    Checks for:
    - Time clustering
    - Stock concentration
    - Day-of-week effects
    - Seasonal patterns
    
    Returns:
        Dictionary with validation results
    """
    # Use first horizon for validation
    first_horizon = min(samples.keys())
    sample_list = samples[first_horizon]
    
    if not sample_list:
        return {'error': 'No samples to validate'}
    
    # Extract dates and stocks
    est_end_dates = pd.Series([s['estimation_end'] for s in sample_list])
    permnos = pd.Series([s['permno'] for s in sample_list])
    
    validation_results = {}
    
    # 1. Time clustering test (should be uniform across time)
    est_end_dates_numeric = est_end_dates.astype(np.int64) // 10**9  # Convert to seconds
    time_range = est_end_dates_numeric.max() - est_end_dates_numeric.min()
    
    if time_range > 0:
        # Kolmogorov-Smirnov test for uniform distribution
        from scipy import stats
        normalized_times = (est_end_dates_numeric - est_end_dates_numeric.min()) / time_range
        ks_stat, ks_pval = stats.kstest(normalized_times, 'uniform')
        
        validation_results['time_uniformity'] = {
            'ks_statistic': ks_stat,
            'p_value': ks_pval,
            'is_uniform': ks_pval > 0.05,
            'interpretation': 'Good - dates appear uniformly distributed' if ks_pval > 0.05 
                            else 'Warning - dates may be clustered'
        }
    
    # 2. Stock concentration (should not oversample any stocks)
    stock_counts = permnos.value_counts()
    expected_count = len(sample_list) / len(permnos.unique())
    
    # Chi-square test
    observed = stock_counts.values
    expected = np.full_like(observed, expected_count, dtype=float)
    if len(observed) > 1:
        chi2_stat, chi2_pval = stats.chisquare(observed, expected)
        
        validation_results['stock_balance'] = {
            'max_count': stock_counts.max(),
            'min_count': stock_counts.min(),
            'expected_count': expected_count,
            'chi2_statistic': chi2_stat,
            'p_value': chi2_pval,
            'is_balanced': chi2_pval > 0.05,
            'most_sampled_stock': stock_counts.index[0]
        }
    
    # 3. Day of week effects
    dow_counts = est_end_dates.dt.dayofweek.value_counts().sort_index()
    validation_results['day_of_week'] = {
        'distribution': dow_counts.to_dict(),
        'Monday': dow_counts.get(0, 0),
        'Friday': dow_counts.get(4, 0),
        'weekend_samples': dow_counts.get(5, 0) + dow_counts.get(6, 0)  # Should be 0
    }
    
    # 4. Seasonal patterns (by month)
    month_counts = est_end_dates.dt.month.value_counts().sort_index()
    validation_results['monthly_distribution'] = {
        'counts': month_counts.to_dict(),
        'min_month': month_counts.min(),
        'max_month': month_counts.max(),
        'ratio': month_counts.max() / month_counts.min() if month_counts.min() > 0 else np.inf
    }
    
    # 5. Year distribution
    year_counts = est_end_dates.dt.year.value_counts().sort_index()
    validation_results['yearly_distribution'] = {
        'earliest_year': year_counts.index.min(),
        'latest_year': year_counts.index.max(),
        'n_years': len(year_counts),
        'most_sampled_year': year_counts.index[0]
    }
    
    return validation_results


def create_sample_summary_table(samples: Dict[int, List[dict]], 
                              output_format: str = 'dataframe') -> pd.DataFrame:
    """
    Create a summary table of sampling results suitable for paper appendix.
    
    Args:
        samples: Dictionary mapping horizon to samples
        output_format: 'dataframe' or 'latex'
    
    Returns:
        Summary DataFrame or LaTeX string
    """
    summary_data = []
    
    for horizon in sorted(samples.keys()):
        horizon_samples = samples[horizon]
        
        if not horizon_samples:
            continue
        
        # Calculate statistics
        row = {
            'Forecast Horizon': f"{horizon} day{'s' if horizon > 1 else ''}",
            'N': len(horizon_samples),
            'Unique Stocks': len(set(s['permno'] for s in horizon_samples)),
            'Avg Obs/Window': f"{np.mean([s['n_obs'] for s in horizon_samples]):.1f}",
            'Avg Trading Days': f"{np.mean([s['trading_days'] for s in horizon_samples]):.1f}",
        }
        
        # Market cap info
        market_caps = [s['mean_market_cap'] for s in horizon_samples if s['mean_market_cap'] is not None]
        if market_caps:
            row['Avg Market Cap ($M)'] = f"{np.mean(market_caps):,.0f}"
            row['Median Market Cap ($M)'] = f"{np.median(market_caps):,.0f}"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    if output_format == 'latex':
        return summary_df.to_latex(
            index=False,
            caption="Sample Characteristics by Forecast Horizon",
            label="tab:sample_characteristics",
            column_format='l' + 'r' * (len(summary_df.columns) - 1)
        )
    
    return summary_df
