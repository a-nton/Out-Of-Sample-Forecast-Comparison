"""
data_loader.py - Data loading and preprocessing functions.
Handles CRSP stock data and Fama-French factor data with comprehensive validation.
"""

import logging
from datetime import datetime
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd

from config import DATA_PATHS, DATA_FILTERS, MODEL_CONFIG, PRESENTATION_CONFIG
logger = logging.getLogger(__name__)

# === SECTION 1: DATA LOADING FUNCTIONS ===

def load_crsp_data(filepath: str = None) -> pd.DataFrame:
    """Load CRSP data from parquet file with validation."""
    if filepath is None:
        filepath = DATA_PATHS['crsp_file']

    logger.info("Loading CRSP data from %s", filepath)

    try:
        # Use fastparquet engine to avoid pyarrow extension issues in minimal setups
        df = pd.read_parquet(filepath, engine="fastparquet")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CRSP data file not found: {filepath}") from exc
    except Exception as exc:  # pragma: no cover - unexpected read errors
        raise Exception(f"Error loading CRSP data: {str(exc)}") from exc

    # Standardize column names (ensure uppercase)
    df.columns = df.columns.str.upper()

    # Required columns
    required_cols = ['PERMNO', 'DATE', 'SHRCD', 'EXCHCD', 'PRC', 'RET', 'SHROUT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required CRSP columns: {missing_cols}")

    # Standardize datatypes
    logger.debug("Standardizing data types")
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['PERMNO'] = df['PERMNO'].astype(int)

    # Numeric columns - handle CRSP-specific issues
    numeric_cols = ['PRC', 'RET', 'SHROUT', 'VWRETD', 'SPRTRN']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle CRSP-specific data issues
    # PRC < 0 indicates bid-ask average
    df['ABS_PRC'] = df['PRC'].abs()

    # RET = -66.0 to -99.0 are CRSP missing codes
    df.loc[df['RET'] <= -66.0, 'RET'] = np.nan

    # Winsorize returns before applying filters
    if MODEL_CONFIG.get('winsorize_returns', True):
        level = MODEL_CONFIG.get('winsorize_level', 0.005)

        # Store original returns for diagnostics
        df['RET_original'] = df['RET'].copy()

        ret_notna = df['RET'].notna()
        if ret_notna.sum() > 0:
            lower = df.loc[ret_notna, 'RET'].quantile(level)
            upper = df.loc[ret_notna, 'RET'].quantile(1 - level)

            n_lower = (df['RET'] < lower).sum()
            n_upper = (df['RET'] > upper).sum()

            df.loc[df['RET'] < lower, 'RET'] = lower
            df.loc[df['RET'] > upper, 'RET'] = upper

            logger.info(
                f"Winsorized {n_lower} returns below {lower:.4f} ({level*100}%ile)"
            )
            logger.info(
                f"Winsorized {n_upper} returns above {upper:.4f} ({(1 - level)*100}%ile)"
            )

            affected = n_lower + n_upper
            total = ret_notna.sum()
            logger.info(
                f"Total affected: {affected}/{total} ({affected/total*100:.2f}%)"
            )

    # Convert RET to percentage if needed (CRSP usually provides as decimal)
    # Check if returns are likely in decimal format (small values)
    ret_mean = df['RET'].dropna().mean()
    if abs(ret_mean) < 0.01:  # Likely in decimal format
        logger.debug("Returns appear to be in decimal format (as expected)")
    else:
        logger.warning("Returns may not be in decimal format. Mean return = %s", ret_mean)

    # Apply filters if specified
    initial_obs = len(df)
    initial_stocks = df['PERMNO'].nunique()

    # Date filter
    if 'start_date' in DATA_FILTERS and 'end_date' in DATA_FILTERS:
        start_date = pd.to_datetime(DATA_FILTERS['start_date'])
        end_date = pd.to_datetime(DATA_FILTERS['end_date'])
        df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]

    # Share code filter (common stocks)
    if 'shrcd_codes' in DATA_FILTERS:
        df = df[df['SHRCD'].isin(DATA_FILTERS['shrcd_codes'])]

    # Exchange filter
    if 'exchcd_codes' in DATA_FILTERS:
        df = df[df['EXCHCD'].isin(DATA_FILTERS['exchcd_codes'])]

    # Price filter (penny stocks)
    if 'min_price' in DATA_FILTERS:
        df = df[df['ABS_PRC'] >= DATA_FILTERS['min_price']]

    # Basic data quality filters
    df = df[df['SHROUT'] > 0]  # Must have shares outstanding
    df = df[df['RET'].notna()]  # Must have return data

    # Rename DATE to date for consistency
    df = df.rename(columns={'DATE': 'date'})

    logger.info(
        "Loaded %s observations for %s stocks", f"{len(df):,}", f"{df['PERMNO'].nunique():,}"
    )
    logger.debug("Date range: %s to %s", df['date'].min(), df['date'].max())
    logger.debug(
        "Filtered from %s observations and %s stocks", f"{initial_obs:,}", f"{initial_stocks:,}"
    )

    return df


def load_ff_factors(filepath: str = None, factor_model: str = 'ff3') -> pd.DataFrame:
    """Load Fama-French factor data with validation."""
    if filepath is None:
        if factor_model == 'ff3':
            filepath = DATA_PATHS['ff_factors']
        else:
            raise ValueError("FF5 factor data file must be provided via 'filepath'.")

    logger.info("Loading Fama-French %s factors", factor_model.upper())

    try:
        # Skip the 3-line preamble
        df = pd.read_csv(filepath, skiprows=3)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Fama-French data file not found: {filepath}") from exc
    except Exception as exc:  # pragma: no cover - unexpected read errors
        raise Exception(f"Error loading Fama-French data: {str(exc)}") from exc

    # Rename date column
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

    # Drop any bottom rows whose date isn't eight digits
    mask = df['date'].astype(str).str.match(r'^\d{8}$')
    df = df.loc[mask].copy()

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

    # Identify factor columns based on model
    if factor_model == 'ff3':
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
    else:
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

    # Check for missing columns
    missing_cols = [col for col in factor_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing Fama-French factors: {missing_cols}")

    # Convert to numeric
    for col in factor_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check if factors are in percentage or decimal format
    # Using a 1% threshold captures daily Fama-French data typically
    # provided in percentage units while avoiding unnecessary
    # conversion when data are already in decimal format.
    mkt_mean = df['Mkt-RF'].dropna().mean()
    if abs(mkt_mean) > 0.01:  # Likely in percentage format
        logger.debug("Factors appear to be in percentage format (mean market = %.3f)", mkt_mean)
        logger.debug("Converting to decimal format to match CRSP returns")
        for col in factor_cols:
            df[col] = df[col] / 100
    else:
        logger.debug("Factors appear to be in decimal format")

    # Keep only necessary columns
    df = df[['date'] + factor_cols]

    logger.info("Loaded factors from %s to %s", df['date'].min(), df['date'].max())

    return df


# === SECTION 2: DATA MERGING AND PREPARATION ===

def prepare_analysis_data(crsp_df: pd.DataFrame,
                         ff_df: pd.DataFrame,
                         apply_filters: bool = True) -> pd.DataFrame:
    """
    Merge CRSP and FF data and apply optional filters.

    The returned DataFrame includes a market return series ``MKT`` drawn
    from the CRSP value-weighted index so that CAPM estimations use the
    same market definition as value-weighted beta calculations.

    Returns:
        Merged DataFrame ready for analysis
    """
    logger.info("Merging CRSP and Fama-French data")

    # Merge on date (inner join to keep only dates with both data sources)
    df = crsp_df.merge(ff_df, on='date', how='inner')

    # Use CRSP value-weighted market return to ensure consistency with
    # later beta calculations.  This provides a market series built from
    # the same universe and weighting scheme used in ``calculate_vw_beta``.
    if 'VWRETD' in df.columns:
        df = df.rename(columns={'VWRETD': 'MKT'})
    else:  # pragma: no cover - unlikely fallback
        logger.warning('VWRETD missing, computing value-weighted market return')
        temp = crsp_df.copy()
        temp['abs_prc'] = temp['PRC'].abs()
        temp['market_cap'] = temp['abs_prc'] * temp['SHROUT'] / 1000
        mkt_series = (
            temp.groupby('date')
                .apply(lambda x: (x['RET'] * x['market_cap']).sum() / x['market_cap'].sum())
                .rename('MKT')
                .reset_index()
        )
        df = df.merge(mkt_series, on='date', how='left')
    
    # Add calculated fields
    df['abs_prc'] = df['PRC'].abs()
    df['market_cap'] = df['abs_prc'] * df['SHROUT'] / 1000  # In millions
    
    if apply_filters:
        initial_len = len(df)

        # Apply additional filters based on config
        if 'min_market_cap' in DATA_FILTERS and DATA_FILTERS['min_market_cap'] > 0:
            df = df[df['market_cap'] >= DATA_FILTERS['min_market_cap']]

        logger.debug(
            "Additional filters: %s → %s observations",
            f"{initial_len:,}",
            f"{len(df):,}",
        )
    
    # Sort by PERMNO and date for efficient processing
    df = df.sort_values(['PERMNO', 'date']).reset_index(drop=True)
    
    logger.info(
        "Final dataset: %s observations for %s stocks",
        f"{len(df):,}",
        f"{df['PERMNO'].nunique():,}",
    )
    
    return df


# === SECTION 3: DATA VALIDATION ===

def validate_merged_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Comprehensive validation of merged CRSP-FF data.
    Returns validation report and raises exceptions for critical issues.
    """
    print("\n" + "="*70)
    print("DATA VALIDATION REPORT")
    print("="*70)
    
    validation_report = {}
    
    # 1. Check required columns
    print("\n1. Column Validation:")
    required_cols = ['PERMNO', 'date', 'RET', 'RF', 'MKT', 'PRC', 'SHROUT', 'market_cap']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Data type report
    print(f"{'Column':<15} {'Type':<15} {'Non-null':<12} {'Missing %':<10}")
    print("-" * 52)
    for col in required_cols:
        non_null = df[col].notna().sum()
        missing_pct = (1 - non_null/len(df)) * 100
        print(f"{col:<15} {str(df[col].dtype):<15} {non_null:>10,}  {missing_pct:>8.1f}%")
    
    validation_report['missing_data'] = {col: (df[col].isna().sum()/len(df)*100) 
                                        for col in required_cols}
    
    # 2. Return statistics
    print("\n2. Return Statistics:")
    ret_stats = df['RET'].describe()
    print(f"  Mean return: {ret_stats['mean']*100:.4f}%")
    print(f"  Std dev:     {ret_stats['std']*100:.4f}%")
    print(f"  Min return:  {ret_stats['min']*100:.2f}%")
    print(f"  Max return:  {ret_stats['max']*100:.2f}%")
    
    # Check for extreme returns
    extreme_threshold = 0.5  # 50% daily return
    extreme_rets = df[df['RET'].abs() > extreme_threshold]
    if len(extreme_rets) > 0:
        print(f"\n  WARNING: {len(extreme_rets):,} extreme returns (>{extreme_threshold*100:.0f}% daily)")
        print(f"  Consider additional filtering or winsorization")
        validation_report['extreme_returns'] = len(extreme_rets)
    
    # 3. Market cap distribution
    print("\n3. Market Capitalization ($ millions):")
    cap_stats = df['market_cap'].describe()
    print(f"  Mean:    ${cap_stats['mean']:>10,.1f}")
    print(f"  Median:  ${cap_stats['50%']:>10,.1f}")
    print(f"  Min:     ${cap_stats['min']:>10,.1f}")
    print(f"  Max:     ${cap_stats['max']:>10,.1f}")
    
    # Size distribution
    print("\n  Size Distribution:")
    size_cutoffs = [10, 100, 1000, 10000]  # Million dollars
    for i, cutoff in enumerate(size_cutoffs):
        n_below = (df['market_cap'] < cutoff).sum()
        pct_below = n_below / len(df) * 100
        print(f"    < ${cutoff:>6,}M: {n_below:>8,} ({pct_below:>5.1f}%)")
    
    # 4. Factor correlations
    print("\n4. Factor Correlations:")
    factors = ['MKT', 'SMB', 'HML'] if 'SMB' in df.columns else ['MKT']
    if len(factors) > 1:
        corr_matrix = df[factors].corr()
        print(corr_matrix.round(3))
        
        # Check for high correlations
        high_corr_threshold = 0.7
        for i in range(len(factors)):
            for j in range(i+1, len(factors)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > high_corr_threshold:
                    print(f"\n  WARNING: High correlation between {factors[i]} and {factors[j]}: {corr:.3f}")
    
    # 5. Time series properties
    print("\n5. Time Series Properties:")
    date_range = df['date'].max() - df['date'].min()
    print(f"  Date range: {df['date'].min()} to {df['date'].max()} ({date_range.days:,} days)")
    
    # Check for gaps in dates
    all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    trading_dates = df['date'].unique()
    missing_dates = len(all_dates) - len(trading_dates)
    print(f"  Trading days: {len(trading_dates):,}")
    print(f"  Calendar days: {len(all_dates):,}")
    print(f"  Ratio: {len(trading_dates)/len(all_dates):.3f} (expected ~0.69)")
    
    # 6. Data quality summary
    print("\n6. Data Quality Summary:")
    quality_issues = []
    
    if validation_report.get('extreme_returns', 0) > 100:
        quality_issues.append("Many extreme returns detected")
    
    if any(pct > 10 for pct in validation_report['missing_data'].values()):
        quality_issues.append("High percentage of missing data")
    
    if df['market_cap'].min() < 1:
        quality_issues.append("Very small market cap stocks present")
    
    if quality_issues:
        print("  Issues found:")
        for issue in quality_issues:
            print(f"    - {issue}")
    else:
        print("  ✓ No major data quality issues detected")
    
    print("\n" + "="*70)
    
    validation_report['quality_issues'] = quality_issues
    validation_report['n_observations'] = len(df)
    validation_report['n_stocks'] = df['PERMNO'].nunique()
    
    return validation_report


# === SECTION 4: ADDITIONAL UTILITY FUNCTIONS ===

def get_trading_days_between(start_date: pd.Timestamp, end_date: pd.Timestamp, 
                           trading_dates: pd.Series) -> int:
    """
    Count actual trading days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date  
        trading_dates: Series of all trading dates in the dataset
    
    Returns:
        Number of trading days
    """
    mask = (trading_dates >= start_date) & (trading_dates <= end_date)
    return mask.sum()


def check_data_availability(df: pd.DataFrame, min_history: int = 252) -> pd.DataFrame:
    """
    Create a summary of data availability by stock.
    
    Returns:
        DataFrame with PERMNO and various availability metrics
    """
    availability = df.groupby('PERMNO').agg({
        'date': ['min', 'max', 'count'],
        'RET': lambda x: x.notna().sum(),
        'market_cap': 'mean'
    })
    
    availability.columns = ['start_date', 'end_date', 'n_obs', 'n_returns', 'avg_market_cap']
    availability['pct_missing'] = 1 - (availability['n_returns'] / availability['n_obs'])
    availability['has_min_history'] = availability['n_obs'] >= min_history
    
    return availability.reset_index()


def create_data_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table suitable for paper appendix.
    """
    summary = {
        'Statistic': [],
        'Value': []
    }
    
    # Basic counts
    summary['Statistic'].extend([
        'Total observations',
        'Unique stocks (PERMNOs)', 
        'Start date',
        'End date',
        'Trading days'
    ])
    
    summary['Value'].extend([
        f"{len(df):,}",
        f"{df['PERMNO'].nunique():,}",
        str(df['date'].min().date()),
        str(df['date'].max().date()),
        f"{df['date'].nunique():,}"
    ])
    
    # Return statistics
    ret_stats = df['RET'].dropna()
    summary['Statistic'].extend([
        'Mean return (daily)',
        'Std dev return (daily)',
        'Skewness',
        'Kurtosis'
    ])
    
    from scipy import stats
    summary['Value'].extend([
        f"{ret_stats.mean()*100:.4f}%",
        f"{ret_stats.std()*100:.4f}%",
        f"{stats.skew(ret_stats):.3f}",
        f"{stats.kurtosis(ret_stats):.3f}"
    ])
    
    # Market cap statistics
    cap_stats = df['market_cap'].dropna()
    summary['Statistic'].extend([
        'Mean market cap ($M)',
        'Median market cap ($M)',
        '10th percentile ($M)',
        '90th percentile ($M)'
    ])
    
    summary['Value'].extend([
        f"${cap_stats.mean():,.1f}",
        f"${cap_stats.median():,.1f}",
        f"${cap_stats.quantile(0.1):,.1f}",
        f"${cap_stats.quantile(0.9):,.1f}"
    ])

    return pd.DataFrame(summary)


def analyze_winsorization_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the impact of winsorization on return distribution
    """
    if 'RET_original' not in df.columns:
        print("No winsorization applied (RET_original not found)")
        return None

    analysis = pd.DataFrame({
        'Statistic': ['Mean', 'Std Dev', 'Skewness', 'Kurtosis',
                     'Min', '1%', '5%', '95%', '99%', 'Max',
                     'N Changed', '% Changed'],
        'Original': [
            df['RET_original'].mean() * 100,
            df['RET_original'].std() * 100,
            df['RET_original'].skew(),
            df['RET_original'].kurtosis(),
            df['RET_original'].min() * 100,
            df['RET_original'].quantile(0.01) * 100,
            df['RET_original'].quantile(0.05) * 100,
            df['RET_original'].quantile(0.95) * 100,
            df['RET_original'].quantile(0.99) * 100,
            df['RET_original'].max() * 100,
            0,
            0,
        ],
        'Winsorized': [
            df['RET'].mean() * 100,
            df['RET'].std() * 100,
            df['RET'].skew(),
            df['RET'].kurtosis(),
            df['RET'].min() * 100,
            df['RET'].quantile(0.01) * 100,
            df['RET'].quantile(0.05) * 100,
            df['RET'].quantile(0.95) * 100,
            df['RET'].quantile(0.99) * 100,
            df['RET'].max() * 100,
            (df['RET'] != df['RET_original']).sum(),
            (df['RET'] != df['RET_original']).mean() * 100,
        ]
    })

    numeric_rows = analysis['Statistic'].isin(['Mean', 'Std Dev', 'Skewness', 'Kurtosis',
                                               'Min', '1%', '5%', '95%', '99%', 'Max'])
    analysis.loc[numeric_rows, 'Change'] = (
        analysis.loc[numeric_rows, 'Winsorized'] -
        analysis.loc[numeric_rows, 'Original']
    )

    return analysis


def test_winsorization_levels(df: pd.DataFrame,
                              levels: list = [0.001, 0.0025, 0.005, 0.01]) -> pd.DataFrame:
    """
    Test different winsorization levels to find optimal
    """
    results = []

    for level in levels:
        df_temp = df.copy()

        lower = df_temp['RET'].quantile(level)
        upper = df_temp['RET'].quantile(1 - level)

        df_temp['RET_wins'] = df_temp['RET'].clip(lower=lower, upper=upper)

        n_affected = (df_temp['RET'] != df_temp['RET_wins']).sum()
        pct_affected = n_affected / len(df_temp) * 100

        results.append({
            'Level': f"{level*100:.2f}%",
            'Lower Bound': f"{lower*100:.3f}%",
            'Upper Bound': f"{upper*100:.3f}%",
            'N Affected': n_affected,
            '% Affected': f"{pct_affected:.2f}%",
            'New Std Dev': f"{df_temp['RET_wins'].std()*100:.3f}%",
            'Removed Kurtosis': f"{df_temp['RET'].kurtosis() - df_temp['RET_wins'].kurtosis():.2f}"
        })

    return pd.DataFrame(results)
