"""Simplified configuration and formatting utilities for the analysis."""

import os
import numpy as np

# === CORE PARAMETERS ===
SAMPLING_CONFIG = {
    'n_samples': 100,                    # Number of random draws
    'estimation_window': 252,            # Trading days (1 year)
    'min_observations': 200,             # Minimum non-missing returns
    'forecast_horizons': [1, 5, 10, 20], # Trading days
    'random_seed': 42,                   # For reproducibility
}

# === DATA FILTERS ===
DATA_FILTERS = {
    'min_price': 1.0,                    # Penny stock filter
    'min_market_cap': 100,               # Minimum $100M market cap
    'start_date': '1970-01-02',
    'end_date': '2024-12-31',
    # Fixed filters (not configurable)
    'shrcd_codes': [10, 11],             # Common shares only
    'exchcd_codes': [1, 2, 3],           # NYSE, AMEX, NASDAQ
}

# === MODEL SETTINGS ===
# Always compare CAPM with FF3; no need to specify a base model
MODEL_CONFIG = {
    'winsorize_level': 0.005,            # 0.5% in each tail
}

# === ANALYSIS FLAGS ===
ANALYSIS_CONFIG = {
    'bootstrap_iterations': 1000,        # For confidence intervals
    'alpha_percentile_cutoff': 50,       # For subset analysis
}

# === OUTPUT SETTINGS ===
OUTPUT_CONFIG = {
    'results_dir': './results/',
    'figure_format': 'pdf',
    'figure_dpi': 300,
    'table_format': 'latex',
    'verbose': True,
}

# === PRESENTATION SETTINGS ===
# Store captions and notes by table/figure name for flexibility
PRESENTATION_CONFIG = {
    'forecast_comparison_table': {
        'caption': 'Forecast accuracy comparison: CAPM vs FF3',
        'notes': 'RMSE reported in percentage points. Lower is better.',
    }
}

# === FILE PATHS ===
DATA_PATHS = {
    'crsp_file': 'data/CRSP 1970-2024.parquet',
    'ff_factors': 'data/F-F_Research_Data_Factors_daily.csv',
}

# === REMOVED PARAMETERS (were never really used) ===
# - include_alpha: Always comparing both
# - winsorize_returns: Always True now
# - validate_estimations: Too slow, not needed
# - diagnose_models: Too verbose
# - save_intermediate: Never used
# - check_data_quality: Always done
# - run_validation: Separate script if needed
# - analyze_by_size: Always done
# - size_quintiles: Always 5
# - horizon_analysis: Always done
# - calculate_vw_beta: Always done
# - alpha_subset_analysis: Always done
# - run_diebold_mariano: Always done
# - max_attempts_multiplier: Not needed with better sampling

# === UTILITY FUNCTIONS ===

def create_output_dirs():
    """Create necessary output directories."""
    dirs_to_create = [
        OUTPUT_CONFIG['results_dir'],
        os.path.join(OUTPUT_CONFIG['results_dir'], 'figures'),
        os.path.join(OUTPUT_CONFIG['results_dir'], 'tables'),
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)


# === FORMATTING UTILITIES ===

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal value as a percentage string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, kind: str = 'default') -> str:
    """Format numeric values with sensible defaults by type."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    decimals = {
        'coefficients': 4,
        'statistics': 2,
        'default': 2,
    }
    d = decimals.get(kind, decimals['default'])
    return f"{value:.{d}f}"


def get_significance_stars(p_value: float) -> str:
    """Return conventional significance stars for a p-value."""
    if p_value < 0.01:
        return '***'
    if p_value < 0.05:
        return '**'
    if p_value < 0.10:
        return '*'
    return ''

# Run setup when imported
create_output_dirs()

