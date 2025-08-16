"""
config.py - Configuration parameters for event study analysis.
All parameters are centralized here for easy modification and paper reporting.
"""

import os

# === SECTION 1: SAMPLING PARAMETERS ===
SAMPLING_CONFIG = {
    'n_samples': 100,                    # Number of random draws
    'estimation_window': 252,            # Trading days for estimation (1 year)
    'min_observations': 150,             # Minimum non-missing returns required
    'forecast_horizons': [1, 5, 10, 20], # Forecast horizons in days
    'random_seed': 42,                   # For reproducibility
    'max_attempts_multiplier': 20,       # Max attempts = n_samples * this
}

# === SECTION 2: DATA FILTERS ===
DATA_FILTERS = {
    'min_price': 1.0,                    # Penny stock filter (absolute price)
    'shrcd_codes': [10, 11],             # Common shares only
    'exchcd_codes': [1, 2, 3],           # NYSE, AMEX, NASDAQ
    'start_date': '1970-01-02',          # Sample start
    'end_date': '2024-12-31',            # Sample end
    'min_market_cap': 100,               # Minimum market cap in millions
}

# === SECTION 3: MODEL SPECIFICATIONS ===
MODEL_CONFIG = {
    'base_model': 'capm',                # 'capm' or 'ff3' or 'ff5'
    'include_alpha': [True, False],      # Compare with and without intercept
    'winsorize_returns': True,           # Enable return winsorization
    'winsorize_level': 0.005,            # 0.5% winsorization in each tail
    'validate_estimations': False,       # Skip diagnostic checks unless needed
    'diagnose_models': False,            # Disable model diagnostics by default
}

# === SECTION 4: ANALYSIS OPTIONS ===
ANALYSIS_CONFIG = {
    # Core analyses
    'calculate_vw_beta': True,           # Value-weighted beta analysis
    'alpha_subset_analysis': True,       # Analyze high-alpha subset
    'alpha_percentile_cutoff': 50,       # Top X% by |alpha|
    'run_diebold_mariano': True,         # Formal forecast comparison test
    'bootstrap_iterations': 1000,        # For confidence intervals
    
    # Validation and diagnostics
    'run_validation': True,              # Run methodology validation before analysis
    'check_data_quality': True,          # Validate data quality
    'diagnose_models': True,             # Model diagnostics for each estimation
    
    # Additional analyses
    'analyze_by_size': True,             # Cross-sectional analysis by market cap
    'size_quintiles': 5,                 # Number of size groups
    'horizon_analysis': True,            # Compare across forecast horizons
}

# === SECTION 5: OUTPUT SETTINGS ===
OUTPUT_CONFIG = {
    'save_results': True,
    'results_dir': './results/',
    'figure_format': 'pdf',              # For paper submission
    'figure_dpi': 300,
    'table_format': 'latex',             # For paper tables
    'verbose': True,                     # Detailed console output
    'save_intermediate': False,          # Save intermediate results
}

# === SECTION 6: FILE PATHS ===
DATA_PATHS = {
    'data_dir': './data/',
    'crsp_file': 'data/CRSP 1970-2024.parquet',
    'ff_factors': 'data/F-F_Research_Data_Factors_daily.csv',
}

# === SECTION 7: VALIDATION SETTINGS ===
VALIDATION_CONFIG = {
    'n_validation_tests': 50,            # Number of tests for artificial return detection
    'artificial_return_size': 0.01,      # Size of artificial return to inject (1%)
    'n_null_simulations': 100,           # Simulations for null hypothesis validation
    'validation_seed': 123,              # Separate seed for validation
}

# === SECTION 8: RESULTS PRESENTATION ===
PRESENTATION_CONFIG = {
    # Scaling for percentage display
    'return_scale': 100,                 # Convert to percentage (multiply by 100)
    'precision': {
        'returns': 4,                    # Decimal places for returns (e.g., 0.0234%)
        'rmse': 5,                       # Decimal places for RMSE
        'statistics': 3,                 # Decimal places for t-stats, p-values
        'coefficients': 5,               # Decimal places for alpha, beta
    },

    # Optional axis limits for plots (None = auto)
    'axis_limits': {
        'error_range': None,
        'scatter_range': None,
        'alpha_range': None,
        'beta_range': None,
        'horizon_rmse_ylim': None,
        'horizon_improve_ylim': None,
        'size_rmse_ylim': None,
        'size_beta_ylim': None,
    },
    
    # Statistical significance stars
    'significance_levels': {
        '***': 0.01,
        '**': 0.05,
        '*': 0.10
    },
    
    # Table formatting
    'table_caption': "Forecast Performance: Market Model With and Without Intercept",
    'table_notes': """Notes: This table reports out-of-sample forecast performance for market model predictions.
RMSE (α) includes the intercept term, RMSE (0) excludes it. The t-statistic tests equal forecast accuracy
using a paired t-test. DM is the Diebold-Mariano statistic. Returns and RMSEs are in percentage terms.
Significance levels: *** p<0.01, ** p<0.05, * p<0.10."""
}

# === SECTION 9: COMPUTATIONAL SETTINGS ===
COMPUTE_CONFIG = {
    'n_jobs': -1,                        # Number of parallel jobs (-1 = all cores)
    'chunk_size': 1000,                  # Process data in chunks
    'memory_limit': '4GB',               # Maximum memory usage
    'show_progress': True,               # Show progress bars
}

# === UTILITY FUNCTIONS ===

def create_output_dirs():
    """Create necessary output directories if they don't exist."""
    dirs_to_create = [
        OUTPUT_CONFIG['results_dir'],
        os.path.join(OUTPUT_CONFIG['results_dir'], 'figures'),
        os.path.join(OUTPUT_CONFIG['results_dir'], 'tables'),
        os.path.join(OUTPUT_CONFIG['results_dir'], 'data'),
        os.path.join(OUTPUT_CONFIG['results_dir'], 'validation')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

def get_significance_stars(p_value: float) -> str:
    """Return significance stars based on p-value."""
    for stars, threshold in PRESENTATION_CONFIG['significance_levels'].items():
        if p_value < threshold:
            return stars
    return ''

def format_percentage(value: float, precision: int = None) -> str:
    """Format a value as percentage with appropriate precision."""
    if precision is None:
        precision = PRESENTATION_CONFIG['precision']['returns']
    return f"{value * PRESENTATION_CONFIG['return_scale']:.{precision}f}%"

def format_number(value: float, number_type: str = 'statistics') -> str:
    """Format a number with appropriate precision based on type."""
    precision = PRESENTATION_CONFIG['precision'].get(number_type, 3)
    return f"{value:.{precision}f}"

# === VALIDATION FUNCTIONS ===

def validate_config():
    """Validate configuration settings."""
    errors = []
    # Check logical consistency
    if SAMPLING_CONFIG['min_observations'] > SAMPLING_CONFIG['estimation_window']:
        errors.append("min_observations cannot exceed estimation_window")
    
    if MODEL_CONFIG['base_model'] not in ['capm', 'ff3', 'ff5']:
        errors.append(f"Unknown model: {MODEL_CONFIG['base_model']}")
    
    # Check forecast horizons
    if not all(h > 0 for h in SAMPLING_CONFIG['forecast_horizons']):
        errors.append("All forecast horizons must be positive")
    
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

# Run validation when module is imported
if __name__ != "__main__":
    if not validate_config():
        print("\nPlease fix configuration errors before proceeding.")
    else:
        # Create output directories
        create_output_dirs()
        print("Configuration loaded successfully.")
