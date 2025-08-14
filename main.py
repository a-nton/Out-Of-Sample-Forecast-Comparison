"""
main.py - Main execution script for Event Study Market Model analysis.
Orchestrates the complete analysis pipeline with validation and diagnostics.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path if using modular structure
sys.path.append('src')

# Import all modules
from config import *
from data_loader import (
    load_crsp_data, load_ff_factors, prepare_analysis_data, 
    validate_merged_data, create_data_summary_table
)
from models import (
    estimate_capm, estimate_ff3, forecast_capm_return, forecast_ff3_return,
    calculate_vw_beta, analyze_alpha_persistence, diagnose_estimation_quality
)
from sampling import sample_events, analyze_sample_characteristics, validate_sampling_randomness
from evaluation import (
    calculate_forecast_errors, calculate_rmse, calculate_mae, paired_t_test,
    diebold_mariano_test, bootstrap_rmse, analyze_by_characteristic,
    analyze_alpha_subset, calculate_vw_statistics, create_evaluation_report,
    create_forecast_comparison_table
)
from visualization import (
    setup_plot_style, plot_error_comparison, plot_parameter_distributions,
    plot_horizon_analysis, plot_size_analysis, create_summary_table,
    save_all_figures, adjust_figure_for_presentation
)

# Import validation module if it exists
try:
    from validation import run_methodology_validation
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    print("Note: Validation module not found. Skipping methodology validation.")


def main():
    """
    Main execution function that runs the complete analysis pipeline.
    """
    # Record start time
    start_time = datetime.now()
    
    # Setup
    setup_plot_style()
    
    print("="*70)
    print("EVENT STUDY MARKET MODEL FORECASTING")
    print("="*70)
    print(f"\nStart time: {start_time}")
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_CONFIG['base_model'].upper()}")
    print(f"  Samples: {SAMPLING_CONFIG['n_samples']}")
    print(f"  Estimation window: {SAMPLING_CONFIG['estimation_window']} trading days")
    print(f"  Forecast horizons: {SAMPLING_CONFIG['forecast_horizons']}")
    print(f"  Random seed: {SAMPLING_CONFIG['random_seed']}")
    
    # === SECTION 1: DATA LOADING AND VALIDATION ===
    print("\n" + "="*70)
    print("SECTION 1: DATA LOADING AND PREPARATION")
    print("="*70)
    
    # Load data
    try:
        crsp_df = load_crsp_data()
        ff_df = load_ff_factors(factor_model=MODEL_CONFIG['base_model'][:3])
    except Exception as e:
        print(f"\nERROR loading data: {str(e)}")
        print("Please ensure data files are in the correct location:")
        print(f"  - {DATA_PATHS['crsp_file']}")
        print(f"  - {DATA_PATHS['ff_factors']}")
        return
    
    # Merge and prepare
    merged_df = prepare_analysis_data(crsp_df, ff_df, apply_filters=True, validate=True)
    
    # Create data summary for paper
    data_summary = create_data_summary_table(merged_df)
    data_summary.to_csv(os.path.join(OUTPUT_CONFIG['results_dir'], 'data_summary.csv'), index=False)
    
    # === SECTION 2: METHODOLOGY VALIDATION (if available) ===
    if VALIDATION_AVAILABLE and ANALYSIS_CONFIG.get('run_validation', True):
        print("\n" + "="*70)
        print("SECTION 2: METHODOLOGY VALIDATION")
        print("="*70)
        
        validation_results = run_methodology_validation(merged_df, SAMPLING_CONFIG)
        
        # Check if validation passed
        all_passed = all(r.get('passed', False) for r in validation_results.values())
        
        if not all_passed:
            response = input("\nValidation found issues. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting due to validation failures.")
                return
        else:
            print("\nAll validation tests passed! Proceeding with analysis.")
        
        time.sleep(2)
    
    # === SECTION 3: SAMPLING AND ESTIMATION ===
    print("\n" + "="*70)
    print("SECTION 3: SAMPLING AND MODEL ESTIMATION")
    print("="*70)
    
    # Sample events for all horizons
    all_samples = sample_events(
        merged_df,
        n_samples=SAMPLING_CONFIG['n_samples'],
        estimation_window=SAMPLING_CONFIG['estimation_window'],
        forecast_horizons=SAMPLING_CONFIG['forecast_horizons'],
        config={**SAMPLING_CONFIG, **DATA_FILTERS},
        random_seed=SAMPLING_CONFIG['random_seed'],
        verbose=OUTPUT_CONFIG['verbose']
    )
    
    # Analyze sample characteristics
    sample_chars = analyze_sample_characteristics(all_samples)
    print("\nSample Characteristics:")
    print(sample_chars)
    sample_chars.to_csv(os.path.join(OUTPUT_CONFIG['results_dir'], 'sample_characteristics.csv'), index=False)
    
    # Validate randomness
    if len(all_samples[SAMPLING_CONFIG['forecast_horizons'][0]]) >= 50:
        randomness_check = validate_sampling_randomness(all_samples)
        print("\nRandomness Validation:")
        for key, value in randomness_check.items():
            if isinstance(value, dict) and 'interpretation' in value:
                print(f"  {key}: {value['interpretation']}")
    
    # === SECTION 4: MODEL ESTIMATION AND FORECASTING ===
    print("\n" + "="*70)
    print("SECTION 4: MODEL ESTIMATION AND FORECASTING")
    print("="*70)
    
    # Store results for each horizon
    results_by_horizon = {}
    all_diagnostics = []
    
    for horizon in SAMPLING_CONFIG['forecast_horizons']:
        print(f"\nProcessing {horizon}-day horizon...")
        horizon_samples = all_samples[horizon]
        
        results = []
        estimation_issues = []
        
        for i, sample in enumerate(horizon_samples):
            try:
                # Estimate model based on configuration
                if MODEL_CONFIG['base_model'] == 'capm':
                    alpha, beta, stats = estimate_capm(
                        sample['estimation_data'],
                        min_obs=SAMPLING_CONFIG['min_observations'],
                        validate_data=(i == 0)  # Validate only first estimation
                    )
                    
                    # Generate forecasts
                    forecast_row = sample['forecast_data']
                    mkt_excess = forecast_row['Mkt-RF']
                    rf = forecast_row['RF']
                    
                    forecast_alpha = forecast_capm_return(alpha, beta, mkt_excess, rf, use_alpha=True)
                    forecast_zero = forecast_capm_return(alpha, beta, mkt_excess, rf, use_alpha=False)
                    
                elif MODEL_CONFIG['base_model'] == 'ff3':
                    coefficients, stats = estimate_ff3(
                        sample['estimation_data'],
                        min_obs=SAMPLING_CONFIG['min_observations'],
                        validate_data=(i == 0)
                    )
                    
                    alpha = coefficients[0]
                    beta = coefficients[1]  # Market beta
                    
                    # Generate forecasts
                    forecast_row = sample['forecast_data']
                    factors = {
                        'Mkt-RF': forecast_row['Mkt-RF'],
                        'SMB': forecast_row['SMB'],
                        'HML': forecast_row['HML']
                    }
                    rf = forecast_row['RF']
                    
                    forecast_alpha = forecast_ff3_return(coefficients, factors, rf, use_alpha=True)
                    forecast_zero = forecast_ff3_return(coefficients, factors, rf, use_alpha=False)
                
                # Calculate errors
                actual = forecast_row['RET']
                error_alpha = actual - forecast_alpha
                error_zero = actual - forecast_zero
                
                # Diagnostics
                if MODEL_CONFIG.get('diagnose_models', True):
                    diagnostics = diagnose_estimation_quality(alpha, beta, stats)
                    if diagnostics:
                        all_diagnostics.append(diagnostics)
                        # Only print warnings for severe issues
                        severe_issues = [d for d in diagnostics.values() 
                                       if isinstance(d, dict) and d.get('severity') == 'HIGH']
                        if severe_issues:
                            estimation_issues.append({
                                'sample': i,
                                'permno': sample['permno'],
                                'issues': severe_issues
                            })
                
                # Store results
                result = {
                    'permno': sample['permno'],
                    'estimation_end': sample['estimation_end'],
                    'forecast_date': sample['forecast_date'],
                    'horizon': horizon,
                    'alpha': alpha,
                    'beta': beta,
                    'actual': actual,
                    'forecast_alpha': forecast_alpha,
                    'forecast_zero': forecast_zero,
                    'error_alpha': error_alpha,
                    'error_zero': error_zero,
                    'market_cap': sample.get('mean_market_cap', np.nan),
                    **stats  # Include model statistics
                }
                
                results.append(result)
                
            except Exception as e:
                if OUTPUT_CONFIG['verbose']:
                    print(f"  Warning: Failed to process sample {i}: {str(e)}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_by_horizon[horizon] = results_df
        
        print(f"  Successfully processed {len(results)}/{len(horizon_samples)} samples")
        
        if estimation_issues:
            print(f"  Found {len(estimation_issues)} estimations with severe issues")
    
    # === SECTION 5: EVALUATION AND TESTING ===
    print("\n" + "="*70)
    print("SECTION 5: STATISTICAL EVALUATION")
    print("="*70)
    
    # Analyze each horizon
    horizon_results = {}
    
    for horizon in SAMPLING_CONFIG['forecast_horizons']:
        print(f"\n{horizon}-Day Horizon Results:")
        print("-" * 40)
        
        results_df = results_by_horizon[horizon]
        
        # Extract errors
        errors_alpha, errors_zero = calculate_forecast_errors(results_df.to_dict('records'))
        
        # Calculate metrics
        rmse_alpha = calculate_rmse(errors_alpha)
        rmse_zero = calculate_rmse(errors_zero)
        mae_alpha = calculate_mae(errors_alpha)
        mae_zero = calculate_mae(errors_zero)
        
        # Statistical tests
        t_stat, p_val = paired_t_test(errors_alpha, errors_zero)
        dm_stat, dm_pval, dm_details = diebold_mariano_test(errors_alpha, errors_zero, horizon=horizon)
        
        # Bootstrap CIs
        ci_alpha = bootstrap_rmse(errors_alpha, n_bootstrap=ANALYSIS_CONFIG['bootstrap_iterations'])
        ci_zero = bootstrap_rmse(errors_zero, n_bootstrap=ANALYSIS_CONFIG['bootstrap_iterations'])
        
        # Store results
        horizon_results[horizon] = {
            'horizon': horizon,
            'n_samples': len(results_df),
            'results_df': results_df,
            'errors_alpha': errors_alpha,
            'errors_zero': errors_zero,
            'rmse_alpha': rmse_alpha,
            'rmse_zero': rmse_zero,
            'mae_alpha': mae_alpha,
            'mae_zero': mae_zero,
            'rmse_improvement_pct': (rmse_zero - rmse_alpha) / rmse_zero * 100 if rmse_zero > 0 else 0,
            't_statistic': t_stat,
            'p_value': p_val,
            'dm_statistic': dm_stat,
            'dm_pvalue': dm_pval,
            'ci_alpha': ci_alpha,
            'ci_zero': ci_zero,
        }
        
        # Print summary
        print(f"RMSE with alpha:    {format_percentage(rmse_alpha)} [{format_percentage(ci_alpha[0])}, {format_percentage(ci_alpha[1])}]")
        print(f"RMSE without alpha: {format_percentage(rmse_zero)} [{format_percentage(ci_zero[0])}, {format_percentage(ci_zero[1])}]")
        print(f"Improvement:        {horizon_results[horizon]['rmse_improvement_pct']:.2f}%")
        print(f"Paired t-test:      t={t_stat:.3f}, p={p_val:.4f}")
        print(f"Diebold-Mariano:    DM={dm_stat:.3f}, p={dm_pval:.4f}")
    
    # === SECTION 6: ADDITIONAL ANALYSES ===
    print("\n" + "="*70)
    print("SECTION 6: ADDITIONAL ANALYSES")
    print("="*70)
    
    # Use 1-day horizon for additional analyses
    main_results = results_by_horizon[1]
    
    # 6.1 Value-Weighted Analysis
    if ANALYSIS_CONFIG['calculate_vw_beta'] and 'market_cap' in main_results.columns:
        print("\n6.1 Value-Weighted Beta Analysis")
        print("-" * 40)
        
        vw_beta_results = calculate_vw_beta(main_results)
        print(f"Equal-weighted beta: {vw_beta_results['ew_beta']:.3f}")
        print(f"Value-weighted beta: {vw_beta_results['vw_beta']:.3f}")
        print(f"Sanity check: {vw_beta_results['sanity_check']}")
        print(f"Size-beta correlation: {vw_beta_results['size_beta_correlation']:.3f}")
        
        # VW forecast errors
        vw_stats = calculate_vw_statistics(main_results)
        if 'vw_rmse_improvement' in vw_stats:
            print(f"\nValue-weighted RMSE improvement: {vw_stats['vw_rmse_improvement']:.2f}%")
            print(f"Equal-weighted RMSE improvement: {vw_stats['ew_rmse_improvement']:.2f}%")
    
    # 6.2 Alpha Subset Analysis
    if ANALYSIS_CONFIG['alpha_subset_analysis']:
        print("\n6.2 High-Alpha Subset Analysis")
        print("-" * 40)
        
        alpha_analysis = analyze_alpha_subset(
            main_results,
            percentile_cutoff=ANALYSIS_CONFIG['alpha_percentile_cutoff']
        )
        
        print(f"Top {ANALYSIS_CONFIG['alpha_percentile_cutoff']}% by |α|:")
        if alpha_analysis['high_alpha_group']:
            print(f"  High |α| improvement: {alpha_analysis['high_alpha_group']['rmse_improvement_pct']:.2f}%")
            print(f"  Low |α| improvement:  {alpha_analysis['low_alpha_group']['rmse_improvement_pct']:.2f}%")
            print(f"  {alpha_analysis['conclusion']}")
    
    # 6.3 Size Analysis
    if ANALYSIS_CONFIG['analyze_by_size'] and 'market_cap' in main_results.columns:
        print("\n6.3 Analysis by Market Cap")
        print("-" * 40)
        
        size_analysis = analyze_by_characteristic(
            main_results,
            characteristic='market_cap',
            n_groups=ANALYSIS_CONFIG['size_quintiles']
        )
        print("\nRMSE Improvement by Size Quintile:")
        for _, row in size_analysis.iterrows():
            print(f"  {row['group']}: {row['rmse_improvement_pct']:.2f}% (n={row['n_obs']})")
    
    # === SECTION 7: VISUALIZATION ===
    print("\n" + "="*70)
    print("SECTION 7: CREATING VISUALIZATIONS")
    print("="*70)
    
    figures = {}
    
    # Main error comparison (1-day horizon)
    if 1 in horizon_results:
        figures['error_comparison'] = plot_error_comparison(
            horizon_results[1]['errors_alpha'],
            horizon_results[1]['errors_zero']
        )
        
        figures['parameter_distributions'] = plot_parameter_distributions(
            horizon_results[1]['results_df']
        )
    
    # Multi-horizon analysis
    if len(SAMPLING_CONFIG['forecast_horizons']) > 1:
        figures['horizon_analysis'] = plot_horizon_analysis(horizon_results)
    
    # Size analysis
    if ANALYSIS_CONFIG['analyze_by_size'] and 'market_cap' in main_results.columns:
        figures['size_analysis'] = plot_size_analysis(main_results)
    
    # Save all figures
    save_all_figures(figures, base_path=os.path.join(OUTPUT_CONFIG['results_dir'], 'figures'))
    
    # === SECTION 8: SAVE RESULTS AND CREATE TABLES ===
    print("\n" + "="*70)
    print("SECTION 8: SAVING RESULTS")
    print("="*70)
    
    # Save detailed results for each horizon
    for horizon, results in horizon_results.items():
        results['results_df'].to_csv(
            os.path.join(OUTPUT_CONFIG['results_dir'], f'results_h{horizon}.csv'),
            index=False
        )
    
    # Create and save summary table
    summary_table = create_forecast_comparison_table(
        horizon_results,
        format_type=OUTPUT_CONFIG['table_format']
    )
    
    with open(os.path.join(OUTPUT_CONFIG['results_dir'], 'summary_table.tex'), 'w') as f:
        f.write(summary_table)
    
    # Create evaluation report for main horizon
    eval_report = create_evaluation_report(
        horizon_results[1]['results_df'],
        horizon=1,
        save_path=os.path.join(OUTPUT_CONFIG['results_dir'], 'evaluation_report.txt')
    )
    
    # Save configuration for reproducibility
    import json
    config_dict = {
        'SAMPLING_CONFIG': SAMPLING_CONFIG,
        'DATA_FILTERS': DATA_FILTERS,
        'MODEL_CONFIG': MODEL_CONFIG,
        'ANALYSIS_CONFIG': ANALYSIS_CONFIG,
        'run_date': start_time.isoformat()
    }
    
    with open(os.path.join(OUTPUT_CONFIG['results_dir'], 'config_used.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # === SECTION 9: FINAL SUMMARY ===
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    
    # Print key findings
    print("\nKey Findings:")
    for horizon in sorted(horizon_results.keys()):
        r = horizon_results[horizon]
        sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.10 else ""
        print(f"  {horizon}-day: {r['rmse_improvement_pct']:.2f}% improvement {sig}")
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\nEnd time: {end_time}")
    print(f"Total duration: {duration}")
    print(f"\nResults saved to: {OUTPUT_CONFIG['results_dir']}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
