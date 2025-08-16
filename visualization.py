"""
visualization.py - Visualization functions for research paper figures.
Focused on essential plots with clean, publication-ready styling.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, List
from scipy import stats

from config import OUTPUT_CONFIG, PRESENTATION_CONFIG
from evaluation import calculate_rmse

# === SECTION 1: SETUP ===

def setup_plot_style():
    """Set up publication-quality plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (8, 6),
        'figure.dpi': OUTPUT_CONFIG['figure_dpi'],
        'savefig.dpi': OUTPUT_CONFIG['figure_dpi'],
        'savefig.bbox': 'tight',
    })


# === SECTION 2: MAIN COMPARISON PLOT ===

def plot_error_comparison(errors_alpha: np.ndarray, errors_zero: np.ndarray,
                         save_path: Optional[str] = None,
                         error_range: Optional[tuple] = None,
                         scatter_range: Optional[tuple] = None) -> plt.Figure:
    """
    Create 4-panel error comparison figure.
    """
    # Convert to percentage
    errors_alpha_pct = errors_alpha * 100
    errors_zero_pct = errors_zero * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Loss differential histogram
    ax1 = axes[0, 0]
    loss_diff = errors_alpha_pct - errors_zero_pct
    ax1.hist(loss_diff, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    kde = stats.gaussian_kde(loss_diff)
    x_range = np.linspace(loss_diff.min(), loss_diff.max(), 200)
    ax1.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Error Differential (%) [$e_{\\alpha} - e_0$]')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Forecast Error Differential')
    ax1.legend()
    if error_range is not None:
        ax1.set_xlim(error_range)
    
    # Panel 2: Paired error scatter
    ax2 = axes[0, 1]
    ax2.scatter(errors_zero_pct, errors_alpha_pct, alpha=0.5, s=30)
    lim = [min(errors_zero_pct.min(), errors_alpha_pct.min()),
           max(errors_zero_pct.max(), errors_alpha_pct.max())]
    ax2.plot(lim, lim, 'r--', lw=2, label='45° line')
    ax2.set_xlabel('Error without Alpha (%)')
    ax2.set_ylabel('Error with Alpha (%)')
    ax2.set_title('(b) Paired Forecast Errors')
    ax2.legend()
    ax2.set_aspect('equal')
    if scatter_range is not None:
        ax2.set_xlim(scatter_range)
        ax2.set_ylim(scatter_range)
    
    # Panel 3: RMSE comparison
    ax3 = axes[1, 0]
    rmse_alpha = np.sqrt(np.mean(errors_alpha**2)) * 100
    rmse_zero = np.sqrt(np.mean(errors_zero**2)) * 100
    
    models = ['With Alpha', 'Without Alpha']
    rmse_values = [rmse_alpha, rmse_zero]
    bars = ax3.bar(models, rmse_values, color=['steelblue', 'coral'], alpha=0.8)
    
    for bar, val in zip(bars, rmse_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}%', ha='center', va='bottom')
    
    ax3.set_ylabel('RMSE (%)')
    ax3.set_title('(c) RMSE Comparison')
    ax3.set_ylim(0, max(rmse_values) * 1.2)
    
    # Panel 4: Error distributions
    ax4 = axes[1, 1]
    ax4.hist(errors_alpha_pct, bins=30, alpha=0.5, label='With Alpha', density=True)
    ax4.hist(errors_zero_pct, bins=30, alpha=0.5, label='Without Alpha', density=True)
    ax4.set_xlabel('Forecast Error (%)')
    ax4.set_ylabel('Density')
    ax4.set_title('(d) Error Distributions')
    ax4.legend()
    if error_range is not None:
        ax4.set_xlim(error_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# === SECTION 3: PARAMETER DISTRIBUTIONS ===

def plot_parameter_distributions(results_df: pd.DataFrame,
                               save_path: Optional[str] = None,
                               alpha_range: Optional[tuple] = None,
                               beta_range: Optional[tuple] = None) -> plt.Figure:
    """
    Plot alpha and beta distributions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Alpha distribution (in percentage)
    alpha_pct = results_df['alpha'] * 100
    ax1.hist(alpha_pct, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', label='Zero')
    ax1.axvline(alpha_pct.mean(), color='green', linestyle='-', 
                label=f'Mean = {alpha_pct.mean():.4f}%')
    ax1.set_xlabel('Alpha (% daily)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Alpha Distribution')
    ax1.legend()
    if alpha_range is not None:
        ax1.set_xlim(alpha_range)
    
    # Beta distribution
    ax2.hist(results_df['beta'], bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(1, color='red', linestyle='--', label='Market Beta')
    ax2.axvline(results_df['beta'].mean(), color='green', linestyle='-',
                label=f'Mean = {results_df["beta"].mean():.3f}')
    ax2.set_xlabel('Beta')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Beta Distribution')
    ax2.legend()
    if beta_range is not None:
        ax2.set_xlim(beta_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# === SECTION 4: HORIZON ANALYSIS ===

def plot_horizon_analysis(all_results: Dict[int, dict],
                         save_path: Optional[str] = None,
                         rmse_ylim: Optional[tuple] = None,
                         improve_ylim: Optional[tuple] = None) -> plt.Figure:
    """
    Plot performance across forecast horizons.
    """
    horizons = sorted(all_results.keys())
    rmse_alpha = [all_results[h]['rmse_alpha'] * 100 for h in horizons]
    rmse_zero = [all_results[h]['rmse_zero'] * 100 for h in horizons]
    improvements = [all_results[h]['rmse_improvement_pct'] for h in horizons]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Top: RMSE by horizon
    ax1.plot(horizons, rmse_alpha, 'o-', label='With Alpha', linewidth=2)
    ax1.plot(horizons, rmse_zero, 's-', label='Without Alpha', linewidth=2)
    ax1.set_ylabel('RMSE (%)')
    ax1.set_title('Forecast Error by Horizon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if rmse_ylim is not None:
        ax1.set_ylim(rmse_ylim)
    
    # Bottom: Improvement
    bars = ax2.bar(horizons, improvements, color='darkgreen', alpha=0.7)
    ax2.margins(y=0.2)  # add headroom for annotations

    # Add significance stars with offsets to keep them inside the plot
    for i, h in enumerate(horizons):
        if 'p_value' in all_results[h]:
            p_val = all_results[h]['p_value']
            stars = ''
            if p_val < 0.01:
                stars = '***'
            elif p_val < 0.05:
                stars = '**'
            elif p_val < 0.10:
                stars = '*'
            if stars:
                offset = 5 if improvements[i] >= 0 else -5
                va = 'bottom' if improvements[i] >= 0 else 'top'
                ax2.annotate(
                    stars,
                    xy=(h, improvements[i]),
                    xytext=(0, offset),
                    textcoords='offset points',
                    ha='center',
                    va=va
                )
    
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Forecast Horizon (days)')
    ax2.set_ylabel('RMSE Improvement (%)')
    ax2.set_title('Improvement from Including Alpha')
    ax2.set_xticks(horizons)
    if improve_ylim is not None:
        ax2.set_ylim(improve_ylim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

    return fig


def plot_annualized_alpha_analysis(horizon_results: Dict[int, dict],
                                  save_path: Optional[str] = None) -> plt.Figure:
    """Visualize annualized alpha across horizons and samples."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    horizons = sorted(horizon_results.keys())

    ax1 = axes[0, 0]
    annual_alphas = []
    for h in horizons:
        daily_alpha = horizon_results[h]['results_df']['alpha'].mean()
        annual_alphas.append(daily_alpha * 252 * 100)
    ax1.bar(horizons, annual_alphas, color='steelblue', alpha=0.7)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Forecast Horizon (days)')
    ax1.set_ylabel('Annualized Alpha (%)')
    ax1.set_title('Mean Annualized Alpha by Horizon')
    ax1.set_xticks(horizons)

    ax2 = axes[0, 1]
    if 1 in horizon_results:
        annual_alphas_dist = horizon_results[1]['results_df']['alpha'] * 252 * 100
        ax2.hist(annual_alphas_dist, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', label='Zero')
        ax2.axvline(annual_alphas_dist.mean(), color='green',
                    label=f'Mean = {annual_alphas_dist.mean():.1f}%')
        ax2.set_xlabel('Annualized Alpha (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Annualized Alphas (1-day horizon)')
        ax2.legend()

    ax3 = axes[1, 0]
    for h in horizons:
        results_df = horizon_results[h]['results_df']
        cumulative_alpha = results_df['alpha'].mean() * h * 100
        ax3.scatter(h, cumulative_alpha, s=100, label=f'{h}-day')
    ax3.set_xlabel('Forecast Horizon (days)')
    ax3.set_ylabel('Cumulative Alpha Effect (%)')
    ax3.set_title('Cumulative Alpha Impact by Horizon')
    ax3.legend()

    ax4 = axes[1, 1]
    sig_props = []
    for h in horizons:
        if 'p_alpha' in horizon_results[h]['results_df'].columns:
            sig_prop = (horizon_results[h]['results_df']['p_alpha'] < 0.05).mean() * 100
        else:
            sig_prop = 0
        sig_props.append(sig_prop)
    ax4.bar(horizons, sig_props, color='coral', alpha=0.7)
    ax4.axhline(5, color='red', linestyle='--', alpha=0.5, label='5% (by chance)')
    ax4.set_xlabel('Forecast Horizon (days)')
    ax4.set_ylabel('% Significant at 5%')
    ax4.set_title('Proportion of Significant Alphas')
    ax4.set_xticks(horizons)
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


# === SECTION 5: CROSS-SECTIONAL ANALYSIS ===

def plot_size_analysis(results_df: pd.DataFrame,
                      save_path: Optional[str] = None,
                      rmse_ylim: Optional[tuple] = None,
                      beta_ylim: Optional[tuple] = None) -> plt.Figure:
    """
    Plot performance by market cap quintiles.
    """
    # Create quintiles
    results_df['size_quintile'] = pd.qcut(results_df['market_cap'], q=5, 
                                         labels=['Q1 (Small)', 'Q2', 'Q3', 'Q4', 'Q5 (Large)'])
    
    # Calculate metrics by quintile
    metrics = []
    for q in results_df['size_quintile'].cat.categories:
        q_data = results_df[results_df['size_quintile'] == q]
        rmse_alpha = np.sqrt(np.mean(q_data['error_alpha']**2)) * 100
        rmse_zero = np.sqrt(np.mean(q_data['error_zero']**2)) * 100
        improvement = (rmse_zero - rmse_alpha) / rmse_zero * 100 if rmse_zero > 0 else 0
        
        metrics.append({
            'quintile': q,
            'rmse_alpha': rmse_alpha,
            'rmse_zero': rmse_zero,
            'improvement': improvement,
            'mean_beta': q_data['beta'].mean(),
            'n_obs': len(q_data)
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE by size
    x = np.arange(len(metrics_df))
    width = 0.35
    ax1.bar(x - width/2, metrics_df['rmse_alpha'], width, label='With Alpha', alpha=0.8)
    ax1.bar(x + width/2, metrics_df['rmse_zero'], width, label='Without Alpha', alpha=0.8)
    ax1.set_xlabel('Market Cap Quintile')
    ax1.set_ylabel('RMSE (%)')
    ax1.set_title('Forecast Error by Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['quintile'])
    ax1.legend()
    if rmse_ylim is not None:
        ax1.set_ylim(rmse_ylim)
    
    # Beta by size
    ax2.bar(x, metrics_df['mean_beta'], alpha=0.8)
    ax2.axhline(1, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Market Cap Quintile')
    ax2.set_ylabel('Average Beta')
    ax2.set_title('Beta by Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_df['quintile'])
    if beta_ylim is not None:
        ax2.set_ylim(beta_ylim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# === SECTION 6: SUMMARY TABLE ===

def create_summary_table(all_results: Dict[int, dict], 
                        format_type: str = 'latex') -> str:
    """
    Create summary table for paper.
    """
    rows = []
    for horizon in sorted(all_results.keys()):
        r = all_results[horizon]
        row = {
            'Horizon': horizon,
            'N': r['n_samples'],
            'RMSE (α)': f"{r['rmse_alpha']*100:.3f}",
            'RMSE (0)': f"{r['rmse_zero']*100:.3f}",
            'Improvement': f"{r['rmse_improvement_pct']:.2f}%",
            't-stat': f"{r['t_statistic']:.3f}",
            'p-value': f"{r['p_value']:.3f}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if format_type == 'latex':
        return df.to_latex(index=False, escape=False)
    else:
        return df.to_markdown(index=False)


# === SECTION 7: UTILITY FUNCTIONS ===

def save_all_figures(figures: Dict[str, plt.Figure], 
                    base_path: str = './results/figures/'):
    """
    Save all figures with consistent naming.
    """
    import os
    os.makedirs(base_path, exist_ok=True)
    
    for name, fig in figures.items():
        if fig is not None:
            filepath = os.path.join(base_path, f"{name}.{OUTPUT_CONFIG['figure_format']}")
            fig.savefig(filepath, bbox_inches='tight', dpi=OUTPUT_CONFIG['figure_dpi'])
            print(f"Saved: {filepath}")


def adjust_figure_for_presentation(fig: plt.Figure,
                                 presentation_mode: bool = True) -> plt.Figure:
    """
    Adjust figure styling for presentation vs paper mode.
    """
    if presentation_mode:
        # Increase font sizes for visibility
        for ax in fig.get_axes():
            ax.title.set_fontsize(ax.title.get_fontsize() * 1.4)
            ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * 1.3)
            ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * 1.3)
            
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(tick.label.get_fontsize() * 1.2)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick.label.get_fontsize() * 1.2)
            
            if ax.get_legend():
                ax.legend(fontsize=ax.get_legend().get_texts()[0].get_fontsize() * 1.2)
        
        # Thicker lines
        for ax in fig.get_axes():
            for line in ax.get_lines():
                line.set_linewidth(line.get_linewidth() * 1.5)

    return fig


def rescale_axes(fig: plt.Figure,
                 xlim: Optional[tuple] = None,
                 ylim: Optional[tuple] = None) -> plt.Figure:
    """Convenience helper to uniformly adjust axes limits."""
    for ax in fig.get_axes():
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    return fig


# === SECTION 8: MODEL COMPARISON PLOT ===

def plot_model_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Create comparison plot for CAPM and FF3 models.

    Falls back to a CAPM-only plot when FF3 columns are absent.
    """

    required_ff3 = {
        'error_ff3_alpha', 'error_ff3_zero', 'r2_ff3',
        'alpha_ff3', 'beta_mkt_ff3', 'beta_smb', 'beta_hml'
    }
    has_ff3 = required_ff3.issubset(results_df.columns)

    if not has_ff3:
        warnings.warn("FF3 results missing; plotting CAPM metrics only")
        fig, ax = plt.subplots(figsize=(6, 4))
        models = ['CAPM+α', 'CAPM']
        rmses = [
            calculate_rmse(results_df['error_capm_alpha']),
            calculate_rmse(results_df['error_capm_zero'])
        ]
        ax.bar(models, np.array(rmses) * 100, color=['blue', 'lightblue'])
        ax.set_ylabel('RMSE (%)')
        ax.set_title('CAPM Forecast Accuracy')
        plt.tight_layout()
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    models = ['CAPM+α', 'CAPM', 'FF3+α', 'FF3']
    rmses = [
        calculate_rmse(results_df['error_capm_alpha']),
        calculate_rmse(results_df['error_capm_zero']),
        calculate_rmse(results_df['error_ff3_alpha']),
        calculate_rmse(results_df['error_ff3_zero'])
    ]

    ax1 = axes[0, 0]
    colors = ['blue', 'lightblue', 'green', 'lightgreen']
    ax1.bar(models, np.array(rmses) * 100, color=colors)
    ax1.set_ylabel('RMSE (%)')
    ax1.set_title('Forecast Accuracy Comparison')

    ax2 = axes[0, 1]
    ax2.scatter(results_df['r2_capm'], results_df['r2_ff3'], alpha=0.5)
    ax2.plot([0, 1], [0, 1], 'r--', label='Equal R²')
    ax2.set_xlabel('CAPM R²')
    ax2.set_ylabel('FF3 R²')
    ax2.set_title('Explanatory Power: FF3 vs CAPM')

    ax3 = axes[1, 0]
    ax3.scatter(results_df['alpha_capm'] * 100,
                results_df['alpha_ff3'] * 100, alpha=0.5)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('CAPM Alpha (%)')
    ax3.set_ylabel('FF3 Alpha (%)')
    ax3.set_title('Alpha Comparison')

    ax4 = axes[1, 1]
    factor_data = [
        results_df['beta_capm'],
        results_df['beta_mkt_ff3'],
        results_df['beta_smb'],
        results_df['beta_hml'],
    ]
    ax4.boxplot(factor_data, labels=['CAPM β', 'FF3 β_MKT', 'β_SMB', 'β_HML'])
    ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Factor Loadings Distribution')

    plt.tight_layout()
    return fig
