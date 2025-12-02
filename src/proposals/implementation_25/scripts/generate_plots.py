#!/usr/bin/env python3.11
"""
Generate all plots and visualizations for NumGeom-Fair paper.

Creates publication-quality figures from experiment data.
All figures saved as both PNG and PGF (for LaTeX).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})


def load_experiment_data(exp_num: int, exp_name: str, base_dir: Path) -> Dict:
    """Load experiment results from JSON."""
    data_path = base_dir / f'experiment{exp_num}' / f'{exp_name}.json'
    
    if not data_path.exists():
        print(f"Warning: {data_path} not found")
        return {}
    
    with open(data_path, 'r') as f:
        return json.load(f)


def save_figure(fig, name: str, output_dir: Path):
    """Save figure as PNG and optionally PGF."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as PNG
    png_path = output_dir / f'{name}.png'
    fig.savefig(png_path, format='png', bbox_inches='tight')
    print(f"  Saved: {png_path}")
    
    # Save as PGF for LaTeX (if tex is available)
    try:
        pgf_path = output_dir / f'{name}.pgf'
        fig.savefig(pgf_path, format='pgf', bbox_inches='tight')
        print(f"  Saved: {pgf_path}")
    except Exception as e:
        print(f"  Note: Could not save PGF (LaTeX may not be available): {e}")
    
    plt.close(fig)


def plot_fairness_with_error_bars(data: Dict, output_dir: Path):
    """
    Figure 1: Fairness metrics with error bars.
    Bar chart showing DPG with error bars, colored by reliability.
    """
    print("\n[Figure 1] Fairness with Error Bars")
    
    if not data or 'datasets' not in data:
        print("  Skipping: no data")
        return
    
    # Extract data for plotting
    labels = []
    dpg_values = []
    error_bounds = []
    colors = []
    
    for dataset in data['datasets']:
        dataset_name = dataset['name']
        for prec_name in ['float64', 'float32', 'float16']:
            if prec_name in dataset['results']['precisions']:
                prec_data = dataset['results']['precisions'][prec_name]
                
                label = f"{dataset_name}\n{prec_name}"
                dpg = prec_data['dpg']
                error = prec_data['error_bound']
                reliable = prec_data['is_reliable']
                
                labels.append(label)
                dpg_values.append(dpg)
                error_bounds.append(error)
                colors.append('green' if reliable else 'red')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, dpg_values, yerr=error_bounds, 
                  color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Dataset and Precision')
    ax.set_ylabel('Demographic Parity Gap')
    ax.set_title('Fairness Metrics with Certified Error Bounds')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Reliable'),
        Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Borderline')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    save_figure(fig, 'fairness_error_bars', output_dir)


def plot_threshold_stability_ribbon(data: Dict, output_dir: Path):
    """
    Figure 2: Threshold stability ribbon.
    x = threshold, y = DPG, ribbon width = uncertainty.
    """
    print("\n[Figure 2] Threshold Stability Ribbon")
    
    if not data or 'models' not in data:
        print("  Skipping: no data")
        return
    
    fig, axes = plt.subplots(1, len(data['models']), figsize=(12, 4))
    
    if len(data['models']) == 1:
        axes = [axes]
    
    for idx, model_data in enumerate(data['models']):
        ax = axes[idx]
        
        thresholds = np.array(model_data['thresholds'])
        dpg_values = np.array(model_data['dpg_values'])
        error_bounds = np.array(model_data['error_bounds'])
        is_reliable = np.array(model_data['is_reliable'])
        
        # Plot DPG line
        ax.plot(thresholds, dpg_values, 'b-', linewidth=2, label='DPG')
        
        # Plot uncertainty ribbon
        ax.fill_between(thresholds, 
                       dpg_values - error_bounds, 
                       dpg_values + error_bounds,
                       alpha=0.3, color='blue', label='Uncertainty')
        
        # Shade unreliable regions
        for i in range(len(thresholds)):
            if not is_reliable[i]:
                ax.axvspan(thresholds[i] - 0.02, thresholds[i] + 0.02, 
                          alpha=0.2, color='red')
        
        ax.set_xlabel('Decision Threshold')
        ax.set_ylabel('Demographic Parity Gap')
        ax.set_title(f'{model_data["model_name"]}'.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0.1, 0.9)
    
    plt.tight_layout()
    save_figure(fig, 'threshold_stability_ribbon', output_dir)


def plot_near_threshold_danger_zone(data: Dict, output_dir: Path):
    """
    Figure 3: Near-threshold danger zone.
    Overlapping density plots of f(x) for G_0, G_1 with threshold line.
    """
    print("\n[Figure 3] Near-Threshold Danger Zone")
    
    if not data or 'models' not in data:
        print("  Skipping: no data")
        return
    
    # Use medium concentration model
    model_data = None
    for m in data['models']:
        if 'medium' in m['config_name']:
            model_data = m
            break
    
    if model_data is None:
        model_data = data['models'][0]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get predictions
    preds_g0 = np.array(model_data['group0']['predictions_raw'])
    preds_g1 = np.array(model_data['group1']['predictions_raw'])
    threshold = model_data['threshold']
    
    # Plot distributions
    ax.hist(preds_g0, bins=50, density=True, alpha=0.5, color='blue', label='Group 0')
    ax.hist(preds_g1, bins=50, density=True, alpha=0.5, color='orange', label='Group 1')
    
    # Threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    # Danger zone (where predictions might flip due to numerical error)
    # Use float32 error estimate
    danger_width = model_data['precision_analysis']['float32']['error_estimate']
    ax.axvspan(threshold - danger_width, threshold + danger_width, 
              alpha=0.2, color='red', label='Danger Zone')
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Near-Threshold Danger Zone: Prediction Distribution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    save_figure(fig, 'near_threshold_danger_zone', output_dir)


def plot_sign_flip_example(data: Dict, output_dir: Path):
    """
    Figure 4: Sign flip example.
    For one case where DPG sign flips, show DPG values with error bars crossing zero.
    """
    print("\n[Figure 4] Sign Flip Example")
    
    if not data or 'sign_flips' not in data:
        print("  Skipping: no data")
        return
    
    # Find a good sign flip example
    sign_flip_cases = [sf for sf in data['sign_flips'] if sf['has_sign_flip']]
    
    if not sign_flip_cases:
        print("  No sign flips found")
        return
    
    # Use first sign flip
    case = sign_flip_cases[0]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    precisions = ['float64', 'float32', 'float16']
    dpg_signed = [case['dpg_signed'][p] for p in precisions]
    error_bounds = [case['error_bounds'][p] for p in precisions]
    
    x_pos = np.arange(len(precisions))
    
    # Plot DPG with error bars
    colors = []
    for dpg, err in zip(dpg_signed, error_bounds):
        # Check if error bar crosses zero
        if (dpg - err) * (dpg + err) <= 0:
            colors.append('red')  # Crosses zero
        else:
            colors.append('green' if abs(dpg) > err else 'orange')
    
    ax.bar(x_pos, dpg_signed, yerr=error_bounds, color=colors, 
          alpha=0.7, capsize=8, edgecolor='black', linewidth=1)
    
    # Zero line
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Precision')
    ax.set_ylabel('Signed Demographic Parity Gap')
    ax.set_title('Sign Flip Example: Fairness Metric Across Precisions')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(precisions)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add text annotation
    ax.text(0.5, 0.95, f'Trial {case["trial"]}: Sign flips between precisions',
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_figure(fig, 'sign_flip_example', output_dir)


def plot_precision_comparison(data: Dict, output_dir: Path):
    """
    Figure 5: Precision comparison summary.
    Shows borderline percentage by precision.
    """
    print("\n[Figure 5] Precision Comparison")
    
    if not data or 'summary' not in data:
        print("  Skipping: no data")
        return
    
    summary = data['summary']['reliable_by_precision']
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    precisions = ['float64', 'float32', 'float16']
    borderline_pcts = [summary[p]['borderline_pct'] for p in precisions]
    
    x_pos = np.arange(len(precisions))
    bars = ax.bar(x_pos, borderline_pcts, color=['green', 'orange', 'red'], 
                 alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, pct) in enumerate(zip(bars, borderline_pcts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Precision')
    ax.set_ylabel('Borderline Assessments (%)')
    ax.set_title('Numerical Reliability by Precision')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(precisions)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    save_figure(fig, 'precision_comparison', output_dir)


def plot_calibration_reliability(data: Dict, output_dir: Path):
    """
    Figure 6: Calibration reliability.
    Shows calibration curves with uncertainty.
    """
    print("\n[Figure 6] Calibration Reliability")
    
    if not data or 'models' not in data:
        print("  Skipping: no data")
        return
    
    # Use first model
    model_data = data['models'][0]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    precisions = ['float64', 'float32', 'float16']
    
    for idx, prec in enumerate(precisions):
        ax = axes[idx]
        
        if prec not in model_data['precision_results']:
            continue
        
        prec_data = model_data['precision_results'][prec]
        
        bin_conf = np.array(prec_data['bin_confidences'])
        bin_acc = np.array(prec_data['bin_accuracies'])
        bin_uncert = np.array(prec_data['bin_uncertainties'])
        reliable_bins = prec_data['reliable_bins']
        
        # Remove empty bins
        valid = bin_conf > 0
        bin_conf = bin_conf[valid]
        bin_acc = bin_acc[valid]
        bin_uncert = bin_uncert[valid]
        reliable_bins_valid = np.array([reliable_bins[i] for i in range(len(reliable_bins)) if valid[i]])
        
        if len(bin_conf) == 0:
            continue
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect Calibration')
        
        # Plot calibration curve with uncertainty
        colors = ['green' if r else 'red' for r in reliable_bins_valid]
        ax.scatter(bin_conf, bin_acc, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Error bars
        ax.errorbar(bin_conf, bin_acc, yerr=bin_uncert, fmt='none', 
                   ecolor='gray', alpha=0.5, capsize=3)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Empirical Accuracy')
        ax.set_title(f'{prec}\nECE: {prec_data["ece"]:.4f}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, 'calibration_reliability', output_dir)


def plot_near_threshold_correlation(data: Dict, output_dir: Path):
    """
    Figure 7: Correlation between near-threshold concentration and unreliability.
    Scatter plot showing p_near vs reliability_score.
    """
    print("\n[Figure 7] Near-Threshold Correlation")
    
    if not data or 'datasets' not in data:
        print("  Skipping: no data")
        return
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Collect data points
    near_threshold_fracs = []
    reliability_scores = []
    colors = []
    labels = []
    
    for dataset in data['datasets']:
        for prec_name in ['float64', 'float32', 'float16']:
            if prec_name in dataset['results']['precisions']:
                prec_data = dataset['results']['precisions'][prec_name]
                
                near_frac = prec_data['near_threshold_fraction']['overall']
                rel_score = prec_data['reliability_score']
                
                # Cap reliability score for plotting
                if rel_score > 100:
                    rel_score = 100
                
                near_threshold_fracs.append(near_frac)
                reliability_scores.append(rel_score)
                
                if prec_name == 'float64':
                    colors.append('blue')
                elif prec_name == 'float32':
                    colors.append('green')
                else:
                    colors.append('red')
                
                labels.append(f"{dataset['name']}-{prec_name}")
    
    # Scatter plot
    ax.scatter(near_threshold_fracs, reliability_scores, c=colors, s=80, alpha=0.7, edgecolors='black')
    
    # Reliability threshold line
    ax.axhline(2.0, color='red', linestyle='--', linewidth=1.5, label='Reliability Threshold')
    
    ax.set_xlabel('Near-Threshold Fraction (p_near)')
    ax.set_ylabel('Reliability Score (DPG / error_bound)')
    ax.set_title('Reliability vs Near-Threshold Concentration')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-5, 105)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='float64'),
        Patch(facecolor='green', alpha=0.7, edgecolor='black', label='float32'),
        Patch(facecolor='red', alpha=0.7, edgecolor='black', label='float16'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Reliability Threshold')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    save_figure(fig, 'near_threshold_correlation', output_dir)


def generate_all_plots(base_dir: Path):
    """Generate all plots from experiment data."""
    print("\n" + "="*70)
    print("GENERATING PLOTS FOR NUMGEOM-FAIR")
    print("="*70)
    
    output_dir = base_dir.parent / 'implementations' / 'docs' / 'proposal25' / 'figures'
    
    # Load data
    exp1_data = load_experiment_data(1, 'experiment1_precision_vs_fairness', base_dir)
    exp2_data = load_experiment_data(2, 'experiment2_near_threshold_distribution', base_dir)
    exp3_data = load_experiment_data(3, 'experiment3_threshold_stability', base_dir)
    exp4_data = load_experiment_data(4, 'experiment4_calibration_reliability', base_dir)
    exp5_data = load_experiment_data(5, 'experiment5_sign_flip_cases', base_dir)
    
    # Generate plots
    plot_fairness_with_error_bars(exp1_data, output_dir)
    plot_threshold_stability_ribbon(exp3_data, output_dir)
    plot_near_threshold_danger_zone(exp2_data, output_dir)
    plot_sign_flip_example(exp5_data, output_dir)
    plot_precision_comparison(exp1_data, output_dir)
    plot_calibration_reliability(exp4_data, output_dir)
    plot_near_threshold_correlation(exp1_data, output_dir)
    
    print("\n" + "="*70)
    print("PLOT GENERATION COMPLETE")
    print("="*70)
    print(f"\nFigures saved to: {output_dir}")
    print("\nGenerated figures:")
    print("  1. fairness_error_bars.png - DPG with certified bounds")
    print("  2. threshold_stability_ribbon.png - Stability across thresholds")
    print("  3. near_threshold_danger_zone.png - Prediction distributions")
    print("  4. sign_flip_example.png - Fairness sign flips")
    print("  5. precision_comparison.png - Borderline % by precision")
    print("  6. calibration_reliability.png - Calibration curves")
    print("  7. near_threshold_correlation.png - p_near vs reliability")


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent / 'data'
    
    if not base_dir.exists():
        print(f"Error: Data directory not found: {base_dir}")
        print("Please run: python3.11 scripts/run_all_experiments.py")
        return
    
    generate_all_plots(base_dir)


if __name__ == '__main__':
    main()
