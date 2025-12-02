#!/usr/bin/env python3.11
"""
Generate plot for adversarial sign flip experiment.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'figure.figsize': (7, 4),
    'figure.dpi': 300,
})


def plot_adversarial_sign_flips():
    """Plot adversarial sign flip results."""
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'experiment5' / 'adversarial_sign_flips.json'
    with open(data_path) as f:
        data = json.load(f)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left plot: Sign flip rates by scenario
    scenarios = data['scenarios']
    scenario_names = [s['name'].replace('_', ' ').title() for s in scenarios]
    flip_rates = [s['sign_flip_rate'] * 100 for s in scenarios]
    
    bars = ax1.bar(range(len(scenario_names)), flip_rates, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Concentration Scenario')
    ax1.set_ylabel('Sign Flip Rate (%)')
    ax1.set_title('Sign Flip Rates Under Numerical Perturbation')
    ax1.set_xticks(range(len(scenario_names)))
    ax1.set_xticklabels(scenario_names, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=17.5, color='red', linestyle='--', linewidth=1, label='Overall Average (17.5%)')
    ax1.legend()
    
    # Right plot: Example sign flip case
    # Find a trial with a sign flip
    sign_flip_example = None
    for scenario in scenarios:
        for trial in scenario['trials']:
            if trial['has_sign_flip']:
                sign_flip_example = trial
                break
        if sign_flip_example:
            break
    
    if sign_flip_example:
        precisions = ['float64', 'float32', 'float16']
        dpg_values = [sign_flip_example['dpg_by_precision'][p] for p in precisions]
        colors_sign = ['green' if v > 0 else 'red' for v in dpg_values]
        
        bars2 = ax2.bar(precisions, dpg_values, color=colors_sign, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Precision')
        ax2.set_ylabel('DPG (Signed)')
        ax2.set_title('Example Sign Flip Across Precisions')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, dpg_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:+.4f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9)
        
        # Indicate which group is favored
        ax2.text(0.95, 0.95, 'Group 0 favored', transform=ax2.transAxes,
                ha='right', va='top', color='green', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.text(0.95, 0.05, 'Group 1 favored', transform=ax2.transAxes,
                ha='right', va='bottom', color='red', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No sign flips detected', ha='center', va='center',
                transform=ax2.transAxes)
        ax2.set_title('Example Sign Flip (None Found)')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent.parent / 'implementations' / 'docs' / 'proposal25' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = output_dir / 'adversarial_sign_flips.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {png_path}")
    
    try:
        pgf_path = output_dir / 'adversarial_sign_flips.pgf'
        fig.savefig(pgf_path, format='pgf', bbox_inches='tight')
        print(f"Saved: {pgf_path}")
    except:
        pass
    
    plt.close()


if __name__ == '__main__':
    plot_adversarial_sign_flips()
    print("\nAdversarial sign flip plot generated successfully!")
