#!/usr/bin/env python3.11
"""
Generate visualization comparing theoretical bounds vs empirical cross-precision effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def plot_adversarial_dpg_comparison():
    """
    Plot DPG across precisions for adversarial scenarios.
    """
    # Load adversarial scenarios data
    data_file = Path(__file__).parent.parent / 'data' / 'adversarial_scenarios' / 'adversarial_scenarios.json'
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        print("Run scripts/generate_adversarial_scenarios.py first")
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    scenarios = data['scenarios']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: DPG across precisions
    scenario_names = [s['name'].replace('_', ' ').title() for s in scenarios]
    x = np.arange(len(scenarios))
    width = 0.25
    
    dpg_64 = [s['dpg_float64'] for s in scenarios]
    dpg_32 = [s['dpg_float32'] for s in scenarios]
    dpg_16 = [s['dpg_float16'] for s in scenarios]
    
    ax1.bar(x - width, dpg_64, width, label='Float64', color='#2E7D32', alpha=0.8)
    ax1.bar(x, dpg_32, width, label='Float32', color='#1976D2', alpha=0.8)
    ax1.bar(x + width, dpg_16, width, label='Float16', color='#D32F2F', alpha=0.8)
    
    ax1.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Demographic Parity Gap', fontsize=12, fontweight='bold')
    ax1.set_title('DPG Across Precisions\n(Adversarial Scenarios)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=9)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Plot 2: DPG differences
    dpg_diff_32 = [s['dpg_diff_32'] for s in scenarios]
    dpg_diff_16 = [s['dpg_diff_16'] for s in scenarios]
    
    ax2.bar(x - width/2, dpg_diff_32, width, label='Float32 - Float64', color='#1976D2', alpha=0.8)
    ax2.bar(x + width/2, dpg_diff_16, width, label='Float16 - Float64', color='#D32F2F', alpha=0.8)
    
    ax2.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax2.set_ylabel('|DPG Difference|', fontsize=12, fontweight='bold')
    ax2.set_title('Precision Effect on DPG\n(Higher = More Numerical Sensitivity)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=9)
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent.parent / 'docs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'adversarial_dpg_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_near_threshold_effects():
    """
    Plot showing how near-threshold concentration affects fairness volatility.
    """
    data_file = Path(__file__).parent.parent / 'data' / 'adversarial_scenarios' / 'adversarial_scenarios.json'
    
    if not data_file.exists():
        return
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    scenarios = data['scenarios']
    
    # Extract data
    scenario_names = [s['name'].replace('_', '\n').replace(' 0.', '\n(spread=0.') + ')' 
                      if 'clustering' in s['name'] else s['name'].replace('_', '\n')
                      for s in scenarios]
    
    # Near-threshold fractions for float16
    near_threshold = [s.get('near_threshold_16', 0) for s in scenarios]
    dpg_diff_16 = [s['dpg_diff_16'] for s in scenarios]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B' if nt > 0.2 else '#4ECDC4' for nt in near_threshold]
    
    bars = ax.bar(range(len(scenarios)), near_threshold, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add DPG diff as text on bars
    for i, (bar, dpg_diff) in enumerate(zip(bars, dpg_diff_16)):
        height = bar.get_height()
        if height > 0.05:  # Only label if significant
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'ΔDPG={dpg_diff:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Scenario', fontsize=13, fontweight='bold')
    ax.set_ylabel('Near-Threshold Fraction', fontsize=13, fontweight='bold')
    ax.set_title('Concentration Near Decision Threshold\n(Float16 Error Bound)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenario_names, rotation=0, ha='center', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(near_threshold) * 1.15 if near_threshold else 1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', edgecolor='black', label='Low volatility (<20%)'),
        Patch(facecolor='#FF6B6B', edgecolor='black', label='High volatility (>20%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / 'docs' / 'figures'
    output_file = output_dir / 'near_threshold_concentration.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_precision_recommendation():
    """
    Decision tree plot for precision recommendation.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create flowchart-style visualization
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    # Define boxes
    boxes = {
        'start': {'pos': (0.5, 0.9), 'text': 'Evaluate Fairness\nMetric', 'color': '#E3F2FD'},
        'compute_bound': {'pos': (0.5, 0.75), 'text': 'Compute\nError Bound δ', 'color': '#FFF9C4'},
        'check_dpg': {'pos': (0.5, 0.6), 'text': 'DPG > 2·δ?', 'color': '#FFE0B2'},
        'reliable': {'pos': (0.25, 0.4), 'text': '✓ Reliable\nUse float32', 'color': '#C8E6C9'},
        'borderline': {'pos': (0.75, 0.4), 'text': '⚠ Borderline\nUse float64', 'color': '#FFCDD2'},
        'check_threshold': {'pos': (0.5, 0.2), 'text': 'Adjust threshold\nor retrain?', 'color': '#F3E5F5'},
    }
    
    # Draw boxes
    for key, box in boxes.items():
        x, y = box['pos']
        fancy_box = FancyBboxPatch(
            (x - 0.1, y - 0.05), 0.2, 0.1,
            boxstyle="round,pad=0.01",
            edgecolor='black', facecolor=box['color'],
            linewidth=2
        )
        ax.add_patch(fancy_box)
        ax.text(x, y, box['text'], ha='center', va='center',
               fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        ('start', 'compute_bound'),
        ('compute_bound', 'check_dpg'),
        ('check_dpg', 'reliable', 'YES'),
        ('check_dpg', 'borderline', 'NO'),
        ('borderline', 'check_threshold'),
    ]
    
    for arrow in arrows:
        if len(arrow) == 2:
            src, dst = arrow
            label = ''
        else:
            src, dst, label = arrow
        
        x1, y1 = boxes[src]['pos']
        x2, y2 = boxes[dst]['pos']
        
        arrow_patch = FancyArrowPatch(
            (x1, y1 - 0.05), (x2, y2 + 0.05),
            arrowstyle='->', mutation_scale=20,
            linewidth=2, color='black'
        )
        ax.add_patch(arrow_patch)
        
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.05, mid_y, label,
                   fontsize=9, fontweight='bold', color='#D32F2F')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('NumGeom-Fair Decision Framework', fontsize=16, fontweight='bold', pad=20)
    
    # Add formula box
    formula_text = (
        'Error Bound: δ = p_near⁽⁰⁾ + p_near⁽¹⁾\n'
        'where p_near⁽ⁱ⁾ = fraction of group i\n'
        'within error functional of threshold'
    )
    ax.text(0.5, 0.05, formula_text,
           ha='center', va='bottom',
           fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / 'docs' / 'figures'
    output_file = output_dir / 'precision_recommendation_flowchart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    print("="*80)
    print("GENERATING ENHANCED VISUALIZATIONS")
    print("="*80)
    
    print("\n1. Adversarial DPG comparison...")
    plot_adversarial_dpg_comparison()
    
    print("\n2. Near-threshold concentration effects...")
    plot_near_threshold_effects()
    
    print("\n3. Precision recommendation flowchart...")
    plot_precision_recommendation()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
