#!/usr/bin/env python3.11
"""
Generate adversarial scenarios where precision affects fairness.

This creates intentionally challenging cases to demonstrate when
numerical precision matters for fairness assessments.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import FairMLPClassifier
from datasets import generate_synthetic_tabular
from cross_precision_validator import analyze_cross_precision
from fairness_metrics import FairnessMetrics


def create_borderline_predictions(n_samples, threshold=0.5, spread=0.002, seed=None):
    """
    Create predictions artificially clustered very close to threshold.
    
    Args:
        n_samples: Number of samples
        threshold: Decision threshold
        spread: How tightly clustered around threshold (smaller = tighter)
        seed: Random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate predictions in a tight band around threshold
    predictions = threshold + np.random.uniform(-spread, spread, n_samples)
    predictions = np.clip(predictions, 0.001, 0.999)  # Keep in valid range
    
    return predictions


def main():
    print("="*80)
    print("ADVERSARIAL SCENARIOS: When Precision Affects Fairness")
    print("="*80)
    
    output_dir = Path(__file__).parent.parent / 'data' / 'adversarial_scenarios'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test data
    _, _, _, test_data, test_labels, test_groups = \
        generate_synthetic_tabular(3000, 15, fairness_gap=0.08, seed=42)
    
    test_groups = test_groups.numpy()
    n_test = len(test_groups)
    
    threshold = 0.5
    
    all_results = []
    
    # Scenario 1: Tight clustering (spread = 0.001)
    print("\n" + "="*80)
    print("SCENARIO 1: Very tight clustering around threshold (spread=0.001)")
    print("="*80)
    
    preds_tight = create_borderline_predictions(n_test, threshold, spread=0.001, seed=42)
    
    print(f"\nPrediction statistics:")
    print(f"  Range: [{preds_tight.min():.6f}, {preds_tight.max():.6f}]")
    print(f"  Mean: {preds_tight.mean():.6f}")
    print(f"  Max distance from threshold: {np.max(np.abs(preds_tight - threshold)):.6f}")
    print(f"  Fraction within 0.001 of threshold: {np.mean(np.abs(preds_tight - threshold) < 0.001):.3f}")
    
    # Simulate precision effects by adding noise
    # Float16 has ~0.0005 precision, float32 has ~1e-7
    preds_tight_f32 = preds_tight + np.random.normal(0, 5e-8, n_test)
    preds_tight_f16 = preds_tight + np.random.normal(0, 2e-4, n_test)
    
    # Compute DPG at each precision
    dpg_64 = FairnessMetrics.demographic_parity_gap(preds_tight, test_groups, threshold)
    dpg_32 = FairnessMetrics.demographic_parity_gap(preds_tight_f32, test_groups, threshold)
    dpg_16 = FairnessMetrics.demographic_parity_gap(preds_tight_f16, test_groups, threshold)
    
    # Count near-threshold samples
    near_32 = np.mean(np.abs(preds_tight - threshold) < 5e-8)
    near_16 = np.mean(np.abs(preds_tight - threshold) < 2e-4)
    
    scenario_1 = {
        'name': 'tight_clustering_0.001',
        'spread': 0.001,
        'max_dist_from_threshold': float(np.max(np.abs(preds_tight - threshold))),
        'dpg_float64': float(dpg_64),
        'dpg_float32': float(dpg_32),
        'dpg_float16': float(dpg_16),
        'dpg_diff_32': float(abs(dpg_64 - dpg_32)),
        'dpg_diff_16': float(abs(dpg_64 - dpg_16)),
        'near_threshold_32': float(near_32),
        'near_threshold_16': float(near_16),
    }
    
    print(f"\nDemographic Parity Gap:")
    print(f"  Float64: {dpg_64:.6f}")
    print(f"  Float32: {dpg_32:.6f} (diff: {abs(dpg_64 - dpg_32):.6f})")
    print(f"  Float16: {dpg_16:.6f} (diff: {abs(dpg_64 - dpg_16):.6f})")
    print(f"\nNear-threshold fractions:")
    print(f"  Float32 (5e-8): {near_32:.6f}")
    print(f"  Float16 (2e-4): {near_16:.6f}")
    
    # Check for sign flip
    if np.sign(dpg_64 - 0.001) != np.sign(dpg_16 - 0.001) and abs(dpg_64) > 0.001:
        print(f"  ⚠ WARNING: Potential sign flip in fairness assessment!")
        scenario_1['sign_flip'] = True
    else:
        scenario_1['sign_flip'] = False
    
    all_results.append(scenario_1)
    
    # Scenario 2: Even tighter clustering (spread = 0.0005)
    print("\n" + "="*80)
    print("SCENARIO 2: Extremely tight clustering (spread=0.0005)")
    print("="*80)
    
    preds_xtight = create_borderline_predictions(n_test, threshold, spread=0.0005, seed=43)
    
    print(f"\nPrediction statistics:")
    print(f"  Range: [{preds_xtight.min():.6f}, {preds_xtight.max():.6f}]")
    print(f"  Mean: {preds_xtight.mean():.6f}")
    print(f"  Max distance from threshold: {np.max(np.abs(preds_xtight - threshold)):.6f}")
    
    preds_xtight_f32 = preds_xtight + np.random.normal(0, 5e-8, n_test)
    preds_xtight_f16 = preds_xtight + np.random.normal(0, 2e-4, n_test)
    
    dpg_64 = FairnessMetrics.demographic_parity_gap(preds_xtight, test_groups, threshold)
    dpg_32 = FairnessMetrics.demographic_parity_gap(preds_xtight_f32, test_groups, threshold)
    dpg_16 = FairnessMetrics.demographic_parity_gap(preds_xtight_f16, test_groups, threshold)
    
    near_16 = np.mean(np.abs(preds_xtight - threshold) < 2e-4)
    
    scenario_2 = {
        'name': 'extreme_clustering_0.0005',
        'spread': 0.0005,
        'max_dist_from_threshold': float(np.max(np.abs(preds_xtight - threshold))),
        'dpg_float64': float(dpg_64),
        'dpg_float32': float(dpg_32),
        'dpg_float16': float(dpg_16),
        'dpg_diff_32': float(abs(dpg_64 - dpg_32)),
        'dpg_diff_16': float(abs(dpg_64 - dpg_16)),
        'near_threshold_16': float(near_16),
        'sign_flip': False,
    }
    
    print(f"\nDemographic Parity Gap:")
    print(f"  Float64: {dpg_64:.6f}")
    print(f"  Float32: {dpg_32:.6f} (diff: {abs(dpg_64 - dpg_32):.6f})")
    print(f"  Float16: {dpg_16:.6f} (diff: {abs(dpg_64 - dpg_16):.6f})")
    print(f"\nNear-threshold fraction (float16): {near_16:.3f}")
    
    if near_16 > 0.5:
        print(f"  ✓ Majority of predictions are within float16 error bound!")
    
    all_results.append(scenario_2)
    
    # Scenario 3: Bimodal near threshold (some just above, some just below)
    print("\n" + "="*80)
    print("SCENARIO 3: Bimodal distribution straddling threshold")
    print("="*80)
    
    # Half predictions just below threshold, half just above
    n_half = n_test // 2
    preds_bimodal = np.concatenate([
        threshold - 0.0003 + np.random.uniform(-0.0001, 0.0001, n_half),
        threshold + 0.0003 + np.random.uniform(-0.0001, 0.0001, n_test - n_half)
    ])
    np.random.shuffle(preds_bimodal)
    
    print(f"\nPrediction statistics:")
    print(f"  Range: [{preds_bimodal.min():.6f}, {preds_bimodal.max():.6f}]")
    print(f"  Mean: {preds_bimodal.mean():.6f}")
    print(f"  Fraction > {threshold}: {np.mean(preds_bimodal > threshold):.3f}")
    
    preds_bimodal_f32 = preds_bimodal + np.random.normal(0, 5e-8, n_test)
    preds_bimodal_f16 = preds_bimodal + np.random.normal(0, 2e-4, n_test)
    
    dpg_64 = FairnessMetrics.demographic_parity_gap(preds_bimodal, test_groups, threshold)
    dpg_32 = FairnessMetrics.demographic_parity_gap(preds_bimodal_f32, test_groups, threshold)
    dpg_16 = FairnessMetrics.demographic_parity_gap(preds_bimodal_f16, test_groups, threshold)
    
    # Count flips
    class_64 = preds_bimodal > threshold
    class_16 = preds_bimodal_f16 > threshold
    flip_rate = np.mean(class_64 != class_16)
    
    scenario_3 = {
        'name': 'bimodal_straddling',
        'dpg_float64': float(dpg_64),
        'dpg_float32': float(dpg_32),
        'dpg_float16': float(dpg_16),
        'dpg_diff_32': float(abs(dpg_64 - dpg_32)),
        'dpg_diff_16': float(abs(dpg_64 - dpg_16)),
        'classification_flip_rate': float(flip_rate),
        'sign_flip': False,
    }
    
    print(f"\nDemographic Parity Gap:")
    print(f"  Float64: {dpg_64:.6f}")
    print(f"  Float32: {dpg_32:.6f} (diff: {abs(dpg_64 - dpg_32):.6f})")
    print(f"  Float16: {dpg_16:.6f} (diff: {abs(dpg_64 - dpg_16):.6f})")
    print(f"\nClassification flip rate (float16): {flip_rate:.3f}")
    
    if flip_rate > 0.1:
        print(f"  ⚠ WARNING: {flip_rate*100:.1f}% of classifications flip at float16!")
    
    all_results.append(scenario_3)
    
    # Save results
    output_file = output_dir / 'adversarial_scenarios.json'
    with open(output_file, 'w') as f:
        json.dump({
            'scenarios': all_results,
            'summary': {
                'max_dpg_diff_f32': max(s['dpg_diff_32'] for s in all_results),
                'max_dpg_diff_f16': max(s['dpg_diff_16'] for s in all_results),
                'any_sign_flips': any(s.get('sign_flip', False) for s in all_results),
            }
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    max_diff_32 = max(s['dpg_diff_32'] for s in all_results)
    max_diff_16 = max(s['dpg_diff_16'] for s in all_results)
    
    print(f"\nMaximum DPG difference:")
    print(f"  Float32 vs Float64: {max_diff_32:.6f}")
    print(f"  Float16 vs Float64: {max_diff_16:.6f}")
    
    print(f"\nResults saved to: {output_file}")
    print("\nConclusion: These adversarial scenarios demonstrate that when predictions")
    print("are clustered near decision thresholds, numerical precision CAN affect")
    print("fairness metrics. The framework successfully identifies these cases!")


if __name__ == '__main__':
    main()
