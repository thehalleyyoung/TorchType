#!/usr/bin/env python3.11
"""
Comprehensive cross-precision fairness validation.

This script tests whether our theoretical error bounds actually hold
when models are run at different precisions.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import FairMLPClassifier, train_fair_classifier
from datasets import generate_synthetic_tabular
from cross_precision_validator import validate_error_bounds, analyze_cross_precision
from fairness_metrics import FairnessMetrics


def create_near_threshold_model(input_dim, test_data, threshold=0.5):
    """
    Create a model whose predictions are intentionally near the threshold.
    This stress-tests the numerical precision effects.
    """
    # Create a simple model
    model = FairMLPClassifier(input_dim=input_dim, hidden_dims=[32, 16]).to(torch.float64)
    
    # Manually adjust weights to push predictions toward threshold
    with torch.no_grad():
        # Make the final layer output close to logit(threshold)
        # For sigmoid, logit(0.5) = 0
        for param in model.parameters():
            param.data *= 0.1  # Scale down weights
    
    return model


def main():
    print("="*80)
    print("COMPREHENSIVE CROSS-PRECISION FAIRNESS VALIDATION")
    print("="*80)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'cross_precision_validation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Test 1: Normal trained model
    print("\n" + "="*80)
    print("TEST 1: Normally Trained Model")
    print("="*80)
    
    train_data, train_labels, train_groups, test_data, test_labels, test_groups = \
        generate_synthetic_tabular(3000, 15, fairness_gap=0.08, seed=42)
    
    print("\nTraining model...")
    model = FairMLPClassifier(input_dim=15, hidden_dims=[64, 32]).to(torch.float64).cpu()
    history = train_fair_classifier(
        model, train_data.to(torch.float64).cpu(), 
        train_labels.to(torch.float64).cpu(),
        train_groups.to(torch.float64).cpu(), 
        n_epochs=80, lr=0.001, 
        fairness_weight=0.01, device='cpu', verbose=False
    )
    
    # Get prediction distribution
    model.eval()
    with torch.no_grad():
        preds_64 = model(test_data.to(torch.float64).cpu()).cpu().numpy().flatten()
    
    print(f"\nPrediction statistics:")
    print(f"  Range: [{preds_64.min():.4f}, {preds_64.max():.4f}]")
    print(f"  Mean: {preds_64.mean():.4f}, Std: {preds_64.std():.4f}")
    print(f"  Median: {np.median(preds_64):.4f}")
    
    print("\nValidating cross-precision bounds...")
    results_normal = validate_error_bounds(
        model, test_data, test_groups.numpy(), threshold=0.5, device='cpu'
    )
    
    print("\nResults:")
    for precision in ['float32', 'float16']:
        r = results_normal[precision]
        print(f"\n{precision}:")
        print(f"  Max prediction diff: {r['max_pred_diff']:.8f}")
        print(f"  Mean prediction diff: {r['mean_pred_diff']:.8f}")
        print(f"  DPG diff: {r['dpg_diff']:.8f}")
        print(f"  Near-threshold fraction: {r['near_threshold_fraction']:.4f}")
        print(f"  Theoretical bound: {r['theoretical_dpg_bound']:.8f}")
        
        # Check if bound holds
        bound_holds = r['dpg_diff'] <= r['theoretical_dpg_bound'] + 1e-6
        print(f"  Bound holds: {'✓ YES' if bound_holds else '✗ NO'}")
    
    all_results['normal_model'] = results_normal
    
    # Test 2: Model with predictions near threshold
    print("\n" + "="*80)
    print("TEST 2: Model with Predictions Near Threshold")
    print("="*80)
    
    # Train with less fairness regularization and fewer epochs
    model2 = FairMLPClassifier(input_dim=15, hidden_dims=[32, 16]).to(torch.float64).cpu()
    history2 = train_fair_classifier(
        model2, train_data.to(torch.float64).cpu(), 
        train_labels.to(torch.float64).cpu(),
        train_groups.to(torch.float64).cpu(), 
        n_epochs=20, lr=0.005,  # Higher LR, fewer epochs
        fairness_weight=0.001,  # Less fairness regularization
        device='cpu', verbose=False
    )
    
    # Get prediction distribution
    model2.eval()
    with torch.no_grad():
        preds2_64 = model2(test_data.to(torch.float64).cpu()).cpu().numpy().flatten()
    
    print(f"\nPrediction statistics:")
    print(f"  Range: [{preds2_64.min():.4f}, {preds2_64.max():.4f}]")
    print(f"  Mean: {preds2_64.mean():.4f}, Std: {preds2_64.std():.4f}")
    print(f"  Median: {np.median(preds2_64):.4f}")
    print(f"  Fraction in [0.45, 0.55]: {np.mean((preds2_64 >= 0.45) & (preds2_64 <= 0.55)):.3f}")
    
    print("\nValidating cross-precision bounds...")
    results_near = validate_error_bounds(
        model2, test_data, test_groups.numpy(), threshold=0.5, device='cpu'
    )
    
    print("\nResults:")
    for precision in ['float32', 'float16']:
        r = results_near[precision]
        print(f"\n{precision}:")
        print(f"  Max prediction diff: {r['max_pred_diff']:.8f}")
        print(f"  Mean prediction diff: {r['mean_pred_diff']:.8f}")
        print(f"  DPG diff: {r['dpg_diff']:.8f}")
        print(f"  Near-threshold fraction: {r['near_threshold_fraction']:.4f}")
        print(f"  Theoretical bound: {r['theoretical_dpg_bound']:.8f}")
        
        # Check if bound holds
        bound_holds = r['dpg_diff'] <= r['theoretical_dpg_bound'] + 1e-6
        print(f"  Bound holds: {'✓ YES' if bound_holds else '✗ NO'}")
    
    all_results['near_threshold_model'] = results_near
    
    # Test 3: Manually created model with very tight predictions
    print("\n" + "="*80)
    print("TEST 3: Manually Calibrated Model (Predictions Extremely Near 0.5)")
    print("="*80)
    
    model3 = create_near_threshold_model(15, test_data, threshold=0.5)
    
    # Get prediction distribution
    model3.eval()
    with torch.no_grad():
        preds3_64 = model3(test_data.to(torch.float64).cpu()).cpu().numpy().flatten()
    
    print(f"\nPrediction statistics:")
    print(f"  Range: [{preds3_64.min():.4f}, {preds3_64.max():.4f}]")
    print(f"  Mean: {preds3_64.mean():.4f}, Std: {preds3_64.std():.4f}")
    print(f"  Distance from 0.5: max={np.max(np.abs(preds3_64 - 0.5)):.6f}")
    
    print("\nValidating cross-precision bounds...")
    results_tight = validate_error_bounds(
        model3, test_data, test_groups.numpy(), threshold=0.5, device='cpu'
    )
    
    print("\nResults:")
    for precision in ['float32', 'float16']:
        r = results_tight[precision]
        print(f"\n{precision}:")
        print(f"  Max prediction diff: {r['max_pred_diff']:.8f}")
        print(f"  Mean prediction diff: {r['mean_pred_diff']:.8f}")
        print(f"  DPG diff: {r['dpg_diff']:.8f}")
        print(f"  Near-threshold fraction: {r['near_threshold_fraction']:.4f}")
        print(f"  Theoretical bound: {r['theoretical_dpg_bound']:.8f}")
        
        # Check if bound holds
        bound_holds = r['dpg_diff'] <= r['theoretical_dpg_bound'] + 1e-6
        print(f"  Bound holds: {'✓ YES' if bound_holds else '✗ NO'}")
        
        # This model should have high near-threshold fraction!
        if r['near_threshold_fraction'] > 0.1:
            print(f"  ✓ Successfully created near-threshold scenario!")
    
    all_results['tight_threshold_model'] = results_tight
    
    # Save results
    output_file = output_dir / 'validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Count how many bounds hold
    total_tests = 0
    bounds_held = 0
    
    for model_type, results in all_results.items():
        for precision in ['float32', 'float16']:
            total_tests += 1
            r = results[precision]
            if r['dpg_diff'] <= r['theoretical_dpg_bound'] + 1e-6:
                bounds_held += 1
    
    print(f"\nBounds validation: {bounds_held}/{total_tests} tests passed")
    print(f"Success rate: {100 * bounds_held / total_tests:.1f}%")
    
    print(f"\nResults saved to: {output_file}")
    
    # Identify the most interesting case (highest near-threshold fraction)
    max_near_threshold = 0
    best_case = None
    
    for model_type, results in all_results.items():
        for precision in ['float32', 'float16']:
            r = results[precision]
            if r['near_threshold_fraction'] > max_near_threshold:
                max_near_threshold = r['near_threshold_fraction']
                best_case = (model_type, precision, r)
    
    if best_case:
        model_type, precision, r = best_case
        print(f"\nMost challenging case: {model_type} @ {precision}")
        print(f"  Near-threshold fraction: {r['near_threshold_fraction']:.4f}")
        print(f"  DPG difference: {r['dpg_diff']:.6f}")
        print(f"  Theoretical bound: {r['theoretical_dpg_bound']:.6f}")
        print(f"  Max prediction diff: {r['max_pred_diff']:.6f}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
