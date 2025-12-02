#!/usr/bin/env python3.11
"""
Quick demonstration of NumGeom-Fair capabilities.

Shows:
1. Training a borderline-fair model
2. Evaluating fairness with certified bounds
3. Identifying numerically stable thresholds
4. Comparing across precisions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np

from error_propagation import ErrorTracker, LinearErrorFunctional
from fairness_metrics import CertifiedFairnessEvaluator, ThresholdStabilityAnalyzer
from models import FairMLPClassifier, train_fair_classifier
from datasets import generate_synthetic_tabular

def main():
    print("\n" + "="*70)
    print("NUMGEOM-FAIR: QUICK DEMONSTRATION")
    print("="*70)
    
    # Setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    print("\n[Step 1] Generating synthetic dataset...")
    train_data, train_labels, train_groups, test_data, test_labels, test_groups = \
        generate_synthetic_tabular(n_samples=2000, n_features=10, fairness_gap=0.06, seed=42)
    print(f"  Train: {len(train_data)} samples, {train_data.shape[1]} features")
    print(f"  Test:  {len(test_data)} samples")
    print(f"  Groups: {(train_groups == 0).sum()} (group 0), {(train_groups == 1).sum()} (group 1)")
    
    print("\n[Step 2] Training borderline-fair model...")
    model = FairMLPClassifier(
        input_dim=10,
        hidden_dims=[32, 16],
        activation='relu'
    ).to(device)
    
    history = train_fair_classifier(
        model, train_data.to(device), train_labels.to(device), train_groups.to(device),
        n_epochs=60, lr=0.001, fairness_weight=0.02, device=device, verbose=False
    )
    
    final_acc = history[-1]['accuracy']
    final_dpg = history[-1]['dpg']
    print(f"  Final accuracy: {final_acc:.4f}")
    print(f"  Final DPG: {final_dpg:.4f}")
    
    print("\n[Step 3] Evaluating with certified bounds...")
    
    # Get model device
    model_device = next(model.parameters()).device
    
    for precision in [torch.float64, torch.float32, torch.float16]:
        tracker = ErrorTracker(precision)
        evaluator = CertifiedFairnessEvaluator(tracker, reliability_threshold=2.0)
        
        # Convert model
        model_copy = FairMLPClassifier(
            input_dim=10,
            hidden_dims=[32, 16],
            activation='relu'
        )
        model_copy.load_state_dict(model.state_dict())
        model_copy = model_copy.to(precision)
        
        # Handle device (MPS doesn't support float64)
        if precision == torch.float64 and model_device.type == 'mps':
            model_copy = model_copy.cpu()
            test_data_device = test_data.to(precision).cpu()
        else:
            model_copy = model_copy.to(model_device)
            test_data_device = test_data.to(precision).to(model_device)
        
        # Get error functional
        layer_dims = model_copy.get_layer_dims()
        activations = model_copy.get_activations()
        error_functional = tracker.track_network(layer_dims, activations)
        
        # Evaluate
        result = evaluator.evaluate_demographic_parity(
            model_copy, test_data_device, test_groups.numpy(),
            threshold=0.5,
            model_error_functional=error_functional
        )
        
        precision_name = tracker.get_precision_name()
        reliable_str = "✓ RELIABLE" if result.is_reliable else "✗ BORDERLINE"
        print(f"\n  {precision_name}:")
        print(f"    DPG: {result.metric_value:.6f} ± {result.error_bound:.6f}")
        print(f"    Reliability score: {result.reliability_score:.2f}")
        print(f"    Status: {reliable_str}")
        print(f"    Near-threshold samples: {result.near_threshold_fraction['overall']:.2%}")
    
    print("\n[Step 4] Analyzing threshold stability...")
    tracker = ErrorTracker(torch.float32)
    evaluator = CertifiedFairnessEvaluator(tracker)
    analyzer = ThresholdStabilityAnalyzer(evaluator)
    
    # Use float32 model on appropriate device
    model_f32 = FairMLPClassifier(
        input_dim=10,
        hidden_dims=[32, 16],
        activation='relu'
    )
    model_f32.load_state_dict(model.state_dict())
    model_f32 = model_f32.to(torch.float32).to(model_device)
    test_data_f32 = test_data.to(torch.float32).to(model_device)
    
    layer_dims = model_f32.get_layer_dims()
    activations = model_f32.get_activations()
    error_functional = tracker.track_network(layer_dims, activations)
    
    stable_regions = analyzer.find_stable_thresholds(
        model_f32, test_data_f32, test_groups.numpy(),
        model_error_functional=error_functional
    )
    
    print(f"\n  Found {len(stable_regions)} numerically stable threshold region(s):")
    for i, (t_min, t_max) in enumerate(stable_regions):
        print(f"    Region {i+1}: [{t_min:.3f}, {t_max:.3f}]")
    
    print("\n[Step 5] Key insights...")
    print("  • Float64 provides reliable fairness assessments")
    print("  • Float32/Float16 can make fairness metrics numerically uncertain")
    print("  • Near-threshold sample concentration predicts unreliability")
    print("  • Certain threshold ranges are more numerically stable")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nFor full experiments:")
    print("  python3.11 scripts/run_all_experiments.py")
    print("\nFor visualizations:")
    print("  python3.11 scripts/generate_plots.py")
    print("\nFor paper:")
    print("  cd implementations/docs/proposal25 && make")

if __name__ == '__main__':
    main()
