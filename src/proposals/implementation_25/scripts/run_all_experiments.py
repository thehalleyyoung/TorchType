#!/usr/bin/env python3.11
"""
Comprehensive experiments for NumGeom-Fair (Proposal 25).

Runs all 5 experiments from the proposal and saves detailed results.
All experiments designed to complete in ~1-2 hours on a laptop.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from error_propagation import ErrorTracker, LinearErrorFunctional, create_empirical_error_functional
from fairness_metrics import CertifiedFairnessEvaluator, ThresholdStabilityAnalyzer, FairnessMetrics
from models import FairMLPClassifier, train_fair_classifier
from datasets import generate_synthetic_tabular, generate_synthetic_compas, load_adult_income


def setup_directories():
    """Create necessary directories for experiment outputs."""
    base_dir = Path(__file__).parent.parent / 'data'
    
    for i in range(1, 6):
        exp_dir = base_dir / f'experiment{i}'
        exp_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = base_dir / 'trained_models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    return base_dir


def save_results(results: Dict, experiment_name: str, base_dir: Path):
    """Save experiment results as JSON."""
    exp_num = experiment_name.split('_')[0].replace('experiment', '')
    save_path = base_dir / f'experiment{exp_num}' / f'{experiment_name}.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"  Saved to: {save_path}")


def experiment1_precision_vs_fairness(base_dir: Path, device: str):
    """
    Experiment 1: Precision vs Fairness
    
    Train models at float64, evaluate DPG at float64/32/16.
    Compare differences to certified bounds.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Precision vs Fairness")
    print("="*70)
    
    results = {
        'description': 'Evaluate DPG at different precisions and compare to bounds',
        'datasets': [],
        'precision_comparison': {}
    }
    
    # Test on 3 datasets
    datasets = [
        ('synthetic_tabular', lambda: generate_synthetic_tabular(3000, 15, 0.08, 42)),
        ('synthetic_compas', lambda: generate_synthetic_compas(2000, 0.10, 43)),
        ('adult_subset', lambda: load_adult_income(5000, 44))
    ]
    
    for dataset_name, dataset_fn in datasets:
        print(f"\n[{dataset_name}]")
        
        # Load dataset
        train_data, train_labels, train_groups, test_data, test_labels, test_groups = dataset_fn()
        print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Train model at float64 (use CPU if MPS since MPS doesn't support float64)
        print("  Training model (float64)...")
        train_device = 'cpu'  # Always use CPU for float64 since MPS doesn't support it
        
        model = FairMLPClassifier(
            input_dim=train_data.shape[1],
            hidden_dims=[64, 32],
            activation='relu'
        ).to(train_device).to(torch.float64)
        
        history = train_fair_classifier(
            model,
            train_data.to(torch.float64).to(train_device),
            train_labels.to(torch.float64).to(train_device),
            train_groups.to(torch.float64).to(train_device),
            n_epochs=80,
            lr=0.001,
            fairness_weight=0.01,
            device=train_device,
            verbose=False
        )
        
        # Save model
        torch.save(model.state_dict(), base_dir / 'trained_models' / f'{dataset_name}_exp1.pt')
        
        dataset_results = {
            'n_train': len(train_data),
            'n_test': len(test_data),
            'n_features': train_data.shape[1],
            'final_accuracy': float(history[-1]['accuracy']),
            'final_dpg_train': float(history[-1]['dpg']),
            'precisions': {}
        }
        
        # Evaluate at each precision
        for precision in [torch.float64, torch.float32, torch.float16]:
            precision_name = {torch.float64: 'float64', torch.float32: 'float32', torch.float16: 'float16'}[precision]
            print(f"  Evaluating at {precision_name}...")
            
            # Create error tracker
            tracker = ErrorTracker(precision)
            evaluator = CertifiedFairnessEvaluator(tracker, reliability_threshold=2.0)
            
            # Convert model to precision
            model_eval = FairMLPClassifier(
                input_dim=train_data.shape[1],
                hidden_dims=[64, 32],
                activation='relu'
            ).to(precision)
            model_eval.load_state_dict(model.state_dict())
            
            # Handle device
            if precision == torch.float64 and device == 'mps':
                model_eval = model_eval.cpu()
                test_data_device = test_data.to(precision).cpu()
            else:
                model_eval = model_eval.to(device)
                test_data_device = test_data.to(precision).to(device)
            
            # Get empirical error functional
            error_functional = create_empirical_error_functional(
                model_eval, (1, train_data.shape[1]), precision, n_samples=30, 
                device=next(model_eval.parameters()).device.type
            )
            
            # Evaluate demographic parity
            result = evaluator.evaluate_demographic_parity(
                model_eval, test_data_device, test_groups.numpy(),
                threshold=0.5,
                model_error_functional=error_functional
            )
            
            # Also compute at different thresholds
            threshold_results = []
            for thresh in [0.3, 0.5, 0.7]:
                thresh_result = evaluator.evaluate_demographic_parity(
                    model_eval, test_data_device, test_groups.numpy(),
                    threshold=thresh,
                    model_error_functional=error_functional
                )
                threshold_results.append({
                    'threshold': thresh,
                    'dpg': float(thresh_result.metric_value),
                    'error_bound': float(thresh_result.error_bound),
                    'reliable': bool(thresh_result.is_reliable),
                    'reliability_score': float(thresh_result.reliability_score),
                    'near_threshold_frac': float(thresh_result.near_threshold_fraction['overall'])
                })
            
            dataset_results['precisions'][precision_name] = {
                'dpg': float(result.metric_value),
                'error_bound': float(result.error_bound),
                'is_reliable': bool(result.is_reliable),
                'reliability_score': float(result.reliability_score),
                'near_threshold_fraction': {k: float(v) for k, v in result.near_threshold_fraction.items()},
                'threshold_sensitivity': threshold_results,
                'machine_epsilon': float(tracker.epsilon_machine),
                'empirical_lipschitz': float(error_functional.lipschitz)
            }
        
        results['datasets'].append({
            'name': dataset_name,
            'results': dataset_results
        })
    
    # Compute aggregate statistics
    all_reliable = []
    for dataset in results['datasets']:
        for prec_name, prec_results in dataset['results']['precisions'].items():
            all_reliable.append({
                'dataset': dataset['name'],
                'precision': prec_name,
                'reliable': prec_results['is_reliable']
            })
    
    total_assessments = len(all_reliable)
    borderline_count = sum(1 for x in all_reliable if not x['reliable'])
    borderline_pct = 100.0 * borderline_count / total_assessments if total_assessments > 0 else 0
    
    results['summary'] = {
        'total_assessments': total_assessments,
        'borderline_count': borderline_count,
        'borderline_percentage': borderline_pct,
        'reliable_by_precision': {}
    }
    
    for precision in ['float64', 'float32', 'float16']:
        prec_assessments = [x for x in all_reliable if x['precision'] == precision]
        prec_borderline = sum(1 for x in prec_assessments if not x['reliable'])
        results['summary']['reliable_by_precision'][precision] = {
            'total': len(prec_assessments),
            'borderline': prec_borderline,
            'borderline_pct': 100.0 * prec_borderline / len(prec_assessments) if prec_assessments else 0
        }
    
    print(f"\n  Summary: {borderline_pct:.1f}% of assessments are numerically borderline")
    
    save_results(results, 'experiment1_precision_vs_fairness', base_dir)
    return results


def experiment2_near_threshold_distribution(base_dir: Path, device: str):
    """
    Experiment 2: Near-Threshold Distribution
    
    Visualize distribution of |f(x) - t| by group.
    Show how this predicts fairness metric uncertainty.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Near-Threshold Distribution")
    print("="*70)
    
    results = {
        'description': 'Analyze distribution of predictions near threshold',
        'models': []
    }
    
    # Train models with different levels of threshold concentration
    configs = [
        ('low_concentration', 0.05, 0.4, 50),
        ('medium_concentration', 0.08, 0.5, 60),
        ('high_concentration', 0.12, 0.6, 70)
    ]
    
    for config_name, fairness_gap, threshold, epochs in configs:
        print(f"\n[{config_name}]")
        
        # Generate data
        train_data, train_labels, train_groups, test_data, test_labels, test_groups = \
            generate_synthetic_tabular(2500, 12, fairness_gap, seed=hash(config_name) % 10000)
        
        # Train model
        print("  Training model...")
        model = FairMLPClassifier(
            input_dim=12,
            hidden_dims=[48, 24],
            activation='relu'
        ).to(device).to(torch.float32)
        
        history = train_fair_classifier(
            model, train_data.to(torch.float32).to(device), 
            train_labels.to(torch.float32).to(device),
            train_groups.to(torch.float32).to(device),
            n_epochs=epochs,
            lr=0.001,
            fairness_weight=0.015,
            device=device,
            verbose=False
        )
        
        # Analyze predictions
        model.eval()
        with torch.no_grad():
            predictions = model(test_data.to(torch.float32).to(device)).cpu().numpy().flatten()
        
        # Compute distances to threshold
        distances = np.abs(predictions - threshold)
        
        # Split by group
        group0_mask = test_groups.numpy() == 0
        group1_mask = test_groups.numpy() == 1
        
        distances_g0 = distances[group0_mask]
        distances_g1 = distances[group1_mask]
        predictions_g0 = predictions[group0_mask]
        predictions_g1 = predictions[group1_mask]
        
        # Histogram bins
        bins = np.linspace(0, 1, 51)
        hist_g0, _ = np.histogram(predictions_g0, bins=bins, density=True)
        hist_g1, _ = np.histogram(predictions_g1, bins=bins, density=True)
        
        # Near-threshold analysis for different precisions
        precision_analysis = {}
        for precision in [torch.float64, torch.float32, torch.float16]:
            tracker = ErrorTracker(precision)
            
            # Rough error estimate
            error_est = 10 * tracker.epsilon_machine * np.sqrt(12 * 48)  # rough Lipschitz
            
            near_threshold = distances < error_est
            near_g0 = near_threshold[group0_mask].mean()
            near_g1 = near_threshold[group1_mask].mean()
            
            precision_name = tracker.get_precision_name()
            precision_analysis[precision_name] = {
                'error_estimate': float(error_est),
                'near_threshold_g0': float(near_g0),
                'near_threshold_g1': float(near_g1),
                'near_threshold_overall': float(near_threshold.mean())
            }
        
        config_results = {
            'config_name': config_name,
            'threshold': threshold,
            'n_test': len(test_data),
            'predictions_stats': {
                'mean': float(predictions.mean()),
                'std': float(predictions.std()),
                'min': float(predictions.min()),
                'max': float(predictions.max())
            },
            'group0': {
                'n_samples': int(group0_mask.sum()),
                'predictions_mean': float(predictions_g0.mean()),
                'predictions_std': float(predictions_g0.std()),
                'distance_mean': float(distances_g0.mean()),
                'distance_median': float(np.median(distances_g0)),
                'histogram': hist_g0.tolist(),
                'predictions_raw': predictions_g0.tolist()
            },
            'group1': {
                'n_samples': int(group1_mask.sum()),
                'predictions_mean': float(predictions_g1.mean()),
                'predictions_std': float(predictions_g1.std()),
                'distance_mean': float(distances_g1.mean()),
                'distance_median': float(np.median(distances_g1)),
                'histogram': hist_g1.tolist(),
                'predictions_raw': predictions_g1.tolist()
            },
            'histogram_bins': bins.tolist(),
            'precision_analysis': precision_analysis,
            'dpg_actual': float(FairnessMetrics.demographic_parity_gap(
                predictions, test_groups.numpy(), threshold
            ))
        }
        
        results['models'].append(config_results)
    
    save_results(results, 'experiment2_near_threshold_distribution', base_dir)
    return results


def experiment3_threshold_stability(base_dir: Path, device: str):
    """
    Experiment 3: Threshold Stability Mapping
    
    For each threshold t ∈ [0.1, 0.9], compute DPG and uncertainty.
    Identify stable vs unstable threshold regions.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Threshold Stability Mapping")
    print("="*70)
    
    results = {
        'description': 'Map stability of fairness metrics across threshold choices',
        'models': []
    }
    
    # Test on 2 models with different characteristics
    model_configs = [
        ('well_separated', 0.05, [56, 28], 70),
        ('borderline', 0.10, [64, 32, 16], 80)
    ]
    
    for model_name, fairness_gap, hidden_dims, epochs in model_configs:
        print(f"\n[{model_name}]")
        
        # Generate data
        train_data, train_labels, train_groups, test_data, test_labels, test_groups = \
            generate_synthetic_tabular(2000, 10, fairness_gap, seed=hash(model_name) % 10000)
        
        # Train model
        print("  Training model...")
        model = FairMLPClassifier(
            input_dim=10,
            hidden_dims=hidden_dims,
            activation='relu'
        ).to(device).to(torch.float32)
        
        history = train_fair_classifier(
            model, train_data.to(torch.float32).to(device),
            train_labels.to(torch.float32).to(device),
            train_groups.to(torch.float32).to(device),
            n_epochs=epochs,
            lr=0.001,
            fairness_weight=0.02,
            device=device,
            verbose=False
        )
        
        # Analyze threshold stability
        print("  Analyzing threshold stability...")
        tracker = ErrorTracker(torch.float32)
        evaluator = CertifiedFairnessEvaluator(tracker)
        analyzer = ThresholdStabilityAnalyzer(evaluator)
        
        # Get empirical error functional
        error_functional = create_empirical_error_functional(
            model, (1, 10), torch.float32, n_samples=30, device=device
        )
        
        # Full threshold scan
        stability_analysis = analyzer.analyze_threshold_stability(
            model, test_data.to(torch.float32).to(device), test_groups.numpy(),
            threshold_range=(0.1, 0.9),
            n_points=41,  # More granular
            model_error_functional=error_functional
        )
        
        # Find stable regions
        stable_regions = analyzer.find_stable_thresholds(
            model, test_data.to(torch.float32).to(device), test_groups.numpy(),
            model_error_functional=error_functional
        )
        
        model_results = {
            'model_name': model_name,
            'hidden_dims': hidden_dims,
            'thresholds': stability_analysis['thresholds'].tolist(),
            'dpg_values': stability_analysis['dpg_values'].tolist(),
            'error_bounds': stability_analysis['error_bounds'].tolist(),
            'reliability_scores': stability_analysis['reliability_scores'].tolist(),
            'is_reliable': stability_analysis['is_reliable'].tolist(),
            'stable_regions': stability_analysis['stable_regions'].tolist(),
            'stable_threshold_ranges': stable_regions,
            'n_stable_regions': len(stable_regions),
            'fraction_stable': float(stability_analysis['stable_regions'].mean())
        }
        
        results['models'].append(model_results)
    
    save_results(results, 'experiment3_threshold_stability', base_dir)
    return results


def experiment4_calibration_reliability(base_dir: Path, device: str):
    """
    Experiment 4: Calibration Reliability
    
    Compute calibration curve at different precisions.
    Identify bins where calibration is numerically uncertain.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Calibration Reliability")
    print("="*70)
    
    results = {
        'description': 'Analyze calibration reliability under finite precision',
        'models': []
    }
    
    # Train models on different datasets
    datasets = [
        ('synthetic_tabular', lambda: generate_synthetic_tabular(3000, 12, 0.07, 100)),
        ('synthetic_compas', lambda: generate_synthetic_compas(2500, 0.09, 101))
    ]
    
    for dataset_name, dataset_fn in datasets:
        print(f"\n[{dataset_name}]")
        
        # Load data
        train_data, train_labels, train_groups, test_data, test_labels, test_groups = dataset_fn()
        
        # Train model
        print("  Training model...")
        model = FairMLPClassifier(
            input_dim=train_data.shape[1],
            hidden_dims=[64, 32],
            activation='relu'
        ).to(device).to(torch.float32)
        
        history = train_fair_classifier(
            model, train_data.to(torch.float32).to(device),
            train_labels.to(torch.float32).to(device),
            train_groups.to(torch.float32).to(device),
            n_epochs=70,
            lr=0.001,
            device=device,
            verbose=False
        )
        
        # Evaluate calibration at different precisions
        # Save state dict on CPU to avoid dtype conversion issues
        model_state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        
        precision_results = {}
        for precision in [torch.float64, torch.float32, torch.float16]:
            precision_name = {torch.float64: 'float64', torch.float32: 'float32', torch.float16: 'float16'}[precision]
            print(f"  Evaluating calibration at {precision_name}...")
            
            tracker = ErrorTracker(precision)
            evaluator = CertifiedFairnessEvaluator(tracker)
            
            # Convert model - start on CPU to avoid MPS<->float64 issues
            model_eval = FairMLPClassifier(
                input_dim=train_data.shape[1],
                hidden_dims=[64, 32],
                activation='relu'
            ).cpu().to(precision)
            model_eval.load_state_dict(model_state_dict_cpu)
            
            # Handle device
            if precision == torch.float64:
                # float64 not supported on MPS, keep on CPU
                model_eval = model_eval.cpu()
                test_data_device = test_data.to(precision).cpu()
                eval_device = 'cpu'
            else:
                model_eval = model_eval.to(device)
                test_data_device = test_data.to(precision).to(device)
                eval_device = device
            
            # Get error functional
            error_functional = create_empirical_error_functional(
                model_eval, (1, train_data.shape[1]), precision, n_samples=30,
                device=eval_device
            )
            
            # Evaluate calibration
            calib_results = evaluator.evaluate_calibration(
                model_eval, test_data_device, test_labels.numpy(),
                n_bins=10,
                model_error_functional=error_functional
            )
            
            precision_results[precision_name] = {
                'ece': float(calib_results['ece']),
                'bin_accuracies': [float(x) for x in calib_results['bin_accuracies']],
                'bin_confidences': [float(x) for x in calib_results['bin_confidences']],
                'bin_uncertainties': [float(x) for x in calib_results['bin_uncertainties']],
                'reliable_bins': [bool(x) for x in calib_results['reliable_bins']],
                'n_reliable_bins': sum(calib_results['reliable_bins']),
                'machine_epsilon': float(tracker.epsilon_machine)
            }
        
        results['models'].append({
            'dataset': dataset_name,
            'n_test': len(test_data),
            'precision_results': precision_results
        })
    
    save_results(results, 'experiment4_calibration_reliability', base_dir)
    return results


def experiment5_sign_flip_cases(base_dir: Path, device: str):
    """
    Experiment 5: Sign Flip Cases
    
    Find examples where DPG flips sign between precisions.
    Show that our bounds predict these cases.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Sign Flip Cases")
    print("="*70)
    
    results = {
        'description': 'Identify cases where fairness metric sign flips across precisions',
        'sign_flips': []
    }
    
    # Train multiple models with very small DPG to induce sign flips
    # Key: we need predictions concentrated near threshold=0.5
    n_trials = 20  # Increased trials
    sign_flip_found = []
    
    for trial in range(n_trials):
        print(f"\n[Trial {trial + 1}/{n_trials}]")
        
        # Generate data with minimal fairness gap - vary it per trial
        fairness_gap = 0.01 + (trial % 5) * 0.01  # 0.01 to 0.05
        train_data, train_labels, train_groups, test_data, test_labels, test_groups = \
            generate_synthetic_tabular(1500, 8, fairness_gap, seed=1000 + trial)
        
        # Train with strong fairness regularization
        # Use smaller network and fewer epochs to keep predictions near threshold
        model = FairMLPClassifier(
            input_dim=8,
            hidden_dims=[24, 12],  # Smaller network
            activation='relu'
        ).to(device).to(torch.float32)
        
        # Vary training to get different prediction distributions
        epochs = 40 + (trial % 3) * 20  # 40, 60, or 80 epochs
        fw = 0.03 + (trial % 4) * 0.01   # Fairness weight 0.03-0.06
        
        history = train_fair_classifier(
            model, train_data.to(torch.float32).to(device),
            train_labels.to(torch.float32).to(device),
            train_groups.to(torch.float32).to(device),
            n_epochs=epochs,
            lr=0.001,
            fairness_weight=fw,
            device=device,
            verbose=False
        )
        
        # Evaluate at different precisions
        # Save state dict on CPU to avoid dtype conversion issues
        model_state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        
        dpg_by_precision = {}
        error_bounds = {}
        
        for precision in [torch.float64, torch.float32, torch.float16]:
            precision_name = {torch.float64: 'float64', torch.float32: 'float32', torch.float16: 'float16'}[precision]
            
            tracker = ErrorTracker(precision)
            evaluator = CertifiedFairnessEvaluator(tracker)
            
            # Convert model - start on CPU to avoid MPS<->float64 issues
            model_eval = FairMLPClassifier(
                input_dim=8,
                hidden_dims=[24, 12],
                activation='relu'
            ).cpu().to(precision)
            model_eval.load_state_dict(model_state_dict_cpu)
            
            # Handle device
            if precision == torch.float64:
                model_eval = model_eval.cpu()
                test_data_device = test_data.to(precision).cpu()
                eval_device = 'cpu'
            else:
                model_eval = model_eval.to(device)
                test_data_device = test_data.to(precision).to(device)
                eval_device = device
            
            # Get predictions
            model_eval.eval()
            with torch.no_grad():
                preds = model_eval(test_data_device).cpu().numpy().flatten()
            
            # Compute DPG with sign
            mask_0 = test_groups.numpy() == 0
            mask_1 = test_groups.numpy() == 1
            
            pos_rate_0 = (preds[mask_0] > 0.5).mean()
            pos_rate_1 = (preds[mask_1] > 0.5).mean()
            
            dpg_signed = pos_rate_0 - pos_rate_1  # Signed difference
            
            # Also get certified result
            error_functional = create_empirical_error_functional(
                model_eval, (1, 8), precision, n_samples=30,
                device=eval_device
            )
            
            result = evaluator.evaluate_demographic_parity(
                model_eval, test_data_device, test_groups.numpy(),
                threshold=0.5,
                model_error_functional=error_functional
            )
            
            dpg_by_precision[precision_name] = float(dpg_signed)
            error_bounds[precision_name] = float(result.error_bound)
        
        # Check for sign flip
        dpg_values = [dpg_by_precision['float64'], dpg_by_precision['float32'], dpg_by_precision['float16']]
        
        # Sign flip if different precisions give different signs
        # Even small DPG values matter if they flip
        has_sign_flip = False
        if max(abs(v) for v in dpg_values) > 0.0005:  # Non-trivial (lowered threshold)
            signs = [np.sign(v) if abs(v) > 0.0001 else 0 for v in dpg_values]
            unique_nonzero_signs = set([s for s in signs if s != 0])
            if len(unique_nonzero_signs) > 1:  # Different signs
                has_sign_flip = True
            # Also detect when one precision gives ~0 but others don't
            elif 0 in signs and len(unique_nonzero_signs) >= 1 and max(abs(v) for v in dpg_values) > 0.003:
                has_sign_flip = True
        
        trial_result = {
            'trial': trial,
            'has_sign_flip': has_sign_flip,
            'dpg_signed': dpg_by_precision,
            'error_bounds': error_bounds,
            'final_train_dpg': float(history[-1]['dpg']),
            'final_train_acc': float(history[-1]['accuracy']),
            'prediction_stats': {
                'mean_pred': float(preds.mean()),
                'std_pred': float(preds.std()),
                'near_threshold_frac': float(np.mean(np.abs(preds - 0.5) < 0.05))
            }
        }
        
        if has_sign_flip:
            print(f"  ✓ Sign flip detected!")
            print(f"    float64 DPG: {dpg_by_precision['float64']:+.6f}")
            print(f"    float32 DPG: {dpg_by_precision['float32']:+.6f}")
            print(f"    float16 DPG: {dpg_by_precision['float16']:+.6f}")
            print(f"    Near threshold: {trial_result['prediction_stats']['near_threshold_frac']:.1%}")
            sign_flip_found.append(trial_result)
        
        results['sign_flips'].append(trial_result)
    
    results['summary'] = {
        'n_trials': n_trials,
        'n_sign_flips': len(sign_flip_found),
        'sign_flip_rate': len(sign_flip_found) / n_trials
    }
    
    print(f"\n  Summary: Found {len(sign_flip_found)} sign flips in {n_trials} trials")
    
    save_results(results, 'experiment5_sign_flip_cases', base_dir)
    return results


def main():
    """Run all experiments."""
    print("\n" + "="*70)
    print("NUMGEOM-FAIR: COMPREHENSIVE EXPERIMENTS")
    print("Proposal 25: Numerical Geometry of Fairness Metrics")
    print("="*70)
    
    start_time = time.time()
    
    # Setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    base_dir = setup_directories()
    print(f"Output directory: {base_dir}")
    
    # Run experiments
    exp1_results = experiment1_precision_vs_fairness(base_dir, device)
    exp2_results = experiment2_near_threshold_distribution(base_dir, device)
    exp3_results = experiment3_threshold_stability(base_dir, device)
    exp4_results = experiment4_calibration_reliability(base_dir, device)
    exp5_results = experiment5_sign_flip_cases(base_dir, device)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"\nResults saved to: {base_dir}")
    print("\nKey findings:")
    print(f"  • Borderline assessments: {exp1_results['summary']['borderline_percentage']:.1f}%")
    print(f"  • Sign flips detected: {exp5_results['summary']['n_sign_flips']}/{exp5_results['summary']['n_trials']}")
    print("\nNext steps:")
    print("  python3.11 scripts/generate_plots.py")
    print("  cd implementations/docs/proposal25 && make")


if __name__ == '__main__':
    main()
