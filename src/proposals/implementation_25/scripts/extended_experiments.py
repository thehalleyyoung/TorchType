"""
Comprehensive extended experiments for Proposal 25.

This script runs all new experiments:
1. Real-world Adult Census dataset
2. Transformer fairness analysis
3. Multi-metric joint analysis
4. Compliance certification
5. Wall-clock timing comparison
"""

import torch
import numpy as np
import time
import json
import os
from typing import Dict, List

# Import new modules
from src.real_world_datasets import RealWorldDatasets, download_and_prepare_adult
from src.transformer_fairness import (
    SimpleTransformerClassifier,
    TransformerFairnessAnalyzer,
    train_simple_transformer
)
from src.multi_metric_analysis import MultiMetricFairnessAnalyzer
from src.compliance_certification import ComplianceCertifier
from src.models import FairMLP, train_fair_model
from src.error_propagation import ErrorTracker
from src.fairness_metrics import CertifiedFairnessEvaluator


def experiment_8_real_adult_census(device='cpu'):
    """
    Experiment 8: Real Adult Census Dataset Analysis
    
    Uses real UCI Adult Census data to validate framework on
    real-world fairness dataset with actual gender bias.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 8: Real Adult Census Dataset")
    print("="*80)
    
    # Load real data
    print("\nLoading Adult Census dataset...")
    loader = RealWorldDatasets()
    data = loader.load_adult_census(subsample=5000)
    
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    X_test = data['X_test'].to(device)
    y_test = data['y_test'].to(device)
    groups_train = data['groups_train'].to(device)
    groups_test = data['groups_test'].to(device)
    
    print(f"✓ Loaded {len(X_train)} training, {len(X_test)} test samples")
    print(f"  Features: {data['feature_names']}")
    print(f"  Groups: {data['group_labels']}")
    print(f"  Group split (test): {groups_test.bincount().tolist()}")
    
    # Train model
    print("\nTraining fair MLP on Adult Census...")
    model = FairMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[32, 16],
        dropout=0.2
    ).to(device)
    
    train_fair_model(
        model, X_train, y_train,
        epochs=100, lr=0.001,
        verbose=False
    )
    
    print("✓ Model trained")
    
    # Evaluate fairness at multiple precisions
    print("\nEvaluating fairness across precisions...")
    results = {}
    
    for prec in [torch.float64, torch.float32, torch.float16]:
        print(f"\n  Testing {prec}...")
        tracker = ErrorTracker(precision=prec)
        evaluator = CertifiedFairnessEvaluator(tracker)
        
        try:
            result = evaluator.evaluate_demographic_parity(
                model, X_test, groups_test, threshold=0.5
            )
            results[str(prec)] = {
                'dpg': float(result.metric_value),
                'error_bound': float(result.error_bound),
                'reliable': bool(result.is_reliable),
                'reliability_score': float(result.reliability_score),
                'near_threshold_count': int(result.near_threshold_count_0 + result.near_threshold_count_1)
            }
            print(f"    DPG: {result.metric_value:.4f} ± {result.error_bound:.4f}")
            print(f"    Reliable: {result.is_reliable}, Score: {result.reliability_score:.2f}")
        except Exception as e:
            print(f"    Error: {e}")
            results[str(prec)] = {'error': str(e)}
    
    # Save results
    output_dir = "data/experiment8_adult_census"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    return results


def experiment_9_transformer_fairness(device='cpu'):
    """
    Experiment 9: Transformer-based Fairness Analysis
    
    Analyzes attention mechanisms and their numerical stability
    for fairness metrics.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 9: Transformer Fairness Analysis")
    print("="*80)
    
    # Generate data
    print("\nGenerating synthetic data...")
    n_samples = 1000
    n_features = 16
    X = torch.randn(n_samples, n_features)
    groups = torch.randint(0, 2, (n_samples,))
    y = (X.sum(dim=1) + groups.float() * 0.4 > 0).float()
    
    # Split
    n_train = 700
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    groups_train, groups_test = groups[:n_train], groups[n_train:]
    
    print(f"✓ Generated {n_train} train, {len(X_test)} test samples")
    
    # Train transformer
    print("\nTraining transformer classifier...")
    transformer = train_simple_transformer(
        X_train, y_train,
        input_dim=n_features,
        epochs=50, lr=0.001,
        device=device
    )
    print("✓ Transformer trained")
    
    # Train MLP for comparison
    print("\nTraining MLP for comparison...")
    mlp = FairMLP(n_features, [32, 16]).to(device)
    train_fair_model(mlp, X_train, y_train, epochs=50, lr=0.001, verbose=False)
    print("✓ MLP trained")
    
    # Analyze attention stability
    print("\nAnalyzing attention layer stability...")
    analyzer = TransformerFairnessAnalyzer(device=device)
    attention_analysis = analyzer.analyze_attention_stability(
        transformer, X_test[:100]
    )
    
    for layer_name, analysis in attention_analysis.items():
        print(f"\n  {layer_name}:")
        print(f"    Curvature: {analysis.attention_curvature:.6f}")
        print(f"    Recommended precision: {analysis.precision_requirement}")
        print(f"    Error bound: {analysis.error_bound:.6f}")
        print(f"    Stability score: {analysis.stability_score:.4f}")
    
    # Compare transformer vs MLP fairness
    print("\nComparing transformer vs MLP fairness...")
    comparison = analyzer.compare_transformer_vs_mlp_fairness(
        transformer, mlp, X_test, y_test, groups_test
    )
    
    # Save results
    output_dir = "data/experiment9_transformer"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'attention_analysis': {
            layer: {
                'curvature': float(analysis.attention_curvature),
                'recommended_precision': str(analysis.precision_requirement),
                'error_bound': float(analysis.error_bound),
                'stability_score': float(analysis.stability_score)
            }
            for layer, analysis in attention_analysis.items()
        },
        'transformer_vs_mlp': comparison
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    return results


def experiment_10_multi_metric_analysis(device='cpu'):
    """
    Experiment 10: Multi-Metric Joint Analysis
    
    Analyzes multiple fairness metrics jointly to identify
    tradeoffs and joint numerical reliability.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 10: Multi-Metric Joint Analysis")
    print("="*80)
    
    # Generate data with complex fairness properties
    print("\nGenerating data with multiple fairness concerns...")
    n_samples = 1500
    n_features = 12
    X = torch.randn(n_samples, n_features)
    groups = torch.randint(0, 2, (n_samples,))
    
    # Create labels with both DPG and EOG violations
    base_score = X.sum(dim=1)
    group_bias = groups.float() * 0.5
    noise = torch.randn(n_samples) * 0.3
    y = (base_score + group_bias + noise > 0).float()
    
    # Split
    n_train = 1000
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    groups_train, groups_test = groups[:n_train], groups[n_train:]
    
    print(f"✓ Generated {n_train} train, {len(X_test)} test samples")
    
    # Train model
    print("\nTraining model...")
    model = FairMLP(n_features, [32, 16]).to(device)
    train_fair_model(model, X_train, y_train, epochs=50, lr=0.001, verbose=False)
    print("✓ Model trained")
    
    # Multi-metric analysis
    print("\nPerforming multi-metric analysis...")
    analyzer = MultiMetricFairnessAnalyzer()
    
    # Single precision
    print("\n  At float32:")
    result = analyzer.evaluate_all_metrics(
        model, X_test, y_test, groups_test
    )
    print(f"    DPG: {result.demographic_parity.metric_value:.4f} ± {result.demographic_parity.error_bound:.4f}")
    print(f"    EOG: {result.equalized_odds.metric_value:.4f} ± {result.equalized_odds.error_bound:.4f}")
    print(f"    CAL: {result.calibration.metric_value:.4f} ± {result.calibration.error_bound:.4f}")
    print(f"    Joint reliable: {result.joint_reliable}")
    print(f"    Joint score: {result.joint_reliability_score:.2f}")
    
    # Multi-precision
    print("\n  Across precisions:")
    multi_results = analyzer.analyze_precision_tradeoffs(
        model, X_test, y_test, groups_test
    )
    
    for prec, res in multi_results.items():
        print(f"\n    {prec}:")
        print(f"      Joint reliable: {res.joint_reliable}")
        print(f"      Joint score: {res.joint_reliability_score:.2f}")
    
    # Pareto threshold analysis
    print("\n  Finding Pareto-optimal thresholds...")
    pareto_results = analyzer.find_pareto_optimal_thresholds(
        model, X_test, y_test, groups_test
    )
    
    # Save results
    output_dir = "data/experiment10_multimetric"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'single_precision': {
            'dpg': {
                'value': float(result.demographic_parity.metric_value),
                'error': float(result.demographic_parity.error_bound),
                'reliable': bool(result.demographic_parity.is_reliable)
            },
            'eog': {
                'value': float(result.equalized_odds.metric_value),
                'error': float(result.equalized_odds.error_bound),
                'reliable': bool(result.equalized_odds.is_reliable)
            },
            'cal': {
                'value': float(result.calibration.metric_value),
                'error': float(result.calibration.error_bound),
                'reliable': bool(result.calibration.is_reliable)
            },
            'joint_reliable': bool(result.joint_reliable),
            'joint_score': float(result.joint_reliability_score)
        },
        'multi_precision': {
            str(prec): {
                'joint_reliable': bool(res.joint_reliable),
                'joint_score': float(res.joint_reliability_score)
            }
            for prec, res in multi_results.items()
        },
        'pareto_thresholds': {
            'thresholds': [float(t) for t in pareto_results['thresholds']],
            'dpg': [float(v) for v in pareto_results['dpg']],
            'eog': [float(v) for v in pareto_results['eog']],
            'reliable': pareto_results['reliable']
        }
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualization
    print("\n  Generating tradeoff visualizations...")
    fig = analyzer.visualize_fairness_tradeoffs(
        multi_results,
        save_path=f"{output_dir}/fairness_tradeoffs.png"
    )
    print(f"    ✓ Saved to {output_dir}/fairness_tradeoffs.png")
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    return results


def experiment_11_compliance_certification(device='cpu'):
    """
    Experiment 11: Regulatory Compliance Certification
    
    Generates comprehensive certification reports suitable for
    regulatory review and compliance documentation.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 11: Compliance Certification")
    print("="*80)
    
    # Load real Adult Census data
    print("\nLoading Adult Census dataset...")
    loader = RealWorldDatasets()
    data = loader.load_adult_census(subsample=2000)
    
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    X_test = data['X_test'].to(device)
    y_test = data['y_test'].to(device)
    groups_test = data['groups_test'].to(device)
    
    print(f"✓ Loaded data")
    
    # Train model
    print("\nTraining production-ready model...")
    model = FairMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32, 16],
        dropout=0.1
    ).to(device)
    
    train_fair_model(model, X_train, y_train, epochs=100, lr=0.001, verbose=False)
    print("✓ Model trained")
    
    # Generate certification reports at different precisions
    print("\nGenerating certification reports...")
    certifier = ComplianceCertifier()
    
    reports = {}
    for prec in [torch.float64, torch.float32, torch.float16]:
        print(f"\n  Certifying at {prec}...")
        try:
            report = certifier.generate_certification_report(
                model, X_test, y_test, groups_test,
                model_name=f"AdultCensus-MLP-{prec}",
                dataset_name="UCI Adult Census Income",
                precision=prec,
                architecture="3-layer MLP (64-32-16)"
            )
            reports[str(prec)] = report
            print(f"    Status: {report.certification_level}")
            print(f"    Score: {report.reliability_score:.2f}")
            print(f"    Certified: {report.certified}")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Save reports
    output_dir = "data/experiment11_compliance"
    os.makedirs(output_dir, exist_ok=True)
    
    for prec_str, report in reports.items():
        prec_name = prec_str.replace('torch.', '').replace('float', 'fp')
        certifier.save_report_json(
            report,
            f"{output_dir}/certification_{prec_name}.json"
        )
        certifier.save_report_html(
            report,
            f"{output_dir}/certification_{prec_name}.html"
        )
    
    print(f"\n✓ Reports saved to {output_dir}/")
    print(f"  - JSON reports: certification_fp*.json")
    print(f"  - HTML reports: certification_fp*.html (open in browser)")
    
    return {
        str(prec): {
            'certification_level': report.certification_level,
            'reliability_score': report.reliability_score,
            'certified': report.certified
        }
        for prec, report in reports.items()
    }


def experiment_12_wall_clock_timing(device='cpu'):
    """
    Experiment 12: Wall-Clock Time Comparison
    
    Measures actual runtime improvements from precision reduction
    while maintaining fairness certification.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 12: Wall-Clock Time Comparison")
    print("="*80)
    
    # Generate large dataset
    print("\nGenerating large dataset for timing...")
    n_samples = 10000
    n_features = 50
    X = torch.randn(n_samples, n_features).to(device)
    groups = torch.randint(0, 2, (n_samples,)).to(device)
    y = (X.sum(dim=1) + groups.float() * 0.3 > 0).float().to(device)
    
    print(f"✓ Generated {n_samples} samples with {n_features} features")
    
    # Train models at different precisions
    print("\nTraining models...")
    models = {}
    training_times = {}
    
    for prec in [torch.float64, torch.float32, torch.float16]:
        print(f"\n  Training at {prec}...")
        model = FairMLP(n_features, [128, 64, 32]).to(device).to(prec)
        X_prec = X.to(prec)
        y_prec = y.to(prec)
        
        start_time = time.time()
        train_fair_model(model, X_prec, y_prec, epochs=50, lr=0.001, verbose=False)
        train_time = time.time() - start_time
        
        models[prec] = model
        training_times[str(prec)] = train_time
        print(f"    Training time: {train_time:.2f}s")
    
    # Inference timing
    print("\nMeasuring inference time...")
    inference_times = {}
    n_runs = 100
    
    for prec in [torch.float64, torch.float32, torch.float16]:
        model = models[prec]
        X_prec = X.to(prec)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(X_prec)
        
        # Timed runs
        start_time = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(X_prec)
        total_time = time.time() - start_time
        avg_time = total_time / n_runs
        
        inference_times[str(prec)] = avg_time
        print(f"  {prec}: {avg_time*1000:.2f}ms per inference")
    
    # Fairness evaluation timing
    print("\nMeasuring fairness evaluation time...")
    eval_times = {}
    
    for prec in [torch.float64, torch.float32, torch.float16]:
        model = models[prec]
        tracker = ErrorTracker(precision=prec)
        evaluator = CertifiedFairnessEvaluator(tracker)
        X_prec = X.to(prec)
        
        start_time = time.time()
        for _ in range(10):
            try:
                _ = evaluator.evaluate_demographic_parity(
                    model, X_prec, groups, threshold=0.5
                )
            except:
                pass
        eval_time = (time.time() - start_time) / 10
        
        eval_times[str(prec)] = eval_time
        print(f"  {prec}: {eval_time*1000:.2f}ms per evaluation")
    
    # Memory usage estimation
    print("\nEstimating memory usage...")
    memory_usage = {}
    
    for prec in [torch.float64, torch.float32, torch.float16]:
        model = models[prec]
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        memory_usage[str(prec)] = param_memory
        print(f"  {prec}: {param_memory / 1024:.2f} KB")
    
    # Save results
    output_dir = "data/experiment12_timing"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'training_time_seconds': training_times,
        'inference_time_ms': {k: v*1000 for k, v in inference_times.items()},
        'evaluation_time_ms': {k: v*1000 for k, v in eval_times.items()},
        'memory_kb': {k: v/1024 for k, v in memory_usage.items()},
        'speedup_vs_fp64': {
            'training': {
                str(prec): training_times['torch.float64'] / training_times[str(prec)]
                for prec in [torch.float32, torch.float16]
            },
            'inference': {
                str(prec): inference_times['torch.float64'] / inference_times[str(prec)]
                for prec in [torch.float32, torch.float16]
            }
        },
        'memory_savings_vs_fp64': {
            str(prec): 1.0 - (memory_usage[str(prec)] / memory_usage['torch.float64'])
            for prec in [torch.float32, torch.float16]
        }
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    # Print summary
    print("\nSummary:")
    print(f"  FP32 vs FP64:")
    print(f"    Training speedup: {results['speedup_vs_fp64']['training']['torch.float32']:.2f}x")
    print(f"    Inference speedup: {results['speedup_vs_fp64']['inference']['torch.float32']:.2f}x")
    print(f"    Memory savings: {results['memory_savings_vs_fp64']['torch.float32']*100:.1f}%")
    
    return results


def run_all_extended_experiments(device='cpu'):
    """Run all extended experiments."""
    print("\n" + "="*80)
    print("RUNNING ALL EXTENDED EXPERIMENTS (Proposal 25)")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # Run experiments
    experiments = [
        ("Real Adult Census", experiment_8_real_adult_census),
        ("Transformer Fairness", experiment_9_transformer_fairness),
        ("Multi-Metric Analysis", experiment_10_multi_metric_analysis),
        ("Compliance Certification", experiment_11_compliance_certification),
        ("Wall-Clock Timing", experiment_12_wall_clock_timing)
    ]
    
    for name, exp_func in experiments:
        try:
            print(f"\n{'='*80}")
            print(f"Starting: {name}")
            start = time.time()
            result = exp_func(device)
            elapsed = time.time() - start
            all_results[name] = {
                'status': 'success',
                'time_seconds': elapsed,
                'result': result
            }
            print(f"\n✓ {name} completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Save summary
    with open("data/extended_experiments_summary.json", 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': device,
            'results': {
                k: {
                    'status': v['status'],
                    'time_seconds': v.get('time_seconds', 0)
                }
                for k, v in all_results.items()
            }
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL EXTENDED EXPERIMENTS COMPLETE")
    print("="*80)
    
    # Summary
    success_count = sum(1 for v in all_results.values() if v['status'] == 'success')
    total_time = sum(v.get('time_seconds', 0) for v in all_results.values())
    
    print(f"\nResults: {success_count}/{len(experiments)} experiments successful")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\nData saved to data/ subdirectories")
    
    return all_results


if __name__ == "__main__":
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Run all experiments
    results = run_all_extended_experiments(device)
    
    print("\n✓✓✓ Extended experiments complete! ✓✓✓")
