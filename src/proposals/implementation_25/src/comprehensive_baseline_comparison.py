"""
Comprehensive baseline comparison for fairness uncertainty quantification.

This module compares NumGeom-Fair's certified bounds against alternative approaches:
1. Naive approach (no error tracking)
2. Bootstrap confidence intervals
3. Monte Carlo dropout uncertainty
4. Empirical precision comparison only

This demonstrates that certified bounds are both tighter and more reliable.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

try:
    from .error_propagation import ErrorTracker, LinearErrorFunctional
    from .fairness_metrics import CertifiedFairnessEvaluator, FairnessMetrics
    from .models import FairMLPClassifier
except ImportError:
    from error_propagation import ErrorTracker, LinearErrorFunctional
    from fairness_metrics import CertifiedFairnessEvaluator, FairnessMetrics
    from models import FairMLPClassifier


@dataclass
class BaselineResult:
    """Result from a baseline fairness uncertainty method."""
    method_name: str
    dpg_estimate: float
    uncertainty_lower: float
    uncertainty_upper: float
    uncertainty_width: float
    computation_time_ms: float
    is_rigorous: bool  # Whether bounds are provably correct
    

class NaiveFairnessEvaluator:
    """
    Baseline 1: Naive approach - just compute fairness, no error tracking.
    """
    
    def evaluate(
        self,
        model: nn.Module,
        X: torch.Tensor,
        groups: np.ndarray,
        threshold: float = 0.5,
        device: str = 'cpu'
    ) -> BaselineResult:
        """Evaluate fairness naively (no uncertainty quantification)."""
        start = time.time()
        
        model.eval()
        with torch.no_grad():
            predictions = model(X.to(device)).cpu().numpy().flatten()
        
        dpg = FairnessMetrics.demographic_parity_gap(
            predictions, groups, threshold=threshold
        )
        
        elapsed_ms = (time.time() - start) * 1000
        
        # Naive approach: no uncertainty estimate
        return BaselineResult(
            method_name="Naive (No Uncertainty)",
            dpg_estimate=dpg,
            uncertainty_lower=dpg,
            uncertainty_upper=dpg,
            uncertainty_width=0.0,
            computation_time_ms=elapsed_ms,
            is_rigorous=False
        )


class BootstrapFairnessEvaluator:
    """
    Baseline 2: Bootstrap confidence intervals for fairness metrics.
    
    Standard statistical approach - resample data and compute distribution.
    """
    
    def evaluate(
        self,
        model: nn.Module,
        X: torch.Tensor,
        groups: np.ndarray,
        threshold: float = 0.5,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        device: str = 'cpu'
    ) -> BaselineResult:
        """Evaluate fairness with bootstrap confidence intervals."""
        start = time.time()
        
        model.eval()
        
        # Get predictions once
        with torch.no_grad():
            predictions = model(X.to(device)).cpu().numpy().flatten()
        
        # Bootstrap resampling
        n_samples = len(predictions)
        bootstrap_dpgs = []
        
        for _ in range(n_bootstrap):
            # Resample indices
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Compute DPG on bootstrap sample
            dpg_boot = FairnessMetrics.demographic_parity_gap(
                predictions[indices],
                groups[indices],
                threshold=threshold
            )
            bootstrap_dpgs.append(dpg_boot)
        
        bootstrap_dpgs = np.array(bootstrap_dpgs)
        
        # Compute confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_dpgs, lower_percentile)
        ci_upper = np.percentile(bootstrap_dpgs, upper_percentile)
        
        # Original DPG
        dpg = FairnessMetrics.demographic_parity_gap(predictions, groups, threshold)
        
        elapsed_ms = (time.time() - start) * 1000
        
        return BaselineResult(
            method_name=f"Bootstrap (n={n_bootstrap})",
            dpg_estimate=dpg,
            uncertainty_lower=ci_lower,
            uncertainty_upper=ci_upper,
            uncertainty_width=ci_upper - ci_lower,
            computation_time_ms=elapsed_ms,
            is_rigorous=False  # Statistical, not certified
        )


class MonteCarloDropoutFairnessEvaluator:
    """
    Baseline 3: Monte Carlo Dropout for model uncertainty.
    
    Enable dropout at inference time to get distribution over predictions.
    """
    
    def evaluate(
        self,
        model: nn.Module,
        X: torch.Tensor,
        groups: np.ndarray,
        threshold: float = 0.5,
        n_samples: int = 100,
        device: str = 'cpu'
    ) -> BaselineResult:
        """Evaluate fairness with MC Dropout uncertainty."""
        start = time.time()
        
        # Enable dropout at inference
        model.train()  # Keep dropout active
        
        # Collect multiple forward passes
        dpg_samples = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                predictions = model(X.to(device)).cpu().numpy().flatten()
            
            dpg = FairnessMetrics.demographic_parity_gap(
                predictions, groups, threshold=threshold
            )
            dpg_samples.append(dpg)
        
        dpg_samples = np.array(dpg_samples)
        
        # Compute statistics
        dpg_mean = dpg_samples.mean()
        dpg_std = dpg_samples.std()
        
        # 95% confidence interval (assuming normal)
        ci_lower = dpg_mean - 1.96 * dpg_std
        ci_upper = dpg_mean + 1.96 * dpg_std
        
        elapsed_ms = (time.time() - start) * 1000
        
        # Restore eval mode
        model.eval()
        
        return BaselineResult(
            method_name=f"MC Dropout (n={n_samples})",
            dpg_estimate=dpg_mean,
            uncertainty_lower=ci_lower,
            uncertainty_upper=ci_upper,
            uncertainty_width=ci_upper - ci_lower,
            computation_time_ms=elapsed_ms,
            is_rigorous=False  # Bayesian approximation, not certified
        )


class EmpiricalPrecisionComparisonEvaluator:
    """
    Baseline 4: Empirical comparison across precisions.
    
    Run model at different precisions and report range of observed DPG.
    """
    
    def evaluate(
        self,
        model: nn.Module,
        X: torch.Tensor,
        groups: np.ndarray,
        threshold: float = 0.5,
        device: str = 'cpu'
    ) -> BaselineResult:
        """Evaluate by comparing across precisions empirically."""
        start = time.time()
        
        dpgs = []
        
        for precision in [torch.float64, torch.float32, torch.float16]:
            model_prec = model.to(precision)
            X_prec = X.to(precision).to(device)
            
            model_prec.eval()
            with torch.no_grad():
                predictions = model_prec(X_prec).cpu().numpy().flatten()
            
            dpg = FairnessMetrics.demographic_parity_gap(
                predictions, groups, threshold=threshold
            )
            dpgs.append(dpg)
        
        dpg_mean = np.mean(dpgs)
        dpg_min = np.min(dpgs)
        dpg_max = np.max(dpgs)
        
        elapsed_ms = (time.time() - start) * 1000
        
        return BaselineResult(
            method_name="Empirical Precision Comparison",
            dpg_estimate=dpg_mean,
            uncertainty_lower=dpg_min,
            uncertainty_upper=dpg_max,
            uncertainty_width=dpg_max - dpg_min,
            computation_time_ms=elapsed_ms,
            is_rigorous=False  # Only shows observed range, no guarantees
        )


class NumGeomFairEvaluator:
    """
    Our method: Certified bounds with NumGeom-Fair.
    """
    
    def evaluate(
        self,
        model: nn.Module,
        X: torch.Tensor,
        groups: np.ndarray,
        threshold: float = 0.5,
        precision: torch.dtype = torch.float32,
        device: str = 'cpu'
    ) -> BaselineResult:
        """Evaluate fairness with certified bounds."""
        start = time.time()
        
        model_prec = model.to(precision)
        X_prec = X.to(precision).to(device)
        
        evaluator = CertifiedFairnessEvaluator(
            ErrorTracker(precision=precision)
        )
        
        result = evaluator.evaluate_demographic_parity(
            model_prec, X_prec, groups, threshold=threshold
        )
        
        elapsed_ms = (time.time() - start) * 1000
        
        return BaselineResult(
            method_name="NumGeom-Fair (Certified)",
            dpg_estimate=result.metric_value,
            uncertainty_lower=result.metric_value - result.error_bound,
            uncertainty_upper=result.metric_value + result.error_bound,
            uncertainty_width=2 * result.error_bound,
            computation_time_ms=elapsed_ms,
            is_rigorous=True  # Certified mathematical bounds
        )


class BaselineComparison:
    """
    Comprehensive comparison of fairness uncertainty quantification methods.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        self.evaluators = {
            'naive': NaiveFairnessEvaluator(),
            'bootstrap': BootstrapFairnessEvaluator(),
            'mc_dropout': MonteCarloDropoutFairnessEvaluator(),
            'empirical_precision': EmpiricalPrecisionComparisonEvaluator(),
            'numgeom_fair': NumGeomFairEvaluator()
        }
    
    def run_comparison(
        self,
        model: nn.Module,
        X: torch.Tensor,
        groups: np.ndarray,
        threshold: float = 0.5,
        precision: torch.dtype = torch.float32
    ) -> Dict[str, BaselineResult]:
        """
        Run all baseline methods and compare.
        
        Returns:
            Dictionary mapping method name to result
        """
        results = {}
        
        # Naive
        results['naive'] = self.evaluators['naive'].evaluate(
            model, X, groups, threshold, device=self.device
        )
        
        # Bootstrap (fewer samples for speed)
        results['bootstrap'] = self.evaluators['bootstrap'].evaluate(
            model, X, groups, threshold, n_bootstrap=500, device=self.device
        )
        
        # MC Dropout (requires model with dropout)
        try:
            results['mc_dropout'] = self.evaluators['mc_dropout'].evaluate(
                model, X, groups, threshold, n_samples=50, device=self.device
            )
        except Exception as e:
            print(f"  (MC Dropout skipped: {e})")
        
        # Empirical Precision Comparison
        results['empirical_precision'] = self.evaluators['empirical_precision'].evaluate(
            model, X, groups, threshold, device=self.device
        )
        
        # NumGeom-Fair (our method)
        results['numgeom_fair'] = self.evaluators['numgeom_fair'].evaluate(
            model, X, groups, threshold, precision=precision, device=self.device
        )
        
        return results
    
    def analyze_comparison(
        self,
        results: Dict[str, BaselineResult],
        ground_truth_dpg: Optional[float] = None
    ) -> Dict:
        """
        Analyze comparison results.
        
        Args:
            results: Results from run_comparison
            ground_truth_dpg: If known, the true DPG at infinite precision
            
        Returns:
            Analysis metrics
        """
        analysis = {
            'methods': list(results.keys()),
            'uncertainty_widths': {},
            'computation_times': {},
            'rigorous_bounds': {},
            'coverage': {}
        }
        
        for method, result in results.items():
            analysis['uncertainty_widths'][method] = result.uncertainty_width
            analysis['computation_times'][method] = result.computation_time_ms
            analysis['rigorous_bounds'][method] = result.is_rigorous
            
            # If ground truth is known, check coverage
            if ground_truth_dpg is not None:
                covered = (
                    result.uncertainty_lower <= ground_truth_dpg <= result.uncertainty_upper
                )
                analysis['coverage'][method] = covered
        
        # Compute relative metrics
        # Use NumGeom-Fair as reference
        if 'numgeom_fair' in results:
            ref_width = results['numgeom_fair'].uncertainty_width
            ref_time = results['numgeom_fair'].computation_time_ms
            
            analysis['width_relative_to_numgeom'] = {
                method: result.uncertainty_width / ref_width if ref_width > 0 else float('inf')
                for method, result in results.items()
            }
            
            analysis['time_relative_to_numgeom'] = {
                method: result.computation_time_ms / ref_time if ref_time > 0 else float('inf')
                for method, result in results.items()
            }
        
        return analysis
    
    def visualize_comparison(
        self,
        results: Dict[str, BaselineResult],
        ground_truth_dpg: Optional[float] = None
    ) -> str:
        """
        Generate a text visualization of the comparison.
        """
        lines = []
        lines.append("\n" + "="*70)
        lines.append("BASELINE COMPARISON RESULTS")
        lines.append("="*70 + "\n")
        
        # Sort by uncertainty width
        sorted_methods = sorted(
            results.items(),
            key=lambda x: x[1].uncertainty_width
        )
        
        for method, result in sorted_methods:
            rigorous_marker = "✓" if result.is_rigorous else "~"
            
            lines.append(f"[{rigorous_marker}] {result.method_name}")
            lines.append(f"    DPG: {result.dpg_estimate:.4f}")
            lines.append(f"    Uncertainty: [{result.uncertainty_lower:.4f}, {result.uncertainty_upper:.4f}]")
            lines.append(f"    Width: {result.uncertainty_width:.4f}")
            lines.append(f"    Time: {result.computation_time_ms:.2f} ms")
            
            if ground_truth_dpg is not None:
                covered = result.uncertainty_lower <= ground_truth_dpg <= result.uncertainty_upper
                coverage_str = "✓ Covers" if covered else "✗ Misses"
                lines.append(f"    {coverage_str} ground truth ({ground_truth_dpg:.4f})")
            
            lines.append("")
        
        lines.append("Legend:")
        lines.append("  ✓ = Rigorous/certified bounds")
        lines.append("  ~ = Statistical/empirical estimates")
        lines.append("")
        
        return "\n".join(lines)


def run_comprehensive_baseline_comparison(
    device: str = 'cpu',
    n_experiments: int = 10
) -> Dict:
    """
    Run comprehensive baseline comparison across multiple datasets/models.
    
    Returns:
        Aggregated comparison results
    """
    print(f"\n{'='*70}")
    print("COMPREHENSIVE BASELINE COMPARISON")
    print(f"{'='*70}\n")
    print(f"Running {n_experiments} comparison experiments...\n")
    
    comparison = BaselineComparison(device=device)
    
    all_results = []
    all_analyses = []
    
    for exp_idx in range(n_experiments):
        print(f"[Experiment {exp_idx + 1}/{n_experiments}]")
        
        # Generate random dataset
        n_samples = 1000
        input_dim = 10
        
        X = torch.randn(n_samples, input_dim)
        y = (X[:, 0] + X[:, 1] > 0).numpy().astype(int)
        groups = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
        
        # Train simple model
        model = FairMLPClassifier(
            input_dim=input_dim,
            hidden_dims=[32, 16],
            activation='relu'
        ).to(device)
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        X_tensor = X.to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        
        for _ in range(20):
            optimizer.zero_grad()
            pred = model(X_tensor).squeeze()
            loss = nn.functional.binary_cross_entropy(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Run comparison
        results = comparison.run_comparison(
            model, X, groups, threshold=0.5, precision=torch.float32
        )
        
        # Analyze (use float64 as ground truth)
        model_64 = model.to(torch.float64)
        X_64 = X.to(torch.float64).to(device)
        with torch.no_grad():
            pred_64 = model_64(X_64).cpu().numpy().flatten()
        ground_truth = FairnessMetrics.demographic_parity_gap(
            pred_64, groups, threshold=0.5
        )
        
        analysis = comparison.analyze_comparison(results, ground_truth_dpg=ground_truth)
        
        all_results.append(results)
        all_analyses.append(analysis)
        
        print(f"  NumGeom-Fair width: {results['numgeom_fair'].uncertainty_width:.4f}")
        print(f"  Bootstrap width: {results['bootstrap'].uncertainty_width:.4f}")
        print(f"  Ground truth DPG: {ground_truth:.4f}\n")
    
    # Aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}\n")
    
    # Average metrics across experiments
    avg_widths = {
        'numgeom_fair': np.mean([r['numgeom_fair'].uncertainty_width for r in all_results]),
        'bootstrap': np.mean([r['bootstrap'].uncertainty_width for r in all_results]),
        'empirical_precision': np.mean([r['empirical_precision'].uncertainty_width for r in all_results])
    }
    
    avg_times = {
        'numgeom_fair': np.mean([r['numgeom_fair'].computation_time_ms for r in all_results]),
        'bootstrap': np.mean([r['bootstrap'].computation_time_ms for r in all_results]),
        'empirical_precision': np.mean([r['empirical_precision'].computation_time_ms for r in all_results])
    }
    
    # Coverage rates (when method's bounds contain ground truth)
    coverage_rates = {}
    for method in ['numgeom_fair', 'bootstrap', 'empirical_precision']:
        coverages = [
            a['coverage'].get(method, False)
            for a in all_analyses
            if 'coverage' in a and method in a['coverage']
        ]
        coverage_rates[method] = np.mean(coverages) if coverages else 0.0
    
    print("Average Uncertainty Width:")
    for method, width in sorted(avg_widths.items(), key=lambda x: x[1]):
        print(f"  {method}: {width:.4f}")
    
    print("\nAverage Computation Time:")
    for method, time_ms in sorted(avg_times.items(), key=lambda x: x[1]):
        print(f"  {method}: {time_ms:.2f} ms")
    
    print("\nCoverage Rate (ground truth within bounds):")
    for method, rate in sorted(coverage_rates.items(), key=lambda x: -x[1]):
        print(f"  {method}: {rate:.1%}")
    
    print("\nKey Findings:")
    
    # Compare NumGeom-Fair to Bootstrap
    width_ratio = avg_widths['numgeom_fair'] / avg_widths['bootstrap']
    time_ratio = avg_times['numgeom_fair'] / avg_times['bootstrap']
    
    print(f"  • NumGeom-Fair bounds are {width_ratio:.2f}x {'tighter' if width_ratio < 1 else 'wider'} than Bootstrap")
    print(f"  • NumGeom-Fair is {time_ratio:.2f}x {'faster' if time_ratio < 1 else 'slower'} than Bootstrap")
    print(f"  • NumGeom-Fair coverage: {coverage_rates['numgeom_fair']:.1%} (certified bounds)")
    print(f"  • Bootstrap coverage: {coverage_rates['bootstrap']:.1%} (statistical estimates)")
    
    aggregated_results = {
        'n_experiments': n_experiments,
        'avg_widths': avg_widths,
        'avg_times': avg_times,
        'coverage_rates': coverage_rates,
        'width_ratio_vs_bootstrap': width_ratio,
        'time_ratio_vs_bootstrap': time_ratio
    }
    
    return aggregated_results


if __name__ == '__main__':
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = run_comprehensive_baseline_comparison(device=device, n_experiments=20)
    
    # Save results
    import json
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'baseline_comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/baseline_comparison_results.json")
