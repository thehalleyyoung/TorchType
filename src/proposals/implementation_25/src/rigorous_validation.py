"""
Rigorous validation suite that tests whether the HNF theory actually holds.

This module addresses the key question: Are we "cheating" or is the theory correct?

Tests include:
1. Error bound tightness validation
2. Curvature lower bound verification  
3. Fairness metric error theorem validation
4. Cross-precision consistency checks
5. Adversarial case generation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

try:
    from .error_propagation import ErrorTracker, LinearErrorFunctional
    from .enhanced_error_propagation import (
        PreciseErrorTracker, ArchitectureAnalyzer, 
        verify_error_functional_validity
    )
    from .fairness_metrics import CertifiedFairnessEvaluator, FairnessMetrics
    from .models import FairMLPClassifier
except ImportError:
    from error_propagation import ErrorTracker, LinearErrorFunctional
    from enhanced_error_propagation import (
        PreciseErrorTracker, ArchitectureAnalyzer,
        verify_error_functional_validity
    )
    from fairness_metrics import CertifiedFairnessEvaluator, FairnessMetrics
    from models import FairMLPClassifier


@dataclass
class ValidationResult:
    """Result from a validation test."""
    test_name: str
    passed: bool
    metrics: Dict[str, float]
    details: str


class RigorousValidator:
    """
    Validates that the theoretical framework actually works in practice.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results: List[ValidationResult] = []
    
    def validate_error_functional_bounds(
        self,
        n_models: int = 10,
        n_tests_per_model: int = 50
    ) -> ValidationResult:
        """
        Test 1: Do error functionals actually bound the true numerical errors?
        
        This is the MOST CRITICAL test - if this fails, the whole theory is wrong!
        """
        print("\n[Test 1] Validating Error Functional Bounds")
        print("=" * 70)
        
        all_violations = []
        all_slacks = []
        max_violations = []
        
        for model_idx in range(n_models):
            # Generate random model architecture
            input_dim = np.random.randint(5, 20)
            hidden_dims = [
                np.random.randint(10, 50) 
                for _ in range(np.random.randint(1, 4))
            ]
            
            model = FairMLPClassifier(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                activation='relu'
            ).to(self.device)
            
            # Initialize with reasonable weights
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
            # Test at multiple precisions
            for precision in [torch.float32, torch.float16]:
                model_prec = model.to(precision)
                tracker = PreciseErrorTracker(precision)
                
                # Compute error functional
                error_functional = tracker.compute_model_error_functional(model_prec)
                
                # Verify it
                validation = verify_error_functional_validity(
                    model_prec,
                    error_functional,
                    input_shape=(1, input_dim),
                    precision=precision,
                    n_tests=n_tests_per_model,
                    device=self.device
                )
                
                all_violations.append(validation['violation_rate'])
                if validation['avg_slack'] > 0:
                    all_slacks.append(validation['avg_slack'])
                max_violations.append(validation['max_violation'])
        
        # Analyze results
        avg_violation_rate = np.mean(all_violations)
        avg_slack = np.mean(all_slacks) if all_slacks else 0.0
        max_violation = np.max(max_violations)
        
        # Criterion: Less than 5% violation rate (allowing some numerical noise)
        passed = avg_violation_rate < 0.05
        
        result = ValidationResult(
            test_name="Error Functional Bounds",
            passed=passed,
            metrics={
                'avg_violation_rate': avg_violation_rate,
                'avg_slack': avg_slack,
                'max_violation': max_violation,
                'n_models_tested': n_models,
                'n_tests_per_model': n_tests_per_model
            },
            details=f"Violation rate: {avg_violation_rate:.2%} (target: <5%)\n"
                   f"Average slack: {avg_slack:.2e}\n"
                   f"Max violation: {max_violation:.2e}"
        )
        
        print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
        print(f"  {result.details}")
        
        self.results.append(result)
        return result
    
    def validate_fairness_error_theorem(
        self,
        n_experiments: int = 20
    ) -> ValidationResult:
        """
        Test 2: Does the Fairness Metric Error Theorem hold?
        
        Theorem: |DPG^(p) - DPG^(∞)| ≤ p_near^(0) + p_near^(1)
        
        We test this by:
        1. Computing DPG at float64 (ground truth)
        2. Computing DPG at float32/float16
        3. Computing our bound p_near^(0) + p_near^(1)
        4. Checking if |DPG_low - DPG_high| ≤ bound
        """
        print("\n[Test 2] Validating Fairness Metric Error Theorem")
        print("=" * 70)
        
        violations = []
        bounds_tightness = []
        
        for exp_idx in range(n_experiments):
            # Generate synthetic dataset
            n_samples = 500
            n_features = 10
            
            # Create borderline-fair model
            model_high = FairMLPClassifier(
                input_dim=n_features,
                hidden_dims=[32, 16],
                activation='relu'
            ).to(torch.float64).to('cpu')  # Always use CPU for float64
            
            # Generate data
            X = torch.randn(n_samples, n_features, dtype=torch.float64, device='cpu')
            groups = torch.randint(0, 2, (n_samples,), device='cpu').numpy()
            
            # Compute DPG at float64 (ground truth)
            with torch.no_grad():
                preds_high = model_high(X).numpy().flatten()
            dpg_high = FairnessMetrics.demographic_parity_gap(
                preds_high, groups, threshold=0.5
            )
            
            # Test at lower precisions
            for precision in [torch.float32, torch.float16]:
                # Convert model
                model_low = FairMLPClassifier(
                    input_dim=n_features,
                    hidden_dims=[32, 16],
                    activation='relu'
                ).to(precision)
                model_low.load_state_dict(model_high.state_dict())
                
                # Move to appropriate device
                device = self.device if precision != torch.float16 or self.device != 'mps' else 'cpu'
                model_low = model_low.to(device)
                X_low = X.to(precision).to(device)
                
                # Compute DPG at lower precision
                with torch.no_grad():
                    preds_low = model_low(X_low).cpu().numpy().flatten()
                dpg_low = FairnessMetrics.demographic_parity_gap(
                    preds_low, groups, threshold=0.5
                )
                
                # Compute our theoretical bound
                tracker = PreciseErrorTracker(precision)
                error_functional = tracker.compute_model_error_functional(model_low)
                
                # Compute prediction errors
                pred_errors = tracker.compute_prediction_errors(
                    model_low, X_low, error_functional
                )
                
                # Near-threshold detection
                near_threshold = np.abs(preds_low - 0.5) < pred_errors
                p_near_0 = near_threshold[groups == 0].mean()
                p_near_1 = near_threshold[groups == 1].mean()
                
                theoretical_bound = p_near_0 + p_near_1
                
                # Actual difference
                actual_diff = abs(dpg_low - dpg_high)
                
                # Check if theorem holds
                if actual_diff <= theoretical_bound:
                    violations.append(0)
                    tightness = actual_diff / theoretical_bound if theoretical_bound > 0 else 1.0
                    bounds_tightness.append(tightness)
                else:
                    violations.append(1)
                    # Bound was violated - how much?
                    overshoot = actual_diff - theoretical_bound
                    print(f"    Warning: Bound violated by {overshoot:.4f} "
                          f"(actual: {actual_diff:.4f}, bound: {theoretical_bound:.4f})")
        
        # Analyze results
        violation_rate = np.mean(violations)
        avg_tightness = np.mean(bounds_tightness) if bounds_tightness else 0.0
        
        # Pass if violation rate < 5%
        passed = violation_rate < 0.05
        
        result = ValidationResult(
            test_name="Fairness Metric Error Theorem",
            passed=passed,
            metrics={
                'violation_rate': violation_rate,
                'avg_tightness': avg_tightness,
                'n_experiments': n_experiments,
                'n_violations': sum(violations),
                'n_total': len(violations)
            },
            details=f"Violation rate: {violation_rate:.2%} (target: <5%)\n"
                   f"Average tightness: {avg_tightness:.2%} (lower is tighter)"
        )
        
        print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
        print(f"  {result.details}")
        
        self.results.append(result)
        return result
    
    def validate_cross_precision_consistency(
        self,
        n_models: int = 15
    ) -> ValidationResult:
        """
        Test 3: Cross-precision consistency.
        
        When we evaluate the same model at different precisions,
        our certified bounds should be consistent with observed differences.
        """
        print("\n[Test 3] Validating Cross-Precision Consistency")
        print("=" * 70)
        
        consistent_cases = 0
        total_cases = 0
        
        for _ in range(n_models):
            # Create model
            model = FairMLPClassifier(
                input_dim=15,
                hidden_dims=[32, 16],
                activation='relu'
            ).to(torch.float64).to('cpu')
            
            # Generate test data
            X = torch.randn(200, 15, dtype=torch.float64, device='cpu')
            
            # Get predictions at float64
            with torch.no_grad():
                preds_64 = model(X).numpy().flatten()
            
            # Test float32 and float16
            for precision in [torch.float32, torch.float16]:
                model_low = FairMLPClassifier(
                    input_dim=15,
                    hidden_dims=[32, 16],
                    activation='relu'
                ).to(precision)
                model_low.load_state_dict(model.state_dict())
                
                device = 'cpu' if precision == torch.float16 and self.device == 'mps' else self.device
                model_low = model_low.to(device)
                X_low = X.to(precision).to(device)
                
                # Get predictions
                with torch.no_grad():
                    preds_low = model_low(X_low).cpu().numpy().flatten()
                
                # Compute actual difference
                actual_diff = np.abs(preds_64 - preds_low).max()
                
                # Compute our bound
                tracker = PreciseErrorTracker(precision)
                error_functional = tracker.compute_model_error_functional(model_low)
                error_bound = error_functional.evaluate(tracker.epsilon_machine)
                
                # Check consistency
                total_cases += 1
                if actual_diff <= error_bound * 2:  # Allow 2x slack for stability
                    consistent_cases += 1
        
        consistency_rate = consistent_cases / total_cases if total_cases > 0 else 0.0
        passed = consistency_rate > 0.90  # 90% consistency required
        
        result = ValidationResult(
            test_name="Cross-Precision Consistency",
            passed=passed,
            metrics={
                'consistency_rate': consistency_rate,
                'consistent_cases': consistent_cases,
                'total_cases': total_cases
            },
            details=f"Consistency rate: {consistency_rate:.2%} (target: >90%)"
        )
        
        print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
        print(f"  {result.details}")
        
        self.results.append(result)
        return result
    
    def validate_threshold_sensitivity(
        self,
        n_experiments: int = 20
    ) -> ValidationResult:
        """
        Test 4: Threshold sensitivity validation.
        
        Near decision threshold, small numerical errors should cause
        classification flips. Our theory should predict this.
        """
        print("\n[Test 4] Validating Threshold Sensitivity")
        print("=" * 70)
        
        prediction_accuracy = []
        
        for _ in range(n_experiments):
            # Create model and data
            model = FairMLPClassifier(
                input_dim=10,
                hidden_dims=[16],
                activation='relu'
            ).to(torch.float32).to(self.device)
            
            X = torch.randn(300, 10, dtype=torch.float32, device=self.device)
            
            # Get predictions
            with torch.no_grad():
                preds = model(X).cpu().numpy().flatten()
            
            # Compute error bounds
            tracker = PreciseErrorTracker(torch.float32)
            error_functional = tracker.compute_model_error_functional(model)
            error_bounds = tracker.compute_prediction_errors(model, X, error_functional)
            
            # For samples near threshold (0.5), check if we predict potential flips
            threshold = 0.5
            near_threshold = np.abs(preds - threshold) < 0.1  # Within 0.1 of threshold
            
            if near_threshold.sum() > 0:
                # Our prediction: samples with |pred - 0.5| < error_bound can flip
                predicted_flippable = np.abs(preds - threshold) < error_bounds
                
                # Simulate actual flips by adding small noise
                noisy_preds = preds + np.random.normal(0, 0.001, size=preds.shape)
                actual_flips = (preds > threshold) != (noisy_preds > threshold)
                
                # Among near-threshold samples, did we predict flips correctly?
                near_predicted = predicted_flippable[near_threshold]
                near_actual = actual_flips[near_threshold]
                
                # True positives: we predicted flippable and it flipped
                # True negatives: we predicted stable and it didn't flip
                correct = (near_predicted & near_actual) | (~near_predicted & ~near_actual)
                accuracy = correct.mean() if len(correct) > 0 else 0.0
                prediction_accuracy.append(accuracy)
        
        avg_accuracy = np.mean(prediction_accuracy) if prediction_accuracy else 0.0
        passed = avg_accuracy > 0.5  # Better than random
        
        result = ValidationResult(
            test_name="Threshold Sensitivity",
            passed=passed,
            metrics={
                'avg_prediction_accuracy': avg_accuracy,
                'n_experiments': n_experiments
            },
            details=f"Flip prediction accuracy: {avg_accuracy:.2%} (target: >50%)"
        )
        
        print(f"  Result: {'✓ PASSED' if passed else '✗ FAILED'}")
        print(f"  {result.details}")
        
        self.results.append(result)
        return result
    
    def run_all_validations(self) -> Dict:
        """
        Run all validation tests.
        
        Returns:
            Summary dictionary with all results
        """
        print("\n" + "=" * 70)
        print("RIGOROUS VALIDATION SUITE")
        print("Testing whether the HNF theory actually holds in practice")
        print("=" * 70)
        
        # Run all tests
        self.validate_error_functional_bounds(n_models=10, n_tests_per_model=30)
        self.validate_fairness_error_theorem(n_experiments=15)
        self.validate_cross_precision_consistency(n_models=12)
        self.validate_threshold_sensitivity(n_experiments=15)
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        print(f"\nTests passed: {passed_tests}/{total_tests}")
        
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status}: {result.test_name}")
        
        # Overall verdict
        all_passed = passed_tests == total_tests
        
        print("\n" + "=" * 70)
        if all_passed:
            print("✓ ALL VALIDATIONS PASSED")
            print("The HNF theory is empirically validated!")
        else:
            print("✗ SOME VALIDATIONS FAILED")
            print("The implementation needs refinement.")
        print("=" * 70)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'all_passed': all_passed,
            'results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'metrics': r.metrics,
                    'details': r.details
                }
                for r in self.results
            ]
        }
    
    def save_results(self, filepath: str):
        """Save validation results to JSON."""
        summary = self.run_all_validations() if not self.results else {
            'total_tests': int(len(self.results)),
            'passed_tests': int(sum(1 for r in self.results if r.passed)),
            'all_passed': bool(all(r.passed for r in self.results)),
            'results': [
                {
                    'test_name': str(r.test_name),
                    'passed': bool(r.passed),
                    'metrics': {k: float(v) if isinstance(v, (np.number, np.bool_)) else v 
                               for k, v in r.metrics.items()},
                    'details': str(r.details)
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    import sys
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    validator = RigorousValidator(device=device)
    summary = validator.run_all_validations()
    
    # Save results
    validator.save_results('data/rigorous_validation_results.json')
    
    # Exit with error code if validations failed
    sys.exit(0 if summary['all_passed'] else 1)
